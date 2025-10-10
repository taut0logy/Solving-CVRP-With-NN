import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class CVRPModel(nn.Module):
    """CVRP Model adapted from VRPB - handles only delivery customers"""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.debug_mode = model_params.get('debug_mode', False)

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)
        
        if self.debug_mode:
            self.logger = logging.getLogger('cvrp_model')

    def pre_forward(self, reset_state):
        """Encode all nodes before decision making"""
        if self.debug_mode:
            self.logger.info("="*80)
            self.logger.info("CVRP MODEL PRE_FORWARD - ENCODING PHASE")
            self.logger.info("="*80)
        
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        
        if self.debug_mode:
            self.logger.info("Input shapes:")
            self.logger.info(f"  depot_xy: {depot_xy.shape}")
            self.logger.info(f"  node_xy: {node_xy.shape}")
            self.logger.info(f"  node_demand: {node_demand.shape}")
            
            # Log actual values for first batch
            self.logger.info("First batch values:")
            self.logger.info(f"  depot_xy[0]: {depot_xy[0].squeeze()}")
            self.logger.info(f"  node_xy[0]: {node_xy[0]}")
            self.logger.info(f"  node_demand[0]: {node_demand[0]}")
        
        # Combine coordinates and demands for customer nodes
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        # shape: (batch, problem, 3)
        
        if self.debug_mode:
            self.logger.info(f"Combined node_xy_demand shape: {node_xy_demand.shape}")

        # Encode all nodes (depot + customers)
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        # shape: (batch, problem+1, embedding)
        
        if self.debug_mode:
            self.logger.info(f"Encoded nodes shape: {self.encoded_nodes.shape}")
            self.logger.info("Pre-forward encoding complete!")
        
        # Set key-value pairs for decoder attention
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        """Make routing decision at current step"""
        if self.debug_mode:
            self.logger.info("="*80)
            self.logger.info(f"CVRP MODEL FORWARD - DECISION STEP {state.selected_count}")
            self.logger.info("="*80)
        
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)
        
        if self.debug_mode:
            self.logger.info(f"Batch size: {batch_size}, POMO size: {pomo_size}")
            self.logger.info(f"Current step: {state.selected_count}")

        # Initialize probs to None - will be set appropriately
        probs = None
        
        if state.selected_count == 0:  # First Move: Start at depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))
            # Create one-hot probability distribution for depot selection
            probs = torch.zeros(batch_size, pomo_size, self.encoded_nodes.shape[1])
            probs[:, :, 0] = 1.0  # 100% probability on depot (index 0)
            
            if self.debug_mode:
                self.logger.info("FIRST MOVE: All vehicles start at depot (node 0)")
                self.logger.info(f"Selected nodes: {selected}")

        elif state.selected_count == 1:  # Second Move: POMO initialization
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            # Create one-hot probability distributions for POMO starts
            probs = torch.zeros(batch_size, pomo_size, self.encoded_nodes.shape[1])
            for pomo_idx in range(pomo_size):
                if pomo_idx + 1 < self.encoded_nodes.shape[1]:  # Ensure valid customer index
                    probs[:, pomo_idx, pomo_idx + 1] = 1.0
            
            if self.debug_mode:
                self.logger.info("SECOND MOVE: POMO initialization - different starting customers")
                self.logger.info(f"Selected nodes: {selected}")
                self.logger.info(f"Each POMO rollout starts from customers 1 to {pomo_size}")

        else:  # Subsequent moves: Use attention mechanism
            if self.debug_mode:
                self.logger.info("SUBSEQUENT MOVE: Using attention mechanism")
                self.logger.info(f"Current nodes: {state.current_node}")
                self.logger.info(f"Current load: {state.load}")
                self.logger.info(f"Finished rollouts: {state.finished.sum().item()}/{batch_size * pomo_size}")
            
            # Get encoding of current node for each POMO rollout
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            
            if self.debug_mode:
                self.logger.info(f"Encoded last node shape: {encoded_last_node.shape}")
            
            # Decoder produces probability distribution over next nodes
            probs = self.decoder(encoded_last_node, state.load, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem+1)
            
            if self.debug_mode:
                self.logger.info(f"Decoder output probabilities shape: {probs.shape}")
                # Log probability distribution for first POMO
                for batch_idx in range(min(1, batch_size)):
                    for pomo_idx in range(min(2, pomo_size)):
                        prob_dist = probs[batch_idx, pomo_idx]
                        valid_probs = prob_dist[prob_dist > 1e-6]
                        self.logger.info(f"  Batch {batch_idx}, POMO {pomo_idx}: {len(valid_probs)} valid nodes, max_prob={prob_dist.max():.4f}")

            # Sample from probability distribution or use greedy selection
            if self.training or self.model_params['eval_type'] == 'softmax':
                # Stochastic sampling during training
                while True:  # Handle multinomial sampling edge cases
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break
                        
                if self.debug_mode:
                    self.logger.info("STOCHASTIC selection (multinomial):")
                    self.logger.info(f"  Selected nodes: {selected}")
                    self.logger.info(f"  Selection probabilities: {prob}")

            else:
                # Greedy selection during evaluation
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                
                if self.debug_mode:
                    self.logger.info("GREEDY selection (argmax):")
                    self.logger.info(f"  Selected nodes: {selected}")

        if self.debug_mode:
            self.logger.info("Forward pass complete!")
            self.logger.info("="*80)

        # Always return the full probability distribution for training
        return selected, probs


class CVRP_Encoder(nn.Module):
    """CVRP Encoder - Simplified from VRPB, no delivery/pickup separation"""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.debug_mode = model_params.get('debug_mode', False)
        self.embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        # Embedding layers
        self.embedding_depot = nn.Linear(2, self.embedding_dim)  # Depot: (x, y)
        self.embedding_node = nn.Linear(3, self.embedding_dim)   # Customer: (x, y, demand)
        
        # Self-attention layers (no cross-attention needed for CVRP)
        self.layers = nn.ModuleList([
            EncoderLayer(**model_params) for _ in range(encoder_layer_num)
        ])
        
        if self.debug_mode:
            self.logger = logging.getLogger('cvrp_encoder')

    def forward(self, depot_xy, node_xy_demand):
        """Encode depot and customer nodes"""
        if self.debug_mode:
            self.logger.info("="*60)
            self.logger.info("CVRP ENCODER FORWARD PASS")
            self.logger.info("="*60)
        
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)
        
        # Embed depot and customer nodes
        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand)
        # shape: (batch, problem, embedding)

        if self.debug_mode:
            self.logger.info("Initial embeddings:")
            self.logger.info(f"  depot embedding shape: {embedded_depot.shape}")
            self.logger.info(f"  node embedding shape: {embedded_node.shape}")
            self.logger.info(f"  depot embedding[0]: mean={embedded_depot[0].mean():.4f}, std={embedded_depot[0].std():.4f}")
            self.logger.info(f"  node embedding[0]: mean={embedded_node[0].mean():.4f}, std={embedded_node[0].std():.4f}")

        # Concatenate depot and customers
        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)
        
        if self.debug_mode:
            self.logger.info(f"Concatenated embeddings shape: {out.shape}")

        # Apply self-attention layers
        for i, layer in enumerate(self.layers):
            if self.debug_mode:
                self.logger.info(f"Running self-attention layer {i+1}")
            out = layer(out)

        if self.debug_mode:
            self.logger.info("Final encoder output:")
            self.logger.info(f"  shape: {out.shape}")
            self.logger.info(f"  mean: {out.mean():.4f}, std: {out.std():.4f}")
            self.logger.info("CVRP ENCODER COMPLETE")
            self.logger.info("="*60)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    """Self-attention encoder layer for CVRP"""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.debug_mode = model_params.get('debug_mode', False)
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # Multi-head attention
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # Layer normalization and feed-forward
        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)
        
        if self.debug_mode:
            self.logger = logging.getLogger('cvrp_encoder_layer')

    def forward(self, input_nodes):
        """Apply self-attention to all nodes"""
        head_num = self.model_params['head_num']
        
        if self.debug_mode:
            self.logger.info("  Self-attention on all nodes")

        # Multi-head attention
        q = reshape_by_heads(self.Wq(input_nodes), head_num=head_num)
        k = reshape_by_heads(self.Wk(input_nodes), head_num=head_num)
        v = reshape_by_heads(self.Wv(input_nodes), head_num=head_num)
        # qkv shape: (batch, head_num, problem+1, qkv_dim)
        
        if self.debug_mode:
            self.logger.info(f"    QKV shapes: {q.shape}")

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem+1, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem+1, embedding)

        # Add & norm + feed-forward + add & norm
        out1 = self.add_n_normalization_1(input_nodes, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)
        
        if self.debug_mode:
            self.logger.info(f"    Self-attention output mean: {out3.mean():.4f}, std: {out3.std():.4f}")

        return out3
        # shape: (batch, problem+1, embedding)


class CVRP_Decoder(nn.Module):
    """CVRP Decoder for next node selection"""

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.debug_mode = model_params.get('debug_mode', False)
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # Query generation from current node + load
        self.Wq_last = nn.Linear(embedding_dim+1, head_num * qkv_dim, bias=False)
        # Key and value from all nodes
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # Cached key-value pairs
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        
        if self.debug_mode:
            self.logger = logging.getLogger('cvrp_decoder')

    def set_kv(self, encoded_nodes):
        """Set key-value pairs from encoded nodes"""
        head_num = self.model_params['head_num']
        
        if self.debug_mode:
            self.logger.info(f"Setting decoder K,V from encoded nodes: {encoded_nodes.shape}")

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)
        
        if self.debug_mode:
            self.logger.info(f"  K shape: {self.k.shape}")
            self.logger.info(f"  V shape: {self.v.shape}")
            self.logger.info(f"  Single head key shape: {self.single_head_key.shape}")

    def forward(self, encoded_last_node, load, ninf_mask):
        """Generate probability distribution over next nodes"""
        if self.debug_mode:
            self.logger.info("="*50)
            self.logger.info("CVRP DECODER FORWARD")
            self.logger.info("="*50)
        
        head_num = self.model_params['head_num']
        
        if self.debug_mode:
            self.logger.info("Decoder inputs:")
            self.logger.info(f"  encoded_last_node shape: {encoded_last_node.shape}")
            self.logger.info(f"  load shape: {load.shape}")
            self.logger.info(f"  ninf_mask shape: {ninf_mask.shape}")
            self.logger.info(f"  load values: {load}")

        # Multi-Head Attention
        #######################################################
        # Combine current node encoding with current load
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        # shape = (batch, pomo, EMBEDDING_DIM+1)
        
        if self.debug_mode:
            self.logger.info(f"Concatenated input (node + load) shape: {input_cat.shape}")

        # Generate query from current state
        q = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        
        if self.debug_mode:
            self.logger.info(f"Query tensor shape: {q.shape}")

        # Multi-head attention over all nodes
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)
        
        if self.debug_mode:
            self.logger.info(f"Multi-head attention output shape: {mh_atten_out.shape}")

        # Single-Head Attention for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem+1)
        
        if self.debug_mode:
            self.logger.info(f"Raw attention scores shape: {score.shape}")
            self.logger.info(f"Raw scores range: [{score.min():.4f}, {score.max():.4f}]")

        # Scale and clip scores
        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        score_clipped = logit_clipping * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask

        if self.debug_mode:
            self.logger.info(f"After scaling: [{score_scaled.min():.4f}, {score_scaled.max():.4f}]")
            self.logger.info(f"After clipping: [{score_clipped.min():.4f}, {score_clipped.max():.4f}]")
            self.logger.info(f"After masking: [{score_masked.min():.4f}, {score_masked.max():.4f}]")

        # Convert to probabilities
        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem+1)
        
        if self.debug_mode:
            self.logger.info(f"Final probabilities shape: {probs.shape}")
            self.logger.info(f"Probability sum check: {probs.sum(dim=2)}")
            valid_nodes = (probs > 1e-6).sum(dim=2)
            self.logger.info(f"Valid nodes per rollout: {valid_nodes}")
            self.logger.info("CVRP DECODER COMPLETE")
            self.logger.info("="*50)

        return probs


# Helper functions (reused from VRPB model)
def _get_encoding(encoded_nodes, node_index_to_pick):
    """Extract encoding for specific nodes"""
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


def reshape_by_heads(qkv, head_num):
    """Reshape tensor for multi-head attention"""
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    """Multi-head attention computation"""
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    
    # Apply masks
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    return out_concat


class AddAndInstanceNormalization(nn.Module):
    """Add and instance normalization layer"""
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(1, 2)
        normalized = self.norm(transposed)
        back_trans = normalized.transpose(1, 2)
        return back_trans


class FeedForward(nn.Module):
    """Feed-forward layer"""
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))