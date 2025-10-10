import os
import sys
import torch
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cvrp_model import CVRPModel
from cvrp_data_generator import CVRPDataGenerator, CVRPInstance
from cvrp_benchmark_parser import load_all_benchmarks
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from typing import List, Dict
import numpy as np


class CVRPTrainer:
    """CVRP Neural Network Trainer"""
    
    def __init__(self, model_params: Dict, training_params: Dict):
        self.model_params = model_params
        self.training_params = training_params
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('cvrp_trainer')
        
        # Initialize model
        self.model = CVRPModel(**model_params)
        
        # Setup optimizer and scheduler
        self.optimizer = Adam(
            self.model.parameters(),
            lr=training_params['learning_rate'],
            weight_decay=training_params.get('weight_decay', 1e-6)
        )
        
        self.scheduler = StepLR(
            self.optimizer,
            step_size=training_params.get('lr_step_size', 50),
            gamma=training_params.get('lr_gamma', 0.5)
        )
        
        # Training data
        self.training_data = None
        self.validation_data = None
        
        # Device setup (following VRPB pattern)
        USE_CUDA = training_params.get('use_cuda', True) and torch.cuda.is_available()
        if USE_CUDA:
            cuda_device_num = training_params.get('cuda_device_num', 0)
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
    
    def generate_training_data(self, n_train: int, n_val: int, n_customers: int):
        """Generate training and validation datasets"""
        self.logger.info(f"Generating training data: {n_train} train, {n_val} validation instances")
        
        generator = CVRPDataGenerator()
        
        # Generate training data
        self.training_data = generator.generate_dataset(
            n_instances=n_train,
            n_customers=n_customers,
            capacity_range=(80, 120),
            demand_range=(5, 15)
        )
        
        # Generate validation data
        self.validation_data = generator.generate_dataset(
            n_instances=n_val,
            n_customers=n_customers,
            capacity_range=(80, 120),
            demand_range=(5, 15)
        )
        
        self.logger.info("Training data generation complete!")
    
    def convert_instance_to_model_format(self, instance: CVRPInstance) -> Dict:
        """Convert CVRPInstance to model input format"""
        # Normalize demands by capacity for the model
        normalized_demands = instance.demands / instance.capacity
        
        return {
            'depot_xy': instance.depot_xy.to(self.device),
            'node_xy': instance.node_xy.to(self.device),
            'node_demand': normalized_demands.to(self.device),
            'capacity': instance.capacity
        }
    
    def create_route_target(self, routes: List[List[int]], n_customers: int) -> torch.Tensor:
        """Convert route list to sequence target for supervised learning"""
        # Create a sequence of next node selections
        sequence = [0]  # Start at depot
        
        for route in routes:
            if route:
                # Validate route nodes are within bounds
                valid_route = []
                for node in route:
                    if isinstance(node, (int, float)) and 1 <= node <= n_customers:
                        valid_route.append(int(node))
                    else:
                        self.logger.warning(f"Invalid node {node} in route, skipping")
                
                if valid_route:
                    sequence.extend(valid_route)
                    sequence.append(0)  # Return to depot after each route
        
        # Convert to tensor (limit to reasonable length)
        max_length = min(len(sequence), n_customers * 2)
        target_sequence = torch.zeros(max_length, dtype=torch.long, device=self.device)
        
        for i in range(max_length - 1):
            if i < len(sequence) - 1:
                next_node = sequence[i + 1]
                # Validate target node is within valid range [0, n_customers]
                if 0 <= next_node <= n_customers:
                    target_sequence[i] = next_node
                else:
                    self.logger.warning(f"Target node {next_node} out of bounds [0, {n_customers}], setting to 0")
                    target_sequence[i] = 0
        
        return target_sequence
    
    def train_supervised(self, epochs: int, batch_size: int = 32):
        """Train the model using supervised learning on ground truth solutions"""
        self.logger.info(f"Starting supervised training for {epochs} epochs")
        
        if not self.training_data:
            raise ValueError("No training data available. Call generate_training_data() first.")
        
        self.model.train()
        train_losses = []
        val_distances = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            # Shuffle training data
            np.random.shuffle(self.training_data)
            
            # Process in batches
            for i in range(0, len(self.training_data), batch_size):
                batch = self.training_data[i:i+batch_size]
                
                batch_loss = self._train_batch_supervised(batch)
                epoch_loss += batch_loss
                n_batches += 1
            
            avg_loss = epoch_loss / max(1, n_batches)
            train_losses.append(avg_loss)
            
            # Save checkpoint for this epoch
            self.save_checkpoint(epoch, avg_loss)
            
            # Validation
            if epoch % 5 == 0:
                val_distance = self._validate()
                val_distances.append(val_distance)
                
                self.logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Val Distance={val_distance:.2f}")
            else:
                self.logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
            
            # Update learning rate
            self.scheduler.step()
        
        return train_losses, val_distances
    
    def _train_batch_supervised(self, batch: List[CVRPInstance]) -> float:
        """Train on a single batch using supervised learning"""
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        batch_size = len(batch)
        
        for instance in batch:
            # Convert to model format
            model_input = self.convert_instance_to_model_format(instance)
            
            # Create target sequence
            target_sequence = self.create_route_target(
                instance.optimal_routes, 
                instance.node_xy.shape[0]
            )
            
            if len(target_sequence) < 2:
                continue
            
            # Debug: Log target sequence statistics
            if target_sequence.max().item() > instance.node_xy.shape[0]:
                self.logger.warning(f"Target sequence contains invalid nodes: max={target_sequence.max().item()}, n_customers={instance.node_xy.shape[0]}")
                self.logger.warning(f"Routes: {instance.optimal_routes}")
                continue
            
            if target_sequence.min().item() < 0:
                self.logger.warning(f"Target sequence contains negative nodes: min={target_sequence.min().item()}")
                continue
            
            # Simulate step-by-step prediction
            loss = 0.0
            current_pos = 0
            visited = set([0])  # Start with depot as visited
            used_load = 0.0  # Track used capacity
            
            # Mock environment state for the model
            device = self.device  # Capture device for inner class
            class MockState:
                def __init__(self, selected_count, current_node, used_capacity, n_nodes):
                    self.selected_count = max(selected_count, 2)  # Force attention mechanism (skip first/second moves)
                    self.current_node = torch.tensor([[current_node]], device=device)
                    self.load = torch.tensor([[1.0 - (used_capacity / instance.capacity)]], device=device)  # Remaining capacity (normalized)
                    self.ninf_mask = self._create_mask(visited, n_nodes, used_capacity, instance.demands, instance.capacity)
                    self.BATCH_IDX = torch.tensor([[0]], device=device)
                    self.POMO_IDX = torch.tensor([[0]], device=device)
                    self.finished = torch.tensor([[False]], device=device)
                    self.device = device  # Add device attribute
                
                def _create_mask(self, visited, n_nodes, used_capacity, demands, capacity):
                    mask = torch.zeros(1, 1, n_nodes + 1, device=device)
                    
                    # Mask visited nodes (except depot which can be revisited)
                    for v in visited:
                        if v != 0:  # Don't mask depot
                            mask[0, 0, v] = float('-inf')
                    
                    # Calculate remaining capacity
                    remaining_capacity = capacity - used_capacity
                    
                    # Mask nodes that exceed remaining capacity
                    for i in range(1, n_nodes + 1):
                        if i not in visited and demands[i-1] > remaining_capacity:
                            mask[0, 0, i] = float('-inf')
                    
                    return mask
            
            # Pre-forward pass (encoding)
            reset_state = type('obj', (object,), {
                'depot_xy': model_input['depot_xy'].unsqueeze(0),
                'node_xy': model_input['node_xy'].unsqueeze(0),
                'node_demand': model_input['node_demand'].unsqueeze(0)
            })
            
            self.model.pre_forward(reset_state)
            
            # Step through the target sequence
            for step in range(min(len(target_sequence) - 1, 20)):  # Limit steps
                state = MockState(step, current_pos, used_load, instance.node_xy.shape[0])
                
                # Get model prediction
                selected, probs = self.model(state)
                
                if probs is not None:
                    # Target is the next node in optimal sequence (already on correct device)
                    target_node_idx = step + 1
                    if target_node_idx < len(target_sequence):
                        target_node_val = target_sequence[target_node_idx].item()
                        
                        # Validate target node is within valid range for loss function
                        n_classes = probs.size(-1)  # Number of possible nodes (depot + customers)
                        if 0 <= target_node_val < n_classes:
                            target_node = target_sequence[target_node_idx].unsqueeze(0).unsqueeze(0)
                            
                            # Calculate cross-entropy loss
                            log_probs = torch.log(probs + 1e-8)
                            step_loss = F.nll_loss(
                                log_probs.view(-1, log_probs.size(-1)),
                                target_node.view(-1)
                            )
                            loss += step_loss
                        else:
                            self.logger.warning(f"Invalid target node {target_node_val} for {n_classes} classes, skipping loss")
                    else:
                        break  # End of sequence
                
                # Update state
                target_node_idx = step + 1
                if target_node_idx < len(target_sequence):
                    next_node = target_sequence[target_node_idx].item()
                    
                    # Validate next node is within bounds
                    if 0 <= next_node <= instance.node_xy.shape[0]:
                        if next_node == 0:  # Return to depot - reset load for new route
                            used_load = 0.0
                            # Don't clear visited set completely, just prepare for new route
                        else:
                            visited.add(next_node)
                            if next_node <= len(instance.demands):
                                used_load += instance.demands[next_node - 1].item()
                        
                        current_pos = next_node
                    else:
                        self.logger.warning(f"Invalid next node {next_node}, breaking sequence")
                        break
                else:
                    break  # End of sequence
            
            total_loss += loss
        
        avg_loss = total_loss / batch_size
        avg_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return avg_loss.item()
    
    def _validate(self) -> float:
        """Validate model on validation set"""
        if not self.validation_data:
            return 0.0
        
        self.model.eval()
        total_distance = 0.0
        n_instances = 0
        
        with torch.no_grad():
            for instance in self.validation_data[:10]:  # Validate on subset
                predicted_distance = self._solve_instance(instance)
                total_distance += predicted_distance
                n_instances += 1
        
        self.model.train()
        return total_distance / max(1, n_instances)
    
    def _solve_instance(self, instance: CVRPInstance) -> float:
        """Solve a single instance using the trained model"""
        distance, _ = self._solve_instance_detailed(instance)
        return distance
    
    def _solve_instance_detailed(self, instance: CVRPInstance) -> tuple[float, list]:
        """Solve a single instance using the trained model and return distance and routes"""
        model_input = self.convert_instance_to_model_format(instance)
        
        # Simple greedy decoding
        routes = []
        current_route = []
        current_load = instance.capacity
        visited = set()
        current_pos = 0
        
        reset_state = type('obj', (object,), {
            'depot_xy': model_input['depot_xy'].unsqueeze(0),
            'node_xy': model_input['node_xy'].unsqueeze(0),
            'node_demand': model_input['node_demand'].unsqueeze(0)
        })
        
        self.model.pre_forward(reset_state)
        
        max_steps = instance.node_xy.shape[0] * 2
        for step in range(max_steps):
            if len(visited) == instance.node_xy.shape[0]:
                break
            
            # Create state
            device = self.device  # Capture device for inner class
            class TestState:
                def __init__(self, selected_count, current_node, load_val, visited_set, n_nodes, demands, capacity):
                    self.selected_count = selected_count
                    self.current_node = torch.tensor([[current_node]], device=device) 
                    self.load = torch.tensor([[load_val / capacity]], device=device)
                    self.ninf_mask = self._create_mask(visited_set, n_nodes, load_val, demands, capacity)
                    self.BATCH_IDX = torch.tensor([[0]], device=device)
                    self.POMO_IDX = torch.tensor([[0]], device=device)
                    self.finished = torch.tensor([[False]], device=device)
                    self.device = device
                
                def _create_mask(self, visited_set, n_nodes, current_load, demands, capacity):
                    mask = torch.zeros(1, 1, n_nodes + 1, device=device)
                    
                    # Mask visited nodes
                    for v in visited_set:
                        mask[0, 0, v] = float('-inf')
                    
                    # Mask nodes that exceed capacity
                    for i in range(1, n_nodes + 1):
                        if i not in visited_set and demands[i-1] > current_load:
                            mask[0, 0, i] = float('-inf')
                    
                    # If no customers are feasible, allow depot
                    feasible_customers = False
                    for i in range(1, n_nodes + 1):
                        if mask[0, 0, i] != float('-inf'):
                            feasible_customers = True
                            break
                    
                    if not feasible_customers:
                        mask[0, 0, 0] = 0  # Allow depot
                    
                    return mask
            
            state = TestState(step, current_pos, current_load, visited, 
                            instance.node_xy.shape[0], instance.demands, instance.capacity)
            
            # Get next node
            selected, _ = self.model(state)
            next_node = selected[0, 0].item()
            
            if next_node == 0:  # Return to depot
                if current_route:
                    routes.append(current_route)
                    current_route = []
                current_load = instance.capacity
                current_pos = 0
            else:
                current_route.append(next_node)
                visited.add(next_node)
                current_load -= instance.demands[next_node - 1].item()
                current_pos = next_node
        
        if current_route:
            routes.append(current_route)
        
        # Calculate total distance
        total_distance = 0.0
        coords = torch.cat([instance.depot_xy, instance.node_xy], dim=0)
        
        for route in routes:
            if not route:
                continue
            
            route_coords = [coords[0]]  # Start at depot
            for customer in route:
                route_coords.append(coords[customer])
            route_coords.append(coords[0])  # Return to depot
            
            # Calculate route distance
            for i in range(len(route_coords) - 1):
                dist = torch.sqrt(torch.sum((route_coords[i] - route_coords[i+1]) ** 2))
                total_distance += dist.item()
        
        return total_distance, routes
    
    def save_model(self, filepath: str):
        """Save trained model"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_params': self.model_params,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def save_checkpoint(self, epoch: int, loss: float, checkpoint_dir: str = "trained_model"):
        """Save model checkpoint for a specific epoch"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{epoch+1}.pt")
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'model_params': self.model_params,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Only load optimizer state if it exists (for backward compatibility)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Model loaded from {filepath}")
        
        # Log additional checkpoint info if available
        if 'epoch' in checkpoint:
            self.logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            self.logger.info(f"Training loss at checkpoint: {checkpoint['loss']:.4f}")
    
    def load_checkpoint(self, epoch: int, checkpoint_dir: str = "trained_model"):
        """Load model from a specific epoch checkpoint"""
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{epoch}.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found: {checkpoint_path}")
        
        self.load_model(checkpoint_path)
        return checkpoint_path
    
    def plot_solution(self, instance_name: str, coordinates: torch.Tensor, 
                     predicted_routes: List[List[int]], optimal_routes: List[List[int]], 
                     predicted_distance: float, optimal_distance: float, 
                     save_dir: str):
        """Plot and save both predicted and optimal solutions for comparison"""
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Convert coordinates to numpy for plotting
        coords = coordinates.cpu().numpy() if torch.is_tensor(coordinates) else coordinates
        depot_coord = coords[0]  # First coordinate is depot
        customer_coords = coords[1:]  # Rest are customers
        
        # Define colors for routes
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(predicted_routes), len(optimal_routes))))
        
        # Plot predicted solution
        ax1.scatter(depot_coord[0], depot_coord[1], c='red', s=200, marker='s', 
                   label='Depot', zorder=5)
        ax1.scatter(customer_coords[:, 0], customer_coords[:, 1], c='blue', s=100, 
                   alpha=0.7, label='Customers', zorder=4)
        
        # Plot predicted routes
        for i, route in enumerate(predicted_routes):
            if not route:
                continue
            color = colors[i % len(colors)]
            
            # Create full route including depot
            full_route = [0] + route + [0]  # Start and end at depot
            route_coords = [coords[node] for node in full_route]
            
            # Plot route lines
            for j in range(len(route_coords) - 1):
                ax1.plot([route_coords[j][0], route_coords[j+1][0]], 
                        [route_coords[j][1], route_coords[j+1][1]], 
                        color=color, linewidth=2, alpha=0.7, zorder=2)
            
            # Add route label at midpoint
            if len(route_coords) > 2:
                mid_idx = len(route_coords) // 2
                ax1.text(route_coords[mid_idx][0], route_coords[mid_idx][1], f'R{i+1}', 
                        fontsize=10, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        # Add customer numbers
        for i, coord in enumerate(customer_coords):
            ax1.text(coord[0], coord[1], str(i+1), fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="circle,pad=0.1", facecolor='white', alpha=0.8))
        
        ax1.set_title(f'Predicted Solution\nRoutes: {len(predicted_routes)}, Distance: {predicted_distance:.2f}', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Plot optimal solution
        ax2.scatter(depot_coord[0], depot_coord[1], c='red', s=200, marker='s', 
                   label='Depot', zorder=5)
        ax2.scatter(customer_coords[:, 0], customer_coords[:, 1], c='blue', s=100, 
                   alpha=0.7, label='Customers', zorder=4)
        
        # Plot optimal routes
        for i, route in enumerate(optimal_routes):
            if not route:
                continue
            color = colors[i % len(colors)]
            
            # Create full route including depot
            full_route = [0] + route + [0]  # Start and end at depot
            route_coords = [coords[node] for node in full_route]
            
            # Plot route lines
            for j in range(len(route_coords) - 1):
                ax2.plot([route_coords[j][0], route_coords[j+1][0]], 
                        [route_coords[j][1], route_coords[j+1][1]], 
                        color=color, linewidth=2, alpha=0.7, zorder=2)
            
            # Add route label at midpoint
            if len(route_coords) > 2:
                mid_idx = len(route_coords) // 2
                ax2.text(route_coords[mid_idx][0], route_coords[mid_idx][1], f'R{i+1}', 
                        fontsize=10, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
        
        # Add customer numbers
        for i, coord in enumerate(customer_coords):
            ax2.text(coord[0], coord[1], str(i+1), fontsize=8, ha='center', va='center',
                    bbox=dict(boxstyle="circle,pad=0.1", facecolor='white', alpha=0.8))
        
        ax2.set_title(f'Optimal Solution\nRoutes: {len(optimal_routes)}, Distance: {optimal_distance:.2f}', 
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Add overall title with gap information
        gap = ((predicted_distance - optimal_distance) / optimal_distance * 100) if optimal_distance else 0
        fig.suptitle(f'{instance_name} - Gap: {gap:+.2f}%', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{instance_name}_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Solution comparison saved to {save_path}")
    
    def test_on_benchmarks(self, benchmark_dir: str, save_plots: bool = True) -> Dict:
        """Test trained model on benchmark instances"""
        self.logger.info(f"Testing on benchmarks from {benchmark_dir}")
        
        benchmarks = load_all_benchmarks(benchmark_dir)
        results = {}
        
        # Create timestamped directory for saving plots
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir = os.path.join("benchmark_test", timestamp)
            os.makedirs(plot_dir, exist_ok=True)
            self.logger.info(f"Saving plots to: {plot_dir}")
        
        self.model.eval()
        with torch.no_grad():
            for name, benchmark in benchmarks.items():
                try:
                    # Convert benchmark to our instance format
                    neural_format = benchmark['neural_format']
                    
                    instance = CVRPInstance(
                        depot_xy=neural_format['depot_xy'],
                        node_xy=neural_format['node_xy'],
                        demands=neural_format['node_demand'] * neural_format['capacity'],  # Denormalize
                        capacity=neural_format['capacity'],
                        optimal_routes=[],  # Not needed for testing
                        optimal_distance=0.0
                    )
                    
                    predicted_distance, predicted_routes = self._solve_instance_detailed(instance)
                    optimal_distance = benchmark['solution']['calculated_distance'] if benchmark['solution'] else None
                    optimal_routes = benchmark['solution']['routes'] if benchmark['solution'] else None
                    
                    results[name] = {
                        'predicted_distance': predicted_distance,
                        'predicted_routes': predicted_routes,
                        'optimal_distance': optimal_distance,
                        'optimal_routes': optimal_routes,
                        'gap': ((predicted_distance - optimal_distance) / optimal_distance * 100) if optimal_distance else None
                    }
                    
                    # Generate solution visualization if requested
                    if save_plots and optimal_routes is not None:
                        # Get original coordinates (not normalized) for plotting
                        original_coords = benchmark.get('original_coordinates')
                        if original_coords is not None:
                            self.plot_solution(
                                instance_name=name,
                                coordinates=original_coords,
                                predicted_routes=predicted_routes,
                                optimal_routes=optimal_routes,
                                predicted_distance=predicted_distance,
                                optimal_distance=optimal_distance,
                                save_dir=plot_dir
                            )
                    
                    if optimal_distance is not None:
                        self.logger.info(f"{name}: Predicted={predicted_distance:.2f}, Optimal={optimal_distance:.2f}")
                    else:
                        self.logger.info(f"{name}: Predicted={predicted_distance:.2f}, Optimal=N/A")
                    
                except Exception as e:
                    self.logger.error(f"Error testing {name}: {e}")
                    results[name] = {'error': str(e)}
        
        return results


def main():
    """Main training pipeline"""
    
    # Model parameters
    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** 0.5,
        'encoder_layer_num': 3,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'greedy',
        'debug_mode': False
    }
    
    # Training parameters
    training_params = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'lr_step_size': 30,
        'lr_gamma': 0.5,
        'use_cuda': True
    }
    
    # Create trainer
    trainer = CVRPTrainer(model_params, training_params)
    
    # Generate training data
    trainer.generate_training_data(
        n_train=1000,
        n_val=100,
        n_customers=20
    )
    
    # Train model
    print("Starting training...")
    train_losses, val_distances = trainer.train_supervised(
        epochs=50,
        batch_size=16
    )
    
    # Save model
    trainer.save_model("cvrp_trained_model.pt")
    
    # Test on benchmarks
    benchmark_results = trainer.test_on_benchmarks("benchmark_data")
    
    print("\nBenchmark Results:")
    for name, result in benchmark_results.items():
        if 'error' not in result and result['gap'] is not None:
            print(f"{name}: Gap = {result['gap']:.2f}%")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()