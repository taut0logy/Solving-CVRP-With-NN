import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cvrp_training_pipeline import CVRPTrainer


def train_cvrp_model(epochs=30, batch_size=16):
    """Full training process for CVRP neural network"""
    
    print("=" * 60)
    print("CVRP Neural Network - Full Training")
    print("=" * 60)
    
    # Model parameters (you can adjust these)
    model_params = {
        'embedding_dim': 128,           # Embedding dimension
        'sqrt_embedding_dim': 128 ** 0.5,
        'encoder_layer_num': 3,         # Number of encoder layers
        'qkv_dim': 16,                  # Query/Key/Value dimension
        'head_num': 8,                  # Number of attention heads
        'logit_clipping': 10,           # Logit clipping value
        'ff_hidden_dim': 512,           # Feed-forward hidden dimension
        'eval_type': 'greedy',          # Evaluation type
        'debug_mode': False             # Set True for detailed logging
    }
    
    # Training parameters
    training_params = {
        'learning_rate': 1e-4,          # Learning rate
        'weight_decay': 1e-6,           # Weight decay
        'lr_step_size': 30,             # LR scheduler step size
        'lr_gamma': 0.5,                # LR decay factor
        'use_cuda': torch.cuda.is_available(),
        'cuda_device_num': 0            # CUDA device number (0 = first GPU)
    }
    
    print(f"Using device: {'CUDA' if training_params['use_cuda'] else 'CPU'}")
    
    # Create trainer
    trainer = CVRPTrainer(model_params, training_params)
    
    # Step 1: Generate training data
    print("\n1. Generating Training Data...")
    print("This may take several minutes for large datasets...")
    
    trainer.generate_training_data(
        n_train=500,        # Number of training instances
        n_val=50,           # Number of validation instances  
        n_customers=20      # Number of customers per instance
    )
    
    print("Training data generation complete!")
    
    # Step 2: Train the model
    print("\n2. Training Model...")
    print("Training for 30 epochs with batch size 16...")
    
    train_losses, val_distances = trainer.train_supervised(
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Step 3: Save trained model
    model_path = "trained_model/trained_cvrp_model.pt"
    trainer.save_model(model_path)
    print(f"\nFinal model saved to: {model_path}")
    print("All epoch checkpoints saved in: trained_model/checkpoint-{epoch}.pt")
    
    # Step 4: Plot training progress
    print("\n3. Plotting Training Progress...")
    plot_training_progress(train_losses, val_distances)
    
    # Step 5: Test on some instances
    print("\n4. Testing Trained Model...")
    test_cvrp_model(trainer)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Trained model saved as: {model_path}")
    print("Next step: Test on benchmark datasets!")


def plot_training_progress(train_losses, val_distances):
    """Plot training progress"""
    
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot validation distances
    if val_distances:
        plt.subplot(1, 2, 2)
        epochs = list(range(0, len(train_losses), 5))[:len(val_distances)]
        plt.plot(epochs, val_distances, 'r-o', linewidth=2, label='Validation Distance')
        plt.xlabel('Epoch')
        plt.ylabel('Average Distance')
        plt.title('Validation Performance')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training progress plot saved as 'training_progress.png'")


def test_cvrp_model(trainer):
    """Test the trained model on some instances"""
    
    print("Testing model on validation instances...")
    
    if not trainer.validation_data:
        print("No validation data available")
        return
    
    # Test on first 5 validation instances
    total_gap = 0
    valid_tests = 0
    
    for i, instance in enumerate(trainer.validation_data[:5]):
        try:
            predicted_distance = trainer._solve_instance(instance)
            optimal_distance = instance.optimal_distance
            
            gap = abs(predicted_distance - optimal_distance) / optimal_distance * 100
            total_gap += gap
            valid_tests += 1
            
            print(f"Instance {i+1}: Optimal={optimal_distance:.2f}, Predicted={predicted_distance:.2f}, Gap={gap:.1f}%")
            
        except Exception as e:
            print(f"Instance {i+1}: Error - {e}")
    
    if valid_tests > 0:
        avg_gap = total_gap / valid_tests
        print(f"\nAverage gap: {avg_gap:.1f}%")
    else:
        print("No successful tests")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CVRP model")
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    args = parser.parse_args()

    train_cvrp_model(epochs=args.epochs, batch_size=args.batch_size)