import os
import sys
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cvrp_data_generator import CVRPDataGenerator
from cvrp_training_pipeline import CVRPTrainer


def simple_training_example():
    """Simple example showing the complete CVRP neural network pipeline"""
    
    print("=" * 60)
    print("CVRP Neural Network - Complete Pipeline Demo")
    print("=" * 60)
    
    # Step 1: Test data generation
    print("\n1. Testing Data Generation...")
    generator = CVRPDataGenerator()
    
    # Generate a small test dataset
    test_data = generator.generate_dataset(
        n_instances=5,
        n_customers=10,
        capacity_range=(50, 80),
        demand_range=(5, 15)
    )
    
    print(f"Generated {len(test_data)} instances")
    
    # Show first instance details
    instance = test_data[0]
    print(f"Instance 0: {instance.node_xy.shape[0]} customers, capacity={instance.capacity}")
    print(f"Optimal distance: {instance.optimal_distance:.2f}")
    print(f"Optimal routes: {instance.optimal_routes}")
    
    # Step 2: Model setup
    print("\n2. Setting up CVRP Model...")
    
    model_params = {
        'embedding_dim': 64,  # Smaller for demo
        'sqrt_embedding_dim': 8,
        'encoder_layer_num': 2,
        'qkv_dim': 8,
        'head_num': 4,
        'logit_clipping': 10,
        'ff_hidden_dim': 256,
        'eval_type': 'greedy',
        'debug_mode': True  # Enable debug for demo
    }
    
    training_params = {
        'learning_rate': 1e-3,
        'weight_decay': 1e-6,
        'lr_step_size': 10,
        'lr_gamma': 0.8,
        'use_cuda': torch.cuda.is_available()
    }
    
    trainer = CVRPTrainer(model_params, training_params)
    print(f"Model initialized on device: {trainer.device}")
    
    # Step 3: Quick training demo
    print("\n3. Quick Training Demo (5 epochs)...")
    
    # Generate small training set
    trainer.generate_training_data(
        n_train=20,
        n_val=5,
        n_customers=10
    )
    
    # Train for just a few epochs as demo
    try:
        train_losses, val_distances = trainer.train_supervised(
            epochs=5,
            batch_size=4
        )
        
        print(f"Training completed! Final loss: {train_losses[-1]:.4f}")
        
    except Exception as e:
        print(f"Training error: {e}")
        print("This is expected for demo - model needs more training")
    
    # Step 4: Test model inference
    print("\n4. Testing Model Inference...")
    
    try:
        # Test on first instance
        test_instance = test_data[0]
        predicted_distance = trainer._solve_instance(test_instance)
        
        print(f"Test instance - Optimal: {test_instance.optimal_distance:.2f}, Predicted: {predicted_distance:.2f}")
        gap = abs(predicted_distance - test_instance.optimal_distance) / test_instance.optimal_distance * 100
        print(f"Gap: {gap:.1f}%")
        
    except Exception as e:
        print(f"Inference error: {e}")
        print("This is expected - model needs proper training")
    
    # Step 5: Save/Load demo
    print("\n5. Model Save/Load Demo...")
    
    try:
        model_path = "demo_cvrp_model.pt"
        trainer.save_model(model_path)
        
        # Create new trainer and load
        new_trainer = CVRPTrainer(model_params, training_params)
        new_trainer.load_model(model_path)
        
        print("Model save/load successful!")
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
            
    except Exception as e:
        print(f"Save/load error: {e}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext Steps for Full Training:")
    print("1. Generate larger datasets (1000+ instances)")
    print("2. Train for more epochs (50-100)")
    print("3. Use benchmark validation")
    print("4. Tune hyperparameters")
    print("\nFor benchmark testing, place .vrp files in 'benchmark_data' folder")


def show_data_generation_details():
    """Show detailed data generation process"""
    
    print("\n" + "=" * 50)
    print("Data Generation Details")
    print("=" * 50)
    
    generator = CVRPDataGenerator()
    
    # Generate single instance with visualization
    print("Generating single instance with details...")
    
    dataset = generator.generate_dataset(
        n_instances=1,
        n_customers=15,
        capacity_range=(80, 120),
        demand_range=(5, 20)
    )
    instance = dataset[0]
    
    print("\nInstance Details:")
    print(f"- Customers: {instance.node_xy.shape[0]}")
    print(f"- Vehicle capacity: {instance.capacity}")
    print(f"- Total demand: {instance.demands.sum().item():.1f}")
    print(f"- Optimal distance: {instance.optimal_distance:.2f}")
    print(f"- Number of routes: {len(instance.optimal_routes)}")
    
    for i, route in enumerate(instance.optimal_routes):
        route_demand = sum(instance.demands[j-1].item() for j in route if j > 0)
        print(f"  Route {i+1}: {route} (demand: {route_demand:.1f})")
    
    print("\nCoordinates (first 5 customers):")
    print("Depot:", instance.depot_xy.numpy())
    for i in range(min(5, instance.node_xy.shape[0])):
        print(f"Customer {i+1}: {instance.node_xy[i].numpy()} (demand: {instance.demands[i].item():.1f})")


if __name__ == "__main__":
    # Run simple training example
    simple_training_example()
    
    # Show data generation details
    show_data_generation_details()