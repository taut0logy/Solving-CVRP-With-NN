import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cvrp_training_pipeline import CVRPTrainer
from cvrp_benchmark_parser import CVRPBenchmarkParser


def test_on_benchmarks(enable_plots=True, tabulate=False):
    """Test trained model on benchmark instances"""
    
    print("=" * 60)
    print("CVRP Benchmark Testing")
    print("=" * 60)
    
    # Load trained model
    model_path = "trained_model/trained_cvrp_model.pt"
    if not os.path.exists(model_path):
        # Fallback to old location for backward compatibility
        model_path = "trained_cvrp_model.pt"
        if not os.path.exists(model_path):
            print("Error: Trained model not found in 'trained_model/' or current directory!")
            print("Please run 'python train_cvrp.py' first to train a model.")
            return
    
    # Model parameters (must match training)
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
    
    training_params = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'lr_step_size': 30,
        'lr_gamma': 0.5,
        'use_cuda': torch.cuda.is_available(),
        'cuda_device_num': 0
    }
    
    # Create trainer and load model
    print("Loading trained model...")
    trainer = CVRPTrainer(model_params, training_params)
    trainer.load_model(model_path)
    print("Model loaded successfully!")
    
    # Check for benchmark data
    benchmark_dir = "benchmark_data"
    if not os.path.exists(benchmark_dir):
        print(f"\nCreating benchmark directory: {benchmark_dir}")
        os.makedirs(benchmark_dir)
        print("Please place .vrp benchmark files in the 'benchmark_data' folder")
        print("You can download benchmark instances from: http://vrp.atd-lab.inf.puc-rio.br/index.php/en/")
        return
    
    # Check if benchmark files exist
    vrp_files = [f for f in os.listdir(benchmark_dir) if f.endswith('.vrp')]
    if not vrp_files:
        print(f"\nNo .vrp files found in {benchmark_dir}/")
        print("Please download some benchmark instances and place them in the benchmark_data folder")
        print("\nSuggested benchmarks:")
        print("- A-n32-k5.vrp (32 customers)")
        print("- A-n45-k6.vrp (45 customers)")
        print("- B-n31-k5.vrp (31 customers)")
        return
    
    print(f"\nFound {len(vrp_files)} benchmark files:")
    for f in vrp_files:
        print(f"  - {f}")
    
    # Test on benchmarks
    print("\nTesting model on benchmarks...")
    results = trainer.test_on_benchmarks(benchmark_dir, save_plots=enable_plots)

    # If tabulate requested, collect benchmark metadata to build the table
    benchmark_meta = {}
    if tabulate:
        parser = CVRPBenchmarkParser()
        # Load per-file metadata without the noisy prints
        for f in vrp_files:
            vrp_path = os.path.join(benchmark_dir, f)
            sol_path = os.path.splitext(vrp_path)[0] + '.sol'
            sol_file = sol_path if os.path.exists(sol_path) else None
            try:
                inst = parser.load_benchmark_instance(vrp_path, sol_file)
                name = os.path.splitext(f)[0]
                benchmark_meta[name] = inst
            except Exception:
                # If parsing fails, skip metadata for this instance
                continue
    
    # Display results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    successful_tests = 0
    total_gap = 0
    
    for name, result in results.items():
        if 'error' in result:
            print(f"{name}: ERROR - {result['error']}")
        else:
            predicted = result['predicted_distance']
            optimal = result['optimal_distance']
            gap = result['gap']
            predicted_routes = result.get('predicted_routes', [])
            optimal_routes = result.get('optimal_routes', [])
            
            print(f"\n{name}:")
            if gap is not None:
                print(f"  Gap = {gap:.2f}% (Predicted: {predicted:.2f}, Optimal: {optimal:.2f})")
                total_gap += gap
                successful_tests += 1
            else:
                print(f"  Predicted = {predicted:.2f} (No optimal solution available)")
            
            # Display predicted routes
            if predicted_routes:
                print(f"  Predicted Routes ({len(predicted_routes)} routes):")
                for i, route in enumerate(predicted_routes, 1):
                    route_str = " -> ".join([str(node) for node in route])
                    print(f"    Route {i}: Depot -> {route_str} -> Depot")
            
            # Display optimal routes if available
            if optimal_routes:
                print(f"  Optimal Routes ({len(optimal_routes)} routes):")
                for i, route in enumerate(optimal_routes, 1):
                    route_str = " -> ".join([str(node) for node in route])
                    print(f"    Route {i}: Depot -> {route_str} -> Depot")

    # Print tabulated Markdown table if requested
    if tabulate:
        print("\nBenchmark table: \n")
        header = "| Instance | Customers | Optimal Routes | Predicted Routes | Gap | Performance |"
        sep = "|----------|-----------:|---------------:|-----------------:|-----:|-------------|"
        print(header)
        print(sep)

        def perf_from_gap(g):
            if g is None:
                return 'N/A'
            try:
                g = float(g)
            except Exception:
                return 'N/A'
            if g <= -50:
                return 'Much better (lower distance)'
            elif g <= -40:
                return 'Much better'
            elif g <= -20:
                return 'Better'
            elif g <= 10:
                return 'Good'
            elif g <= 20:
                return 'Decent'
            else:
                return 'Needs improvement'

        total_gap = 0.0
        gap_count = 0
        for name, result in results.items():
            meta = benchmark_meta.get(name)
            customers = meta['problem']['dimension'] if meta else '-'
            optimal_routes_count = len(meta['solution']['routes']) if (meta and meta.get('solution')) else '-'
            predicted_routes_count = len(result.get('predicted_routes', [])) if result.get('predicted_routes') is not None else '-'
            gap = result.get('gap')
            gap_str = f"{gap:.2f}%" if gap is not None else 'N/A'
            perf = perf_from_gap(gap)

            print(f"| {name} | {customers} | {optimal_routes_count} | {predicted_routes_count} | {gap_str} | {perf} |")

            if gap is not None:
                total_gap += gap
                gap_count += 1

        if gap_count:
            avg_gap = total_gap / gap_count
            print(f"| **Average** | - | - | - | **{avg_gap:.2f}%** | **{ 'Excellent' if avg_gap <= 10 else ('Good' if avg_gap <=20 else 'Needs improvement') }** |")
    
    if successful_tests > 0:
        avg_gap = total_gap / successful_tests
        print("\nSUMMARY:")
        print(f"Successful tests: {successful_tests}/{len(results)}")
        print(f"Average gap: {avg_gap:.2f}%")
        
        # Performance assessment
        if avg_gap <= 10:
            print("🎉 EXCELLENT performance!")
        elif avg_gap <= 20:
            print("✅ GOOD performance!")
        elif avg_gap <= 30:
            print("📈 DECENT performance - consider more training")
        else:
            print("🔄 NEEDS IMPROVEMENT - try longer training or larger datasets")
    else:
        print("No successful benchmark tests completed")


def create_sample_instance():
    """Create a sample CVRP instance for testing"""
    
    print("\n" + "=" * 60)
    print("Creating Sample Test Instance")
    print("=" * 60)
    
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
        'debug_mode': True  # Enable debug for detailed output
    }
    
    training_params = {
        'learning_rate': 1e-4,
        'weight_decay': 1e-6,
        'use_cuda': torch.cuda.is_available(),
        'cuda_device_num': 0
    }
    
    # Check if model exists
    model_path = "trained_model/trained_cvrp_model.pt"
    if not os.path.exists(model_path):
        # Fallback to old location
        model_path = "trained_cvrp_model.pt"
        if not os.path.exists(model_path):
            print("No trained model found. Using untrained model for demonstration...")
            trainer = CVRPTrainer(model_params, training_params)
        else:
            print("Loading trained model...")
            trainer = CVRPTrainer(model_params, training_params)
            trainer.load_model(model_path)
    else:
        print("Loading trained model...")
        trainer = CVRPTrainer(model_params, training_params)
        trainer.load_model(model_path)
    
    # Generate a test instance
    from cvrp_data_generator import CVRPDataGenerator
    
    generator = CVRPDataGenerator()
    test_data = generator.generate_dataset(
        n_instances=1,
        n_customers=15,
        capacity_range=(80, 120),
        demand_range=(5, 20)
    )
    
    instance = test_data[0]
    
    print("\nTest Instance Details:")
    print(f"- Customers: {instance.node_xy.shape[0]}")
    print(f"- Vehicle capacity: {instance.capacity:.1f}")
    print(f"- Total demand: {instance.demands.sum().item():.1f}")
    print(f"- Optimal distance (ground truth): {instance.optimal_distance:.2f}")
    print(f"- Optimal routes: {instance.optimal_routes}")
    
    # Solve with neural network
    print("\nSolving with neural network...")
    predicted_distance, predicted_routes = trainer._solve_instance_detailed(instance)
    gap = abs(predicted_distance - instance.optimal_distance) / instance.optimal_distance * 100
    
    print("Neural network solution:")
    print(f"- Predicted distance: {predicted_distance:.2f}")
    print(f"- Gap from optimal: {gap:.1f}%")
    print(f"- Predicted routes ({len(predicted_routes)} routes):")
    for i, route in enumerate(predicted_routes, 1):
        route_str = " -> ".join([str(node) for node in route])
        print(f"  Route {i}: Depot -> {route_str} -> Depot")
    
    if gap < 15:
        print("🎉 Excellent solution!")
    elif gap < 30:
        print("✅ Good solution!")
    else:
        print("📈 Room for improvement")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CVRP model on benchmarks')
    parser.add_argument('--sample', action='store_true', help='Test on a sample instance instead')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting and saving visualizations')
    parser.add_argument('--tabulate', action='store_true', help='Print a markdown table summary of benchmark results')
    args = parser.parse_args()
    
    if args.sample:
        create_sample_instance()
    else:
        enable_plots = not args.no_plots
        test_on_benchmarks(enable_plots, tabulate=args.tabulate)