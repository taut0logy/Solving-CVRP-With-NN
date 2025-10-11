import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cvrp_training_pipeline import CVRPTrainer
from cvrp_benchmark_parser import CVRPBenchmarkParser

def _save_instance_as_vrp(instance, filepath):
    """Save a CVRP instance in VRP format"""
    coordinates = instance.node_xy.numpy()
    demands = instance.demands.numpy()
    # Handle capacity whether it's a tensor or float
    if hasattr(instance.capacity, 'item'):
        capacity = int(instance.capacity.item())
    else:
        capacity = int(instance.capacity)
    
    with open(filepath, 'w') as f:
        f.write("NAME : sample_problem\n")
        f.write("COMMENT : Generated sample CVRP instance\n")
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {len(coordinates)}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"CAPACITY : {capacity}\n")
        f.write("NODE_COORD_SECTION\n")
        
        # Scale coordinates to 0-1000 range for standard VRP format
        coords_scaled = coordinates * 1000
        for i, (x, y) in enumerate(coords_scaled, 1):
            f.write(f"{i} {x:.0f} {y:.0f}\n")
        
        f.write("DEMAND_SECTION\n")
        for i, demand in enumerate(demands, 1):
            f.write(f"{i} {demand:.0f}\n")
        
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")
        f.write("EOF\n")


def _save_solution_file(routes, distance, filepath, problem_name="sample"):
    """Save solution in standard format"""
    with open(filepath, 'w') as f:
        f.write(f"Problem: {problem_name}\n")
        f.write(f"Distance: {distance:.2f}\n")
        f.write(f"Number of Routes: {len(routes)}\n")
        f.write("Routes:\n")
        for i, route in enumerate(routes, 1):
            route_str = " -> ".join([str(node) for node in route])
            f.write(f"Route {i}: Depot -> {route_str} -> Depot\n")


def test_on_benchmarks(enable_plots=True, tabulate=False):
    """Test trained model on benchmark instances"""
    
    print("=" * 60)
    print("CVRP Benchmark Testing")
    print("=" * 60)
    
    # Create output directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"benchmark_test/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
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
    results = trainer.test_on_benchmarks(benchmark_dir, save_plots=enable_plots, output_dir=output_dir)

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
            
            # Save predicted solution to file
            solution_file = os.path.join(output_dir, f"{name}_predicted_solution.txt")
            with open(solution_file, 'w') as f:
                f.write(f"Problem: {name}\n")
                f.write(f"Predicted Distance: {predicted:.2f}\n")
                if optimal is not None:
                    f.write(f"Optimal Distance: {optimal:.2f}\n")
                    f.write(f"Gap: {gap:.2f}%\n")
                f.write(f"Number of Routes: {len(predicted_routes)}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                
                f.write("Predicted Routes:\n")
                for i, route in enumerate(predicted_routes, 1):
                    route_str = " -> ".join([str(node) for node in route])
                    f.write(f"Route {i}: Depot -> {route_str} -> Depot\n")
                
                if optimal_routes:
                    f.write("\nOptimal Routes:\n")
                    for i, route in enumerate(optimal_routes, 1):
                        route_str = " -> ".join([str(node) for node in route])
                        f.write(f"Route {i}: Depot -> {route_str} -> Depot\n")
            
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
            
            print(f"  Solution saved to: {solution_file}")

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
    
    # Save summary report
    summary_file = os.path.join(output_dir, "benchmark_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("CVRP Benchmark Test Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Total instances tested: {len(results)}\n")
        f.write(f"Successful tests: {successful_tests}\n")
        if successful_tests > 0:
            f.write(f"Average gap: {total_gap / successful_tests:.2f}%\n")
        f.write("\nDetailed Results:\n")
        for name, result in results.items():
            if 'error' in result:
                f.write(f"{name}: ERROR - {result['error']}\n")
            else:
                predicted = result['predicted_distance']
                optimal = result['optimal_distance']
                gap = result['gap']
                if gap is not None:
                    f.write(f"{name}: Gap = {gap:.2f}% (Predicted: {predicted:.2f}, Optimal: {optimal:.2f})\n")
                else:
                    f.write(f"{name}: Predicted = {predicted:.2f} (No optimal solution available)\n")
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"All files saved in: {output_dir}")


def create_sample_instance():
    """Create a sample CVRP instance for testing"""
    
    print("\n" + "=" * 60)
    print("Creating Sample Test Instance")
    print("=" * 60)
    
    # Create sample test directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_dir = f"sample_test/{timestamp}"
    os.makedirs(sample_dir, exist_ok=True)
    print(f"Sample test directory: {sample_dir}")
    
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
    
    model_path = "trained_model/trained_cvrp_model.pt"
    if not os.path.exists(model_path):
        print("No trained model found. Using untrained model for demonstration...")
        trainer = CVRPTrainer(model_params, training_params)
    else:
        print("Loading trained model...")
        trainer = CVRPTrainer(model_params, training_params)
        trainer.load_model(model_path)
    
    from cvrp_data_generator import CVRPDataGenerator
    
    generator = CVRPDataGenerator()
    test_data = generator.generate_dataset(
        n_instances=1,
        n_customers=15,
        capacity_range=(80, 120),
        demand_range=(5, 20)
    )
    
    instance = test_data[0]

    problem_file = os.path.join(sample_dir, f"sample_problem_{timestamp}.vrp")
    _save_instance_as_vrp(instance, problem_file)
    
    print("\nTest Instance Details:")
    print(f"- Customers: {instance.node_xy.shape[0]}")
    print(f"- Vehicle capacity: {instance.capacity:.1f}")
    print(f"- Total demand: {instance.demands.sum().item():.1f}")
    print(f"- Optimal distance (ground truth): {instance.optimal_distance:.2f}")
    print(f"- Optimal routes: {instance.optimal_routes}")
    
    print("\nSolving with neural network...")
    predicted_distance, predicted_routes = trainer._solve_instance_detailed(instance)
    gap = abs(predicted_distance - instance.optimal_distance) / instance.optimal_distance * 100
    
    predicted_solution_file = os.path.join(sample_dir, f"predicted_solution_{timestamp}.txt")
    _save_solution_file(predicted_routes, predicted_distance, predicted_solution_file, f"sample_{timestamp}")
    
    optimal_solution_file = os.path.join(sample_dir, f"optimal_solution_{timestamp}.txt")
    _save_solution_file(instance.optimal_routes, instance.optimal_distance, optimal_solution_file, f"sample_{timestamp}_optimal")
    
    comparison_file = os.path.join(sample_dir, f"comparison_{timestamp}.txt")
    with open(comparison_file, 'w') as f:
        f.write("Sample CVRP Instance Test Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Problem file: sample_problem_{timestamp}.vrp\n")
        f.write(f"Customers: {instance.node_xy.shape[0] - 1}\n")
        f.write(f"Vehicle capacity: {instance.capacity:.1f}\n")
        f.write(f"Total demand: {instance.demands.sum().item():.1f}\n\n")
        
        f.write("OPTIMAL SOLUTION:\n")
        f.write(f"Distance: {instance.optimal_distance:.2f}\n")
        f.write(f"Routes ({len(instance.optimal_routes)}):\n")
        for i, route in enumerate(instance.optimal_routes, 1):
            route_str = " -> ".join([str(node) for node in route])
            f.write(f"  Route {i}: Depot -> {route_str} -> Depot\n")
        
        f.write("\nPREDICTED SOLUTION:\n")
        f.write(f"Distance: {predicted_distance:.2f}\n")
        f.write(f"Gap: {gap:.2f}%\n")
        f.write(f"Routes ({len(predicted_routes)}):\n")
        for i, route in enumerate(predicted_routes, 1):
            route_str = " -> ".join([str(node) for node in route])
            f.write(f"  Route {i}: Depot -> {route_str} -> Depot\n")
    
    # Create visualization if possible
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot optimal solution
        coordinates = instance.node_xy.numpy()
        ax1.scatter(coordinates[1:, 0], coordinates[1:, 1], c='blue', s=50, alpha=0.7, label='Customers')
        ax1.scatter(coordinates[0, 0], coordinates[0, 1], c='red', s=100, marker='s', label='Depot')
        
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, route in enumerate(instance.optimal_routes):
            color = colors[i % len(colors)]
            # Optimal routes use 0-based indexing (internal format)
            route_coords = [coordinates[0]] + [coordinates[node] for node in route] + [coordinates[0]]
            route_x = [coord[0] for coord in route_coords]
            route_y = [coord[1] for coord in route_coords]
            ax1.plot(route_x, route_y, c=color, linewidth=2, alpha=0.7)
        
        ax1.set_title(f'Optimal Solution\nDistance: {instance.optimal_distance:.2f}')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot predicted solution
        ax2.scatter(coordinates[1:, 0], coordinates[1:, 1], c='blue', s=50, alpha=0.7, label='Customers')
        ax2.scatter(coordinates[0, 0], coordinates[0, 1], c='red', s=100, marker='s', label='Depot')
        
        for i, route in enumerate(predicted_routes):
            color = colors[i % len(colors)]
            # Convert 1-based VRPLIB indices to 0-based for coordinate lookup
            route_coords = [coordinates[0]]  # Start at depot
            for node in route:
                if node <= len(coordinates):
                    route_coords.append(coordinates[node - 1])  # Convert to 0-based index
            route_coords.append(coordinates[0])  # Return to depot
            route_x = [coord[0] for coord in route_coords]
            route_y = [coord[1] for coord in route_coords]
            ax2.plot(route_x, route_y, c=color, linewidth=2, alpha=0.7)
        
        ax2.set_title(f'Predicted Solution\nDistance: {predicted_distance:.2f} (Gap: {gap:.1f}%)')
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(sample_dir, f"solution_comparison_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {plot_file}")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
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
    
    print(f"\nFiles saved in: {sample_dir}")
    print(f"- Problem: sample_problem_{timestamp}.vrp")
    print(f"- Predicted solution: predicted_solution_{timestamp}.txt")
    print(f"- Optimal solution: optimal_solution_{timestamp}.txt")
    print(f"- Comparison: comparison_{timestamp}.txt")


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