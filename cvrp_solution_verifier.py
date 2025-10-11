#!/usr/bin/env python3
"""
CVRP Solution Verification Tool
==============================

A comprehensive tool for verifying CVRP solutions with multiple validation modes:
1. Constraint validation (capacity, coverage, node validity)
2. Distance calculation verification
3. Model performance analysis
4. Cross-scale comparison (normalized vs original coordinates)

Usage:
    python cvrp_solution_verifier.py --mode constraints --vrp data.vrp --solution sol.txt
    python cvrp_solution_verifier.py --mode performance --model model.pt --benchmark_dir benchmarks/
    python cvrp_solution_verifier.py --mode distance --vrp data.vrp --solution sol.txt
    python cvrp_solution_verifier.py --mode all --model model.pt --benchmark_dir benchmarks/
"""

import argparse
import os
import sys
import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from cvrp_training_pipeline import CVRPTrainer, CVRPInstance
    from cvrp_benchmark_parser import load_all_benchmarks
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class CVRPSolutionVerifier:
    """Comprehensive CVRP solution verification tool"""
    
    def __init__(self):
        self.results = {}
        self.verbose = True
    
    def parse_vrp_file(self, filepath: str) -> Dict:
        """Parse VRP file and extract problem data"""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"VRP file not found: {filepath}")
        
        dimension = None
        capacity = None
        demands = {}
        coordinates = {}
        depot = 1  # Default depot
        
        section = None
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('EOF'):
                continue
                
            # Parse header information
            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
            elif line.startswith('CAPACITY'):
                capacity = int(line.split(':')[1].strip())
            
            # Section markers
            elif line == 'NODE_COORD_SECTION':
                section = 'coordinates'
            elif line == 'DEMAND_SECTION':
                section = 'demands'
            elif line == 'DEPOT_SECTION':
                section = 'depot'
            
            # Parse section data
            elif section == 'coordinates':
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        node_id = int(parts[0])
                        x, y = float(parts[1]), float(parts[2])
                        coordinates[node_id] = (x, y)
                    except ValueError:
                        continue
            
            elif section == 'demands':
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        node_id = int(parts[0])
                        demand = int(parts[1])
                        demands[node_id] = demand
                    except ValueError:
                        continue
            
            elif section == 'depot':
                if line.startswith('-1'):
                    continue
                try:
                    depot = int(line)
                except ValueError:
                    continue
        
        if dimension is None or capacity is None:
            raise ValueError(f"Invalid VRP file: missing dimension or capacity")
        
        return {
            'dimension': dimension,
            'capacity': capacity,
            'demands': demands,
            'coordinates': coordinates,
            'depot': depot
        }
    
    def parse_solution_routes(self, filepath: str) -> List[List[int]]:
        """Parse solution routes from various formats"""
        routes = []
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Solution file not found: {filepath}")
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Handle "Route X: Depot -> ... -> Depot" format
            if line.lower().startswith('route') and ':' in line:
                after_colon = line.split(':', 1)[1]
                # Extract all integers from the line
                integers = re.findall(r'\b(\d+)\b', after_colon)
                if integers:
                    route = [int(x) for x in integers]
                    # Remove depot nodes (assuming depot is 1 or 0)
                    route = [node for node in route if node not in [0, 1] or 
                            (node == 1 and line.count('1') > 2)]  # Allow customer 1
                    if route:
                        routes.append(route)
            
            # Handle "Route #X: node1 node2 ..." format (.sol files)
            elif line.startswith('Route #'):
                route_part = line.split(':', 1)[1].strip()
                if route_part:
                    route = [int(x) for x in route_part.split()]
                    routes.append(route)
        
        return routes
    
    def verify_constraints(self, vrp_data: Dict, routes: List[List[int]]) -> Dict:
        """Verify CVRP constraints: capacity, coverage, validity"""
        errors = []
        warnings = []
        
        dimension = vrp_data['dimension']
        capacity = vrp_data['capacity']
        demands = vrp_data['demands']
        depot = vrp_data['depot']
        
        # Expected customers (all nodes except depot)
        expected_customers = set(range(1, dimension + 1)) - {depot}
        visited_customers = set()
        
        print(f"\n=== CONSTRAINT VERIFICATION ===")
        print(f"Problem: {dimension} nodes, capacity {capacity}, depot {depot}")
        print(f"Expected customers: {len(expected_customers)} ({min(expected_customers)}-{max(expected_customers)})")
        print(f"Found routes: {len(routes)}")
        
        route_stats = []
        for i, route in enumerate(routes, 1):
            route_load = 0
            route_customers = set()
            
            for node in route:
                # Check node validity
                if node < 1 or node > dimension:
                    errors.append(f"Route {i}: Invalid node {node} (valid range: 1-{dimension})")
                    continue
                
                # Check if depot appears as customer
                if node == depot:
                    errors.append(f"Route {i}: Depot node {depot} appears as customer")
                    continue
                
                # Check for duplicates within route
                if node in route_customers:
                    errors.append(f"Route {i}: Duplicate visit to node {node}")
                
                route_customers.add(node)
                
                # Check demand and capacity
                if node in demands:
                    route_load += demands[node]
                else:
                    warnings.append(f"Route {i}: No demand data for node {node}")
            
            # Check capacity constraint
            if route_load > capacity:
                errors.append(f"Route {i}: Capacity exceeded ({route_load} > {capacity})")
            
            visited_customers.update(route_customers)
            route_stats.append({
                'route': i,
                'nodes': len(route_customers),
                'load': route_load,
                'capacity_utilization': route_load / capacity * 100 if capacity > 0 else 0
            })
            
            print(f"  Route {i}: {len(route_customers)} customers, load {route_load}/{capacity} ({route_load/capacity*100:.1f}%)")
        
        # Check customer coverage
        missing_customers = expected_customers - visited_customers
        extra_customers = visited_customers - expected_customers
        
        if missing_customers:
            errors.append(f"Missing customers: {sorted(missing_customers)}")
        
        if extra_customers:
            errors.append(f"Extra customers (not in problem): {sorted(extra_customers)}")
        
        # Overall statistics
        total_customers = len(visited_customers)
        coverage_pct = total_customers / len(expected_customers) * 100 if expected_customers else 0
        
        is_valid = len(errors) == 0
        
        result = {
            'valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'statistics': {
                'total_routes': len(routes),
                'total_customers_visited': total_customers,
                'expected_customers': len(expected_customers),
                'coverage_percentage': coverage_pct,
                'route_details': route_stats
            }
        }
        
        # Print summary
        print(f"\nCoverage: {total_customers}/{len(expected_customers)} customers ({coverage_pct:.1f}%)")
        
        if is_valid:
            print("✅ VALID CVRP SOLUTION: All constraints satisfied")
        else:
            print("❌ INVALID CVRP SOLUTION:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        
        if warnings:
            print(f"⚠️  {len(warnings)} warnings")
        
        return result
    
    def calculate_distance(self, routes: List[List[int]], coordinates: Dict[int, Tuple[float, float]], 
                         depot: int = 1, normalize: bool = False) -> float:
        """Calculate total distance for routes"""
        if normalize:
            # Normalize coordinates to [0,1]
            coords_array = np.array(list(coordinates.values()))
            min_coord = coords_array.min(axis=0)
            max_coord = coords_array.max(axis=0)
            coord_range = max_coord - min_coord
            
            normalized_coords = {}
            for node_id, (x, y) in coordinates.items():
                x_norm = (x - min_coord[0]) / coord_range[0] if coord_range[0] > 0 else 0
                y_norm = (y - min_coord[1]) / coord_range[1] if coord_range[1] > 0 else 0
                normalized_coords[node_id] = (x_norm, y_norm)
            
            coordinates = normalized_coords
        
        total_distance = 0.0
        
        for route in routes:
            if not route:
                continue
            
            # Create full route: depot -> customers -> depot
            full_route = [depot] + route + [depot]
            
            route_distance = 0.0
            for i in range(len(full_route) - 1):
                from_node = full_route[i]
                to_node = full_route[i + 1]
                
                if from_node not in coordinates or to_node not in coordinates:
                    continue
                
                x1, y1 = coordinates[from_node]
                x2, y2 = coordinates[to_node]
                
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                route_distance += distance
            
            total_distance += route_distance
        
        return total_distance
    
    def verify_distance_calculation(self, vrp_file: str, solution_file: str) -> Dict:
        """Verify distance calculation on both original and normalized scales"""
        print(f"\n=== DISTANCE VERIFICATION ===")
        
        vrp_data = self.parse_vrp_file(vrp_file)
        routes = self.parse_solution_routes(solution_file)
        
        # Calculate on original scale
        original_distance = self.calculate_distance(routes, vrp_data['coordinates'], 
                                                  vrp_data['depot'], normalize=False)
        
        # Calculate on normalized scale
        normalized_distance = self.calculate_distance(routes, vrp_data['coordinates'], 
                                                    vrp_data['depot'], normalize=True)
        
        print(f"Routes: {len(routes)}")
        print(f"Original scale distance: {original_distance:.2f}")
        print(f"Normalized scale distance: {normalized_distance:.2f}")
        print(f"Scale ratio: {original_distance / normalized_distance:.2f}x" if normalized_distance > 0 else "N/A")
        
        return {
            'original_distance': original_distance,
            'normalized_distance': normalized_distance,
            'scale_ratio': original_distance / normalized_distance if normalized_distance > 0 else None
        }
    
    def verify_model_performance(self, model_path: str, benchmark_dir: str) -> Dict:
        """Comprehensive model performance verification"""
        print(f"\n=== MODEL PERFORMANCE VERIFICATION ===")
        
        # Load model
        try:
            model_params = {
                'embedding_dim': 128, 'sqrt_embedding_dim': 128 ** 0.5, 'encoder_layer_num': 3,
                'qkv_dim': 16, 'head_num': 8, 'logit_clipping': 10, 'ff_hidden_dim': 512,
                'eval_type': 'greedy', 'debug_mode': False
            }
            training_params = {
                'learning_rate': 1e-4, 'weight_decay': 1e-6, 'lr_step_size': 30,
                'lr_gamma': 0.5, 'use_cuda': False
            }
            
            trainer = CVRPTrainer(model_params, training_params)
            trainer.load_model(model_path)
            print(f"✅ Model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return {'error': str(e)}
        
        # Load benchmarks
        try:
            benchmarks = load_all_benchmarks(benchmark_dir)
            print(f"✅ Loaded {len(benchmarks)} benchmarks")
        except Exception as e:
            print(f"❌ Failed to load benchmarks: {e}")
            return {'error': str(e)}
        
        # Test each benchmark
        results = {}
        print(f"\n{'Instance':<15} {'Predicted':<10} {'Optimal':<10} {'Gap':<8} {'Status'}")
        print("-" * 55)
        
        for name, benchmark in benchmarks.items():
            try:
                neural_format = benchmark['neural_format']
                instance = CVRPInstance(
                    depot_xy=neural_format['depot_xy'],
                    node_xy=neural_format['node_xy'],
                    demands=neural_format['node_demand'] * neural_format['capacity'],
                    capacity=neural_format['capacity'],
                    optimal_routes=[], optimal_distance=0.0
                )
                
                predicted_distance, predicted_routes = trainer._solve_instance_detailed(instance)
                optimal_distance = benchmark['solution']['calculated_distance']
                gap = ((predicted_distance - optimal_distance) / optimal_distance * 100)
                
                # Verify predicted routes
                coords = torch.cat([instance.depot_xy, instance.node_xy], dim=0).cpu().numpy()
                manual_distance = self._calculate_manual_distance(predicted_routes, coords)
                distance_error = abs(predicted_distance - manual_distance)
                
                # Status assessment
                distance_ok = distance_error < 0.01
                gap_reasonable = -60 <= gap <= 30
                status = "✅ GOOD" if distance_ok and gap_reasonable else "⚠️ CHECK"
                
                if not distance_ok:
                    status = "❌ DIST_ERR"
                elif not gap_reasonable:
                    status = "⚠️ UNUSUAL"
                
                print(f"{name:<15} {predicted_distance:<10.2f} {optimal_distance:<10.2f} {gap:<+7.1f}% {status}")
                
                results[name] = {
                    'predicted_distance': predicted_distance,
                    'optimal_distance': optimal_distance,
                    'gap_percentage': gap,
                    'predicted_routes': predicted_routes,
                    'manual_distance': manual_distance,
                    'distance_calculation_error': distance_error,
                    'distance_calculation_ok': distance_ok,
                    'gap_reasonable': gap_reasonable,
                    'overall_status': status
                }
                
            except Exception as e:
                print(f"{name:<15} ERROR: {str(e)[:30]}")
                results[name] = {'error': str(e)}
        
        # Summary statistics
        valid_results = [r for r in results.values() if 'error' not in r]
        if valid_results:
            avg_gap = np.mean([r['gap_percentage'] for r in valid_results])
            print(f"\n📊 Average gap: {avg_gap:+.1f}%")
            print(f"📊 Successful tests: {len(valid_results)}/{len(results)}")
        
        return results
    
    def _calculate_manual_distance(self, routes: List[List[int]], coords: np.ndarray) -> float:
        """Calculate distance manually for verification"""
        total_distance = 0.0
        
        for route in routes:
            if not route:
                continue
            
            # Start at depot (index 0)
            prev_coord = coords[0]
            
            for customer in route:
                # Convert VRPLIB customer to coordinate index
                coord_idx = customer - 1
                if coord_idx < len(coords):
                    curr_coord = coords[coord_idx]
                    dist = np.sqrt(np.sum((curr_coord - prev_coord) ** 2))
                    total_distance += dist
                    prev_coord = curr_coord
            
            # Return to depot
            depot_coord = coords[0]
            final_dist = np.sqrt(np.sum((depot_coord - prev_coord) ** 2))
            total_distance += final_dist
        
        return total_distance
    
    def run_comprehensive_verification(self, args) -> Dict:
        """Run all verification modes"""
        all_results = {}
        
        if args.mode in ['constraints', 'all']:
            if args.vrp and args.solution:
                try:
                    vrp_data = self.parse_vrp_file(args.vrp)
                    routes = self.parse_solution_routes(args.solution)
                    all_results['constraints'] = self.verify_constraints(vrp_data, routes)
                except Exception as e:
                    all_results['constraints'] = {'error': str(e)}
            else:
                print("⚠️ Skipping constraint verification: need --vrp and --solution")
        
        if args.mode in ['distance', 'all']:
            if args.vrp and args.solution:
                try:
                    all_results['distance'] = self.verify_distance_calculation(args.vrp, args.solution)
                except Exception as e:
                    all_results['distance'] = {'error': str(e)}
            else:
                print("⚠️ Skipping distance verification: need --vrp and --solution")
        
        if args.mode in ['performance', 'all']:
            if args.model and args.benchmark_dir:
                try:
                    all_results['performance'] = self.verify_model_performance(args.model, args.benchmark_dir)
                except Exception as e:
                    all_results['performance'] = {'error': str(e)}
            else:
                print("⚠️ Skipping performance verification: need --model and --benchmark_dir")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive CVRP Solution Verification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--mode', choices=['constraints', 'distance', 'performance', 'all'],
                       default='all', help='Verification mode')
    parser.add_argument('--vrp', help='VRP problem file (.vrp)')
    parser.add_argument('--solution', help='Solution file (.txt or .sol)')
    parser.add_argument('--model', help='Trained model file (.pt)')
    parser.add_argument('--benchmark_dir', help='Directory containing benchmark files')
    parser.add_argument('--output', help='Output file for results (JSON)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['constraints', 'distance'] and (not args.vrp or not args.solution):
        parser.error(f"Mode '{args.mode}' requires --vrp and --solution")
    
    if args.mode == 'performance' and (not args.model or not args.benchmark_dir):
        parser.error("Mode 'performance' requires --model and --benchmark_dir")
    
    # Run verification
    verifier = CVRPSolutionVerifier()
    verifier.verbose = args.verbose
    
    print("🔍 CVRP Solution Verification Tool")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        results = verifier.run_comprehensive_verification(args)
        
        # Save results if requested
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {args.output}")
        
        print("\n✅ Verification complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())