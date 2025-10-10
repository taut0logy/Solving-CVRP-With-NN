import torch
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CVRPBenchmarkParser:
    """Parse CVRP benchmark files in VRPLIB format"""
    
    def __init__(self):
        self.problem_data = {}
        self.solution_data = {}
    
    def parse_vrp_file(self, filepath: str) -> Dict:
        """Parse a .vrp file and extract problem information"""
        problem = {
            'name': '',
            'dimension': 0,
            'capacity': 0,
            'coordinates': [],
            'demands': [],
            'depot': 0
        }
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header information
        for line in lines:
            line = line.strip()
            if line.startswith('NAME'):
                problem['name'] = line.split(':')[1].strip()
            elif line.startswith('DIMENSION'):
                problem['dimension'] = int(line.split(':')[1].strip())
            elif line.startswith('CAPACITY'):
                problem['capacity'] = int(line.split(':')[1].strip())
            elif line.startswith('NODE_COORD_SECTION'):
                break
        
        # Parse coordinates
        coord_started = False
        demand_started = False
        depot_started = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('NODE_COORD_SECTION'):
                coord_started = True
                continue
            elif line.startswith('DEMAND_SECTION'):
                coord_started = False
                demand_started = True
                continue
            elif line.startswith('DEPOT_SECTION'):
                demand_started = False
                depot_started = True
                continue
            elif line.startswith('EOF'):
                break
            
            if coord_started and line:
                parts = line.split()
                if len(parts) >= 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    problem['coordinates'].append((node_id, x, y))
            
            elif demand_started and line:
                parts = line.split()
                if len(parts) >= 2:
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    problem['demands'].append((node_id, demand))
            
            elif depot_started and line and not line.startswith('-1'):
                problem['depot'] = int(line)
        
        return problem
    
    def parse_solution_file(self, filepath: str) -> Dict:
        """Parse a .sol file and extract solution routes"""
        solution = {
            'routes': [],
            'total_cost': 0
        }
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith('Route #'):
                    # Extract route
                    route_part = line.split(':', 1)[1].strip()
                    if route_part:
                        route = [int(x) for x in route_part.split()]
                        solution['routes'].append(route)
                elif line.startswith('Cost'):
                    # Extract total cost if available
                    cost_match = re.search(r'(\d+(?:\.\d+)?)', line)
                    if cost_match:
                        solution['total_cost'] = float(cost_match.group(1))
        except Exception as e:
            print(f"Error parsing solution file {filepath}: {e}")
            return None
        
        return solution
    
    def convert_to_neural_format(self, problem: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Convert benchmark problem to neural network format"""
        # Extract coordinates and demands
        coords = torch.zeros(problem['dimension'], 2)
        demands = torch.zeros(problem['dimension'])
        
        # Sort by node ID to ensure correct order
        problem['coordinates'].sort(key=lambda x: x[0])
        problem['demands'].sort(key=lambda x: x[0])
        
        for i, (node_id, x, y) in enumerate(problem['coordinates']):
            coords[i] = torch.tensor([x, y])
        
        for i, (node_id, demand) in enumerate(problem['demands']):
            demands[i] = demand
        
        # Normalize coordinates to [0, 1]
        min_coord = coords.min(dim=0)[0]
        max_coord = coords.max(dim=0)[0]
        coord_range = max_coord - min_coord
        coords_normalized = (coords - min_coord) / coord_range
        
        # Normalize demands by capacity
        demands_normalized = demands / problem['capacity']
        
        # Split depot and customer nodes
        depot_xy = coords_normalized[0:1]  # First node is depot
        node_xy = coords_normalized[1:]    # Rest are customers
        node_demand = demands_normalized[1:]  # Customer demands (depot demand is 0)
        
        return depot_xy, node_xy, node_demand, problem['capacity']
    
    def calculate_route_distance(self, route: List[int], coordinates: List[Tuple], use_normalized: bool = False) -> float:
        """Calculate total distance for a route"""
        if not route:
            return 0.0
        
        # Create coordinate lookup
        coord_dict = {node_id: (x, y) for node_id, x, y in coordinates}
        
        if use_normalized:
            # Normalize coordinates the same way as in convert_to_neural_format
            coords_array = np.array([(x, y) for _, x, y in coordinates])
            min_coord = coords_array.min(axis=0)
            max_coord = coords_array.max(axis=0)
            coord_range = max_coord - min_coord
            
            # Update coord_dict with normalized coordinates
            for node_id, x, y in coordinates:
                x_norm = (x - min_coord[0]) / coord_range[0]
                y_norm = (y - min_coord[1]) / coord_range[1]
                coord_dict[node_id] = (x_norm, y_norm)
        
        # Add depot at start and end
        full_route = [1] + route + [1]  # Depot is node 1
        
        total_distance = 0.0
        for i in range(len(full_route) - 1):
            from_node = full_route[i]
            to_node = full_route[i + 1]
            
            x1, y1 = coord_dict[from_node]
            x2, y2 = coord_dict[to_node]
            
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        return total_distance
    
    def load_benchmark_instance(self, vrp_file: str, sol_file: Optional[str] = None) -> Dict:
        """Load a complete benchmark instance with problem and solution"""
        problem = self.parse_vrp_file(vrp_file)
        solution = None
        
        if sol_file and Path(sol_file).exists():
            solution = self.parse_solution_file(sol_file)
            
            # Calculate total distance for solution validation
            # Use normalized coordinates to match model training data
            total_distance = 0.0
            for route in solution['routes']:
                route_dist = self.calculate_route_distance(route, problem['coordinates'], use_normalized=True)
                total_distance += route_dist
            solution['calculated_distance'] = total_distance
        
        depot_xy, node_xy, node_demand, capacity = self.convert_to_neural_format(problem)
        
        # Also store original coordinates for plotting
        original_coords = torch.zeros(problem['dimension'], 2)
        problem['coordinates'].sort(key=lambda x: x[0])
        for i, (node_id, x, y) in enumerate(problem['coordinates']):
            original_coords[i] = torch.tensor([x, y])
        
        return {
            'problem': problem,
            'solution': solution,
            'neural_format': {
                'depot_xy': depot_xy,
                'node_xy': node_xy,
                'node_demand': node_demand,
                'capacity': capacity
            },
            'original_coordinates': original_coords
        }


def load_all_benchmarks(benchmark_dir: str) -> Dict:
    """Load all benchmark instances from directory"""
    parser = CVRPBenchmarkParser()
    benchmark_dir = Path(benchmark_dir)
    
    benchmarks = {}
    
    # Find all .vrp files
    vrp_files = list(benchmark_dir.glob("*.vrp"))
    
    for vrp_file in vrp_files:
        # Find corresponding solution file
        base_name = vrp_file.stem
        sol_file = benchmark_dir / f"{base_name}.sol"
        
        try:
            instance = parser.load_benchmark_instance(str(vrp_file), str(sol_file) if sol_file.exists() else None)
            benchmarks[base_name] = instance
            print(f"Loaded benchmark: {base_name}")
            print(f"  Dimension: {instance['problem']['dimension']}")
            print(f"  Capacity: {instance['problem']['capacity']}")
            if instance['solution']:
                print(f"  Routes: {len(instance['solution']['routes'])}")
                print(f"  Distance: {instance['solution']['calculated_distance']:.2f}")
            print()
        except Exception as e:
            print(f"Error loading {vrp_file}: {e}")
    
    return benchmarks


if __name__ == "__main__":
    # Test the parser
    benchmark_dir = "../benchmark_data"
    benchmarks = load_all_benchmarks(benchmark_dir)
    
    print(f"Successfully loaded {len(benchmarks)} benchmark instances")
    
    # Show example conversion
    if benchmarks:
        example_name = list(benchmarks.keys())[0]
        example = benchmarks[example_name]
        
        print(f"\nExample: {example_name}")
        print(f"Original coordinates shape: {len(example['problem']['coordinates'])}")
        print("Neural format shapes:")
        print(f"  depot_xy: {example['neural_format']['depot_xy'].shape}")
        print(f"  node_xy: {example['neural_format']['node_xy'].shape}")
        print(f"  node_demand: {example['neural_format']['node_demand'].shape}")