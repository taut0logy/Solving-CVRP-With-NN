import torch
import numpy as np
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class CVRPInstance:
    depot_xy: torch.Tensor      # shape: (1, 2)
    node_xy: torch.Tensor       # shape: (n_customers, 2)
    demands: torch.Tensor       # shape: (n_customers,)
    capacity: float
    optimal_routes: List[List[int]]  # List of routes (node indices)
    optimal_distance: float


class TraditionalCVRPSolver:
    """Traditional CVRP solver using Nearest Neighbor + 2-opt improvement"""
    
    def __init__(self):
        pass
    
    def solve(self, depot_xy: torch.Tensor, node_xy: torch.Tensor, 
              demands: torch.Tensor, capacity: float) -> Tuple[List[List[int]], float]:
        """
        Solve CVRP using traditional heuristics
        Returns: (routes, total_distance)
        """
        # Combine depot and customer coordinates
        all_coords = torch.cat([depot_xy, node_xy], dim=0)  # shape: (n+1, 2)
        n_customers = node_xy.shape[0]
        
        # Create unvisited customers set
        unvisited = set(range(1, n_customers + 1))  # Customer indices (1 to n)
        routes = []
        total_distance = 0.0
        
        while unvisited:
            route, route_distance = self._construct_route(
                all_coords, demands, capacity, unvisited
            )
            routes.append(route)
            total_distance += route_distance
            
            # Remove visited customers
            for customer in route:
                unvisited.discard(customer)
        
        # Apply 2-opt improvement to each route
        improved_routes = []
        improved_total_distance = 0.0
        
        for route in routes:
            improved_route, improved_distance = self._improve_route_2opt(
                route, all_coords
            )
            improved_routes.append(improved_route)
            improved_total_distance += improved_distance
        
        return improved_routes, improved_total_distance
    
    def _construct_route(self, all_coords: torch.Tensor, demands: torch.Tensor, 
                        capacity: float, unvisited: set) -> Tuple[List[int], float]:
        """Construct a single route using nearest neighbor heuristic"""
        if not unvisited:
            return [], 0.0
        
        route = []
        current_load = 0.0
        current_pos = 0  # Start at depot (index 0)
        route_distance = 0.0
        
        while unvisited:
            # Find nearest feasible customer
            best_customer = None
            best_distance = float('inf')
            
            for customer in unvisited:
                # Check capacity constraint
                if current_load + demands[customer - 1] <= capacity:
                    # Calculate distance
                    distance = self._euclidean_distance(
                        all_coords[current_pos], all_coords[customer]
                    )
                    if distance < best_distance:
                        best_distance = distance
                        best_customer = customer
            
            if best_customer is None:
                # No feasible customer found, return to depot
                break
            
            # Add customer to route
            route.append(best_customer)
            current_load += demands[best_customer - 1]
            route_distance += best_distance
            current_pos = best_customer
            
            # Remove customer from unvisited set
            unvisited.remove(best_customer)
        
        # Return to depot
        if route:
            return_distance = self._euclidean_distance(
                all_coords[current_pos], all_coords[0]
            )
            route_distance += return_distance
        
        return route, route_distance
    
    def _improve_route_2opt(self, route: List[int], all_coords: torch.Tensor) -> Tuple[List[int], float]:
        """Improve route using 2-opt local search"""
        if len(route) < 3:
            return route, self._calculate_route_distance(route, all_coords)
        
        improved = True
        best_route = route.copy()
        best_distance = self._calculate_route_distance(best_route, all_coords)
        
        max_iterations = 100
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # Try 2-opt swap
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    new_distance = self._calculate_route_distance(new_route, all_coords)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        route = new_route
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_route, best_distance
    
    def _calculate_route_distance(self, route: List[int], all_coords: torch.Tensor) -> float:
        """Calculate total distance for a route (including depot visits)"""
        if not route:
            return 0.0
        
        distance = 0.0
        # Depot to first customer
        distance += self._euclidean_distance(all_coords[0], all_coords[route[0]])
        
        # Between customers
        for i in range(len(route) - 1):
            distance += self._euclidean_distance(all_coords[route[i]], all_coords[route[i+1]])
        
        # Last customer to depot
        distance += self._euclidean_distance(all_coords[route[-1]], all_coords[0])
        
        return distance
    
    def _euclidean_distance(self, point1: torch.Tensor, point2: torch.Tensor) -> float:
        """Calculate Euclidean distance between two points"""
        return torch.sqrt(torch.sum((point1 - point2) ** 2)).item()


class CVRPDataGenerator:
    """Generate random CVRP instances with ground truth solutions"""
    
    def __init__(self, solver: TraditionalCVRPSolver = None):
        self.solver = solver if solver else TraditionalCVRPSolver()
    
    def generate_random_instance(self, n_customers: int, capacity_range: Tuple[int, int] = (100, 200),
                                demand_range: Tuple[int, int] = (5, 25)) -> CVRPInstance:
        """Generate a single random CVRP instance"""
        
        # Generate random coordinates in [0, 1]
        depot_xy = torch.rand(1, 2)
        node_xy = torch.rand(n_customers, 2)
        
        # Generate random demands
        min_demand, max_demand = demand_range
        demands = torch.randint(min_demand, max_demand + 1, (n_customers,)).float()
        
        # Set capacity as a multiple of average demand
        avg_demand = demands.mean().item()
        capacity_multiplier = random.uniform(*capacity_range) / avg_demand
        capacity = capacity_multiplier * avg_demand
        
        # Ensure capacity can handle at least the largest demand
        capacity = max(capacity, demands.max().item() * 1.1)
        
        # Solve using traditional method
        optimal_routes, optimal_distance = self.solver.solve(
            depot_xy, node_xy, demands, capacity
        )
        
        return CVRPInstance(
            depot_xy=depot_xy,
            node_xy=node_xy,
            demands=demands,
            capacity=capacity,
            optimal_routes=optimal_routes,
            optimal_distance=optimal_distance
        )
    
    def generate_dataset(self, n_instances: int, n_customers: int, 
                        capacity_range: Tuple[int, int] = (100, 200),
                        demand_range: Tuple[int, int] = (5, 25)) -> List[CVRPInstance]:
        """Generate a dataset of CVRP instances"""
        dataset = []
        
        print(f"Generating {n_instances} CVRP instances with {n_customers} customers...")
        
        for i in range(n_instances):
            if (i + 1) % max(1, n_instances // 10) == 0:
                print(f"Progress: {i + 1}/{n_instances}")
            
            instance = self.generate_random_instance(
                n_customers, capacity_range, demand_range
            )
            dataset.append(instance)
        
        print("Dataset generation complete!")
        self._print_dataset_stats(dataset)
        
        return dataset
    
    def _print_dataset_stats(self, dataset: List[CVRPInstance]):
        """Print statistics about the generated dataset"""
        if not dataset:
            return
        
        distances = [instance.optimal_distance for instance in dataset]
        route_counts = [len(instance.optimal_routes) for instance in dataset]
        
        print("\nDataset Statistics:")
        print(f"  Number of instances: {len(dataset)}")
        print(f"  Customers per instance: {dataset[0].node_xy.shape[0]}")
        print(f"  Average distance: {np.mean(distances):.2f} ± {np.std(distances):.2f}")
        print(f"  Distance range: [{min(distances):.2f}, {max(distances):.2f}]")
        print(f"  Average routes per instance: {np.mean(route_counts):.2f}")
        print(f"  Route count range: [{min(route_counts)}, {max(route_counts)}]")
    
    def visualize_instance(self, instance: CVRPInstance, save_path: str = None):
        """Visualize a CVRP instance and its solution"""
        plt.figure(figsize=(10, 8))
        
        # Plot depot
        depot = instance.depot_xy[0]
        plt.scatter(depot[0], depot[1], c='red', s=200, marker='s', label='Depot', zorder=5)
        
        # Plot customers
        customers = instance.node_xy
        plt.scatter(customers[:, 0], customers[:, 1], c='blue', s=100, label='Customers', zorder=4)
        
        # Add demand labels
        for i, (x, y) in enumerate(customers):
            plt.annotate(f'{int(instance.demands[i])}', (x, y), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot routes
        colors = plt.cm.tab10(np.linspace(0, 1, len(instance.optimal_routes)))
        
        for route_idx, route in enumerate(instance.optimal_routes):
            if not route:
                continue
            
            # Create route coordinates (depot -> customers -> depot)
            route_coords = [depot]
            for customer_idx in route:
                route_coords.append(customers[customer_idx - 1])  # -1 because customer indices start from 1
            route_coords.append(depot)
            
            # Plot route
            route_coords = torch.stack(route_coords)
            plt.plot(route_coords[:, 0], route_coords[:, 1], 
                    color=colors[route_idx], linewidth=2, alpha=0.7,
                    label=f'Route {route_idx + 1}')
        
        plt.title(f'CVRP Instance (Distance: {instance.optimal_distance:.2f}, Capacity: {instance.capacity:.1f})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_dataset(self, dataset: List[CVRPInstance], filepath: str):
        """Save dataset to file"""
        dataset_dict = {
            'instances': [],
            'metadata': {
                'n_instances': len(dataset),
                'n_customers': dataset[0].node_xy.shape[0] if dataset else 0,
                'avg_distance': np.mean([inst.optimal_distance for inst in dataset]) if dataset else 0
            }
        }
        
        for instance in dataset:
            instance_dict = {
                'depot_xy': instance.depot_xy,
                'node_xy': instance.node_xy,
                'demands': instance.demands,
                'capacity': instance.capacity,
                'optimal_routes': instance.optimal_routes,
                'optimal_distance': instance.optimal_distance
            }
            dataset_dict['instances'].append(instance_dict)
        
        torch.save(dataset_dict, filepath)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[CVRPInstance]:
        """Load dataset from file"""
        dataset_dict = torch.load(filepath)
        dataset = []
        
        for instance_dict in dataset_dict['instances']:
            instance = CVRPInstance(
                depot_xy=instance_dict['depot_xy'],
                node_xy=instance_dict['node_xy'],
                demands=instance_dict['demands'],
                capacity=instance_dict['capacity'],
                optimal_routes=instance_dict['optimal_routes'],
                optimal_distance=instance_dict['optimal_distance']
            )
            dataset.append(instance)
        
        print(f"Loaded {len(dataset)} instances from {filepath}")
        return dataset


if __name__ == "__main__":
    # Example usage
    generator = CVRPDataGenerator()
    
    # Generate a small dataset
    dataset = generator.generate_dataset(
        n_instances=10,
        n_customers=20,
        capacity_range=(80, 120),
        demand_range=(5, 15)
    )
    
    # Visualize first instance
    if dataset:
        print("\nExample instance:")
        print(f"  Depot: {dataset[0].depot_xy}")
        print(f"  Customers: {dataset[0].node_xy.shape}")
        print(f"  Demands: {dataset[0].demands}")
        print(f"  Capacity: {dataset[0].capacity}")
        print(f"  Routes: {dataset[0].optimal_routes}")
        print(f"  Distance: {dataset[0].optimal_distance:.2f}")
        
        # Uncomment to visualize
        # generator.visualize_instance(dataset[0])
        
        # Save dataset
        generator.save_dataset(dataset, "cvrp_training_data.pt")