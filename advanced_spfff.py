import math
import heapq
import sys
import random
from typing import Tuple, Optional, Dict, List

sys.setrecursionlimit(4000) 

class Node:
    def __init__(self, node_id: str, x: float, y: float):
        self.id = node_id
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Node({self.id}, x={self.x:.2f}, y={self.y:.2f})"


class Edge:
    def __init__(self, source: str, target: str, base_weight: float, variance: float, angle: float):
        self.source = source
        self.target = target
        self.base_weight = base_weight
        self.variance = variance
        self.angle = angle 

    def weight(self):
        noise = random.uniform(-self.variance, self.variance)
        return max(0.1, self.base_weight + noise)

    def __repr__(self):
        return f"Edge({self.source}->{self.target}, base={self.base_weight:.2f}, var={self.variance:.2f})"


class Graph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.adj: Dict[str, List[Tuple[str, Edge]]] = {} 
        self.edges_dict: Dict[Tuple[str, str], Edge] = {}

    def add_node(self, node: Node):
        self.nodes[node.id] = node
        if node.id not in self.adj:
            self.adj[node.id] = []

    def add_edge(self, edge: Edge):
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("Edge endpoints must be valid nodes.")
        
        if edge.source not in self.adj: self.adj[edge.source] = []
        if edge.target not in self.adj: self.adj[edge.target] = []

        self.adj[edge.source].append((edge.target, edge))
        self.edges_dict[(edge.source, edge.target)] = edge

    def get_neighbors(self, node_id) -> List[Tuple[str, Edge]]:
        return self.adj.get(node_id, [])

    def get_edge_data(self, u, v) -> Optional[Edge]:
        return self.edges_dict.get((u, v))
        
    def get_coords(self, node_id) -> Optional[Tuple[float, float]]:
        node = self.nodes.get(node_id)
        if node:
            return (node.x, node.y)
        return None

    def get_num_nodes(self):
        return len(self.nodes)

GRAPH_DATA_RAW = {
    'Nodes': {
        'A': (0, 0), 'B': (10, 0), 'C': (20, 0), 'D': (30, 0), 'E': (40, 0),
        'F': (0, 10), 'G': (10, 10), 'H': (20, 10), 'I': (30, 10), 'J': (40, 10),
        'K': (0, 20), 'L': (10, 20), 'M': (20, 20), 'N': (30, 20), 'O': (40, 20),
        'P': (0, 30), 'Q': (10, 30), 'R': (20, 30), 'S': (30, 30), 'T': (40, 30),
        'U': (0, 40), 'V': (10, 40), 'W': (20, 40), 'X': (30, 40), 'Y': (40, 40),
        'Z': (50, 40)
    },
    'Edges': [
        ('A', 'B', 10.0, 1.0, 0), ('A', 'F', 10.0, 1.2, 90), ('B', 'C', 10.0, 0.9, 0), 
        ('C', 'D', 10.0, 1.1, 0), ('D', 'E', 10.0, 1.0, 0), ('B', 'G', 10.0, 1.5, 90),
        ('F', 'G', 10.0, 1.0, 0), ('G', 'H', 10.0, 0.8, 0), ('H', 'I', 10.0, 1.1, 0),
        ('I', 'J', 10.0, 1.0, 0), ('F', 'K', 10.0, 1.2, 90), ('G', 'L', 10.0, 1.5, 90),
        ('H', 'M', 10.0, 1.3, 90), ('I', 'N', 10.0, 1.0, 90), ('J', 'O', 10.0, 1.4, 90),
        ('K', 'L', 10.0, 0.9, 0), ('L', 'M', 10.0, 1.1, 0), ('M', 'N', 10.0, 1.0, 0),
        ('N', 'O', 10.0, 1.2, 0), ('K', 'P', 10.0, 1.3, 90), ('L', 'Q', 10.0, 1.5, 90),
        ('M', 'R', 10.0, 1.1, 90), ('N', 'S', 10.0, 0.9, 90), ('O', 'T', 10.0, 1.0, 90),
        ('P', 'Q', 10.0, 1.0, 0), ('Q', 'R', 10.0, 1.1, 0), ('R', 'S', 10.0, 1.2, 0),
        ('S', 'T', 10.0, 0.8, 0), ('P', 'U', 10.0, 1.4, 90), ('Q', 'V', 10.0, 1.2, 90),
        ('R', 'W', 10.0, 1.0, 90), ('S', 'X', 10.0, 1.3, 90), ('T', 'Y', 10.0, 1.1, 90),
        ('U', 'V', 10.0, 1.1, 0), ('V', 'W', 10.0, 0.9, 0), ('W', 'X', 10.0, 1.0, 0),
        ('X', 'Y', 10.0, 1.2, 0),
        ('B', 'F', 14.1, 2.0, 135), ('E', 'I', 14.1, 2.5, 135), ('M', 'S', 14.1, 1.8, 45), 
        ('Y', 'Z', 10.0, 1.0, 90), ('X', 'Z', 10.0, 1.5, 90), ('T', 'Z', 14.1, 2.0, 45), 
        ('D', 'H', 14.1, 2.2, 135), 
        ('A', 'G', 14.1, 2.0, 45), ('B', 'H', 14.1, 2.0, 45), ('C', 'I', 14.1, 2.0, 45), ('D', 'J', 14.1, 2.0, 45),
        ('F', 'L', 14.1, 2.0, 45), ('G', 'M', 14.1, 2.0, 45), ('H', 'N', 14.1, 2.0, 45), ('I', 'O', 14.1, 2.0, 45),
        ('K', 'Q', 14.1, 2.0, 45), ('L', 'R', 14.1, 2.0, 45), ('M', 'S', 14.1, 2.0, 45), ('N', 'T', 14.1, 2.0, 45),
        ('P', 'V', 14.1, 2.0, 45), ('Q', 'W', 14.1, 2.0, 45), ('R', 'X', 14.1, 2.0, 45), ('S', 'Y', 14.1, 2.0, 45),
        ('C', 'G', 14.1, 2.0, 135), 
        ('G', 'K', 14.1, 2.0, 135), ('H', 'L', 14.1, 2.0, 135), ('I', 'M', 14.1, 2.0, 135), ('J', 'N', 14.1, 2.0, 135),
        ('L', 'P', 14.1, 2.0, 135), ('M', 'Q', 14.1, 2.0, 135), ('N', 'R', 14.1, 2.0, 135), ('O', 'S', 14.1, 2.0, 135),
        ('Q', 'U', 14.1, 2.0, 135), ('R', 'V', 14.1, 2.0, 135), ('S', 'W', 14.1, 2.0, 135), ('T', 'X', 14.1, 2.0, 135),
        ('A', 'L', 14.1, 2.0, 135), 
    ]
}

def load_graph_from_data(raw_data: dict) -> Graph:
    g = Graph()
    
    for node_id, coords in raw_data['Nodes'].items():
        g.add_node(Node(node_id, coords[0], coords[1]))
    
    for u, v, weight, variance, angle in raw_data['Edges']:
        g.add_edge(Edge(u, v, weight, variance, angle))
        g.add_edge(Edge(v, u, weight, variance, (angle + 180) % 360))
        
    return g

class CostModel:
    def __init__(self, graph: Graph):
        self.time_factor_cache = {}  
        self.graph = graph 
        
    def calculate_turning_angle(self, prev_node: str, curr_node: str, next_node: str) -> float:
        p1 = self.graph.get_coords(prev_node)
        p2 = self.graph.get_coords(curr_node)
        p3 = self.graph.get_coords(next_node)
        
        if not all([p1, p2, p3]):
            return 0.0
        
        vector1 = (p1[0] - p2[0], p1[1] - p2[1]) 
        vector2 = (p3[0] - p2[0], p3[1] - p2[1]) 
        
        len1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        len2 = math.sqrt(vector2[0]**2 + vector2[1]**2)
        
        if len1 == 0 or len2 == 0:
            return 0.0
        
        dot_product = vector1[0]*vector2[0] + vector1[1]*vector2[1]
        cos_angle = dot_product / (len1 * len2)
        
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        return math.acos(cos_angle)
        
    def get_time_factor(self, hour: int) -> float:
        if hour in self.time_factor_cache:
            return self.time_factor_cache[hour]
        
        if 7 <= hour <= 9:      
            factor = 1.3
        elif 17 <= hour <= 19:  
            factor = 1.4
        elif 12 <= hour <= 13:  
            factor = 1.1
        elif 0 <= hour <= 5:    
            factor = 0.8
        else:                   
            factor = 1.0
        
        self.time_factor_cache[hour] = factor
        return factor
        
    def calculate_geometric_penalty(self, angle: float, lambda_coef: float) -> float:
        angle_deg = math.degrees(angle)
        
        if angle_deg < 30:
            penalty = 0.1 * (angle_deg / 30)
        elif angle_deg < 90:
            penalty = 0.1 + 0.3 * ((angle_deg - 30) / 60)
        else:
            penalty = 0.4 + 0.6 * (min(angle_deg, 180) / 180)
            
        return lambda_coef * penalty
        
    def get_edge_cost_api(self, 
                          base_weight: float,
                          hour: int,
                          lambda_coef: float,
                          prev_node: Optional[str] = None,
                          curr_node: Optional[str] = None,
                          next_node: Optional[str] = None) -> float:
        
        time_factor = self.get_time_factor(hour)
        geometric_penalty = 0.0
        
        if prev_node and curr_node and next_node:
            angle = self.calculate_turning_angle(prev_node, curr_node, next_node)
            geometric_penalty = self.calculate_geometric_penalty(angle, lambda_coef)
            
        final_weight = base_weight * time_factor * (1 + geometric_penalty)
        
        return final_weight

class SolverBase:
    def __init__(self, graph: Graph):
        self.graph = graph

    def solve(self, start_id, end_id, cost_func=None, **kwargs):
        pq = [(0, start_id, [start_id])] 
        distances = {node_id: float('inf') for node_id in self.graph.nodes}
        distances[start_id] = 0
        visited_nodes = 0
        
        while pq:
            current_cost, current_node, current_path = heapq.heappop(pq)
            visited_nodes += 1

            if current_node == end_id:
                return current_path, current_cost, visited_nodes

            if current_cost > distances[current_node]:
                continue

            for neighbor_id, edge in self.graph.get_neighbors(current_node):
                edge_cost = edge.base_weight
                new_cost = current_cost + edge_cost

                if new_cost < distances[neighbor_id]:
                    distances[neighbor_id] = new_cost
                    new_path = current_path + [neighbor_id]
                    heapq.heappush(pq, (new_cost, neighbor_id, new_path))
                    
        return None, float('inf'), visited_nodes


class AStarSolver(SolverBase):
    
    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.visited_nodes = 0
        
    def calculate_heuristic(self, node1: str, node2: str):
        coord1 = self.graph.get_coords(node1)
        coord2 = self.graph.get_coords(node2)
        
        if coord1 and coord2:
            return math.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
        return 0

    def solve(self, start_id: str, end_id: str, cost_model: CostModel, **kwargs):
        
        self.visited_nodes = 0
        
        pq = [(0, 0, start_id, [start_id])] 
        distances = {node_id: float('inf') for node_id in self.graph.nodes}
        distances[start_id] = 0
        
        while pq:
            f_cost, current_g_cost, current_node, current_path = heapq.heappop(pq)
            self.visited_nodes += 1

            if current_node == end_id:
                return current_path, current_g_cost, self.visited_nodes

            if current_g_cost > distances[current_node]:
                continue

            for neighbor_id, edge in self.graph.get_neighbors(current_node):
                u_prev = current_path[-2] if len(current_path) > 1 else None
                u = current_node
                v = neighbor_id
                
                edge_cost = cost_model.get_edge_cost_api(
                    base_weight=edge.base_weight,
                    hour=kwargs.get('t_start', 12),
                    lambda_coef=kwargs.get('lambda_turn', 0.0),
                    prev_node=u_prev,
                    curr_node=u,
                    next_node=v
                )
                
                new_g_cost = current_g_cost + edge_cost

                if new_g_cost < distances[v]:
                    distances[v] = new_g_cost
                    new_path = current_path + [v]
                    
                    h_cost = self.calculate_heuristic(v, end_id)
                    new_f_cost = new_g_cost + h_cost
                    
                    heapq.heappush(pq, (new_f_cost, new_g_cost, v, new_path))
                    
        return None, float('inf'), self.visited_nodes


class KPathSolver:
    def __init__(self, graph: Graph, cost_model: CostModel):
        self.graph = graph
        self.cost_model = cost_model
        self.astar_solver = AStarSolver(graph) 
        self.all_paths: List[Tuple[float, List[str]]] = [] 
        self.cost_limit = float('inf')
        self.cost_kwargs = {}

    def _recursive_find_paths(self, current_node: str, end_id: str, current_cost: float, current_path: List[str]):
        
        if current_cost > self.cost_limit:
            return

        if current_node == end_id:
            path_tuple = tuple(current_path)
            if (current_cost, path_tuple) not in self.all_paths:
                self.all_paths.append((current_cost, current_path[:]))
            return

        for neighbor_id, edge in self.graph.get_neighbors(current_node):
            v = neighbor_id
            
            if v in current_path:
                continue

            u = current_node
            u_prev = current_path[-2] if len(current_path) > 1 else None
            
            edge_cost = self.cost_model.get_edge_cost_api(
                base_weight=edge.base_weight,
                hour=self.cost_kwargs.get('t_start', 12),
                lambda_coef=self.cost_kwargs.get('lambda_turn', 0.0),
                prev_node=u_prev,
                curr_node=u,
                next_node=v
            )
            
            new_cost = current_cost + edge_cost
            
            new_path = current_path + [v]
            self._recursive_find_paths(v, end_id, new_cost, new_path)

    def solve(self, start_id: str, end_id: str, k: int, cost_kwargs: Dict):
        
        self.all_paths = []
        self.cost_kwargs = cost_kwargs
        
        p1_path, p1_cost, p1_visited = self.astar_solver.solve(start_id, end_id, self.cost_model, **cost_kwargs)

        if not p1_path or p1_cost == float('inf'):
            return [] 

        COST_LIMIT_FACTOR = 1.5 
        self.cost_limit = p1_cost * COST_LIMIT_FACTOR

        self._recursive_find_paths(start_id, end_id, 0.0, [start_id])

        self.all_paths.sort(key=lambda x: x[0])

        final_results = []
        seen_paths = set()
        
        for cost, path in self.all_paths:
            path_tuple = tuple(path)
            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                final_results.append({
                    'path': path,
                    'cost': cost,
                    'rank': len(final_results) + 1,
                    'visited': p1_visited 
                })
                if len(final_results) >= k:
                    break

        return final_results

def calculate_path_reliability(path: List[str], graph: Graph):
    total_variance = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge = graph.get_edge_data(u, v)
        if edge:
            total_variance += edge.variance
    return total_variance

def calculate_all_pairs_shortest_path(graph: Graph):
    apsp_matrix = {}
    node_ids = list(graph.nodes.keys())
    
    cost_model_base = CostModel(graph) 
    base_kwargs = {'t_start': 12, 'lambda_turn': 0.0}
    
    solver = AStarSolver(graph) 

    for start_node in node_ids:
        apsp_matrix[start_node] = {}
        for end_node in node_ids:
            path, cost, _ = solver.solve(start_node, end_node, cost_model_base, **base_kwargs)
            apsp_matrix[start_node][end_node] = cost

    return apsp_matrix

def format_path(path_list):
    return " -> ".join(path_list)

def main():
    graph = load_graph_from_data(GRAPH_DATA_RAW)
    cost_model = CostModel(graph)

    print("\nAdvanced SPF Solver")
    
    print("[Background Task] Computing All-Pairs Shortest Path (APSP)...")
    apsp_matrix = calculate_all_pairs_shortest_path(graph)
    
    total_raw_edges = len(GRAPH_DATA_RAW['Edges'])
    total_edges = total_raw_edges * 2

    print(f"[APSP Cache Complete] Nodes: {graph.get_num_nodes()} | Total Edges: {total_edges} | Calculated {graph.get_num_nodes()**2} pairs distance.")

    try:
        start_id = input("Enter Start Node (S) ID: ").strip().upper()
        end_id = input("Enter Target Node (T) ID: ").strip().upper()
        k = int(input("Enter Number of Alternative Paths (K): "))
        t_start = int(input("Enter Start Time (t_start, 0-23): "))
        lambda_turn = float(input("Enter Geometric Penalty Factor (Lambda, 0.0 - 1.0): "))
    except ValueError:
        print("[Error] Input format error. Check if K/t_start/Lambda are numeric.")
        return

    if start_id not in graph.nodes or end_id not in graph.nodes:
        print("[Error] Start or Target Node ID does not exist.")
        return

    cost_kwargs = {'t_start': t_start, 'lambda_turn': lambda_turn}
    
    print("\n[System] Executing Optimized Query...")
    
    k_solver = KPathSolver(graph, cost_model)
    results = k_solver.solve(start_id, end_id, k, cost_kwargs)

    base_solver = SolverBase(graph)
    _, _, base_visited = base_solver.solve(start_id, end_id)
    
    a_star_visited = results[0]['visited'] if results else 0
    
    print("\nPerformance Analysis")
    print(f"[Performance Comparison] P1 (A*) Visited Nodes: {a_star_visited}")
    print(f"[Performance Comparison] Base Dijkstra Visited Nodes: {base_visited}")
    
    if not results:
        print("[Result] Path unreachable or K=0.")
        return

    print("\nP2P K-Shortest Path Results")
    
    final_output = []
    for r in results:
        path = r['path']
        cost = r['cost']
        reliability = calculate_path_reliability(path, graph)
        
        final_output.append({
            'rank': r['rank'],
            'path_str': format_path(path),
            'cost': cost,
            'reliability': reliability
        })
        
    print(r"| Rank | Final Total Cost (min) | Reliability Variance ($\sigma^2$) | Path Sequence |")
    print("|------|------------------------|-----------------------------------|---------------|")
    
    try:
        min_reliability = min(float(item['reliability']) for item in final_output)
    except ValueError:
        min_reliability = float('inf')
    
    for i, item in enumerate(final_output):
        reliability_label = f"{item['reliability']:.2f}"
        if float(item['reliability']) == min_reliability:
            reliability_label += " (Most Reliable)"
        
        print(f"| #{item['rank']:<4} | {item['cost']:<22.2f} | {reliability_label:<31} | {item['path_str']} |")
    
    print("\n[System] All tasks complete.")

if __name__ == "__main__":
    main()