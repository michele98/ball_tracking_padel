import heapq
import numpy as np
from numba import jit


@jit
def euclidean_distance(pos1, pos2, axis=0):
    delta = pos1 - pos2
    if axis==1:
        delta = delta.T
    return np.sqrt(delta[0]**2 + delta[1]**2)


@jit
def squared_distance(pos1, pos2, axis=0):
    delta = pos1 - pos2
    if axis==1:
        delta = delta.T
    return delta[0]**2 + delta[1]**2


@jit
def find_middle_support_index(support):
    support_k = support[:,0]
    return np.argmin(np.abs(np.abs(support_k[-1]-support_k) - np.abs(support_k[0]-support_k)))


class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v, weight):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append((v, weight))
        if v not in self.graph:
            self.graph[v] = []

    def get_adjacent_vertices(self, u):
        return self.graph[u] if u in self.graph else []

    def get_all_vertices(self):
        return list(self.graph.keys())

    def dijkstra(self, start):
        distances = {vertex: (float('inf'), float('inf')) for vertex in self.graph}
        distances[start] = (0., 0.)
        predecessors = {vertex: None for vertex in self.graph}

        priority_queue = [(0, 0, start)]
        while priority_queue:
            current_distance, current_nodes, current_vertex = heapq.heappop(priority_queue)

            if (current_distance, current_nodes) > distances[current_vertex]:
                continue

            for neighbor, weight in self.graph[current_vertex]:
                distance = current_distance + weight
                nodes = current_nodes + 1

                if (distance, nodes) < distances[neighbor]:
                    distances[neighbor] = (distance, nodes)
                    predecessors[neighbor] = current_vertex
                    heapq.heappush(priority_queue, (distance, nodes, neighbor))

        return distances, predecessors

    def get_shortest_path(self, end, predecessors):
        path = []
        current_vertex = end
        while current_vertex is not None:
            path.insert(0, current_vertex)
            current_vertex = predecessors[current_vertex]
        return path
