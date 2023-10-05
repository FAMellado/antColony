import random
import math
import numpy as np     
import pandas as pd #para dataframes
import sys 
import time
import matplotlib.pyplot as plt


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def create_adjacency_matrix(coordinates):
    '''Creates and returns an adjacency matrix filled with distances between cities.'''
    n = len(coordinates)
    
    # Initializing the adjacency matrix with zeros
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):  # Start from i+1 to avoid duplicate calculations
            distance = euclidean_distance(coordinates[i], coordinates[j])
            matrix[i][j] = distance
            matrix[j][i] = distance  # Since it's a non-directed graph
            
    return matrix


class Graph:
    def __init__(self, matrix):
        self.matrix = matrix
        self.num_nodes = len(matrix)
        self.pheromone = [[1 / (self.num_nodes * self.num_nodes)
                           for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]
        self.initial_pheromone = 1 / (self.num_nodes * self.num_nodes)

class Ant:
    def __init__(self, graph, alpha, beta):
        self.graph = graph
        self.alpha = alpha
        alpha = 0.1         
        self.beta = beta
        beta = 2.5
        self.tabu = []  
        self.current_node = random.randint(0, graph.num_nodes - 1)
        self.tabu.append(self.current_node)
        self.path_len = 0.0

    def _select_next(self):
        unvisited = set(range(self.graph.num_nodes)) - set(self.tabu)
        probabilities = [self.probability(n) for n in unvisited]
        return random.choices(list(unvisited), probabilities)[0]

    def probability(self, next_node):
        tau = self.graph.pheromone[self.current_node][next_node]  # Pheromone
        eta = 1 / self.graph.matrix[self.current_node][next_node]  # Inverse of distance
        total = sum([tau * eta**self.beta for tau, eta in 
                     [(self.graph.pheromone[self.current_node][n], 1 / self.graph.matrix[self.current_node][n]) for n in 
                      set(range(self.graph.num_nodes)) - set(self.tabu)]])
        return (tau**self.alpha * eta**self.beta) / total

    def move(self, next_node):
        self.path_len += self.graph.matrix[self.current_node][next_node]
        self.local_update_pheromone(self.current_node, next_node)
        self.tabu.append(next_node)
        self.current_node = next_node

    def local_update_pheromone(self, i, j):
        rho = 0.1  # Local pheromone evaporation coefficient
        self.graph.pheromone[i][j] = (1 - rho) * self.graph.pheromone[i][j] + rho * self.graph.initial_pheromone
        self.graph.pheromone[j][i] = self.graph.pheromone[i][j]  # Assuming an undirected graph

    def update_pheromone_delta(self):
        pheromone_delta = 1 / self.path_len
        delta = [[0 for j in range(self.graph.num_nodes)] for i in range(self.graph.num_nodes)]
        for i in range(len(self.tabu) - 1):
            delta[self.tabu[i]][self.tabu[i+1]] = pheromone_delta
            delta[self.tabu[i+1]][self.tabu[i]] = pheromone_delta
        return delta

def ant_colony(graph, num_ants, num_iterations, decay, alpha, beta):
    best_path = None
    best_distance = float('inf')
    for _ in range(num_iterations):
        ants = [Ant(graph, alpha, beta) for __ in range(num_ants)]
        for ant in ants:
            for __ in range(graph.num_nodes - 1):
                ant.move(ant._select_next())
            ant.path_len += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
            if ant.path_len < best_distance:
                best_path = ant.tabu
                best_distance = ant.path_len
        # Global pheromone update
        global_update_pheromone(graph, best_path, best_distance, decay)
    return best_path, best_distance

def global_update_pheromone(graph, best_path, best_distance, decay):
    pheromone_added = 1.0 / best_distance
    for i in range(len(best_path) - 1):
        graph.pheromone[best_path[i]][best_path[i + 1]] = (1 - decay) * graph.pheromone[best_path[i]][best_path[i + 1]] + decay * pheromone_added
        graph.pheromone[best_path[i + 1]][best_path[i]] = graph.pheromone[best_path[i]][best_path[i + 1]]


def read_coordinates(filename):
    '''Reads the file and returns a list of (x, y) coordinates.'''
    coordinates = []
    
    # Flag to determine if we're reading the NODE_COORD_SECTION
    reading_coords = False
    
    with open(filename, 'r') as file:
        for line in file:
            # Strip newline and whitespace
            line = line.strip()
            
            # Start reading if NODE_COORD_SECTION is found
            if line == 'NODE_COORD_SECTION':
                reading_coords = True
                continue
                
            # Stop reading if EOF is found
            if line == 'EOF':
                break
            
            if reading_coords:
                _, x, y = line.split()
                coordinates.append((float(x), float(y)))
                
    return coordinates

# Using the function
filename = "berlin.txt"
coords = read_coordinates(filename)
matrix = create_adjacency_matrix(coords)
for row in matrix:
    print(row)
    print("\n")





    



    

