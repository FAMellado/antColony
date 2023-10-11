import random
import math
import numpy as np     
import sys 
import time
import matplotlib.pyplot as plt

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
        self.alpha = alpha                                              #Factor de evaporación
        self.beta = beta                                                #Coeficiente heurístico
        self.tabu = []                                                  #Visitados
        self.current_node = random.randint(0, graph.num_nodes - 1)      #Pos inicial hormigas
        self.tabu.append(self.current_node)                             #Marcamos visitados
        self.path_len = 0.0                                             #El path recorrido
        self.traveled_edges = []

    def _select_next(self):
        unvisited = set(range(self.graph.num_nodes)) - set(self.tabu)
        probabilities = [self.probability(n) for n in unvisited]
        return random.choices(list(unvisited), probabilities)[0] 

    def probability(self, next_node):
        tau = self.graph.pheromone[self.current_node][next_node]  # Feromona entre las distancias Estas 2 variables representan el deseo de optar por esta ruta
        eta = 1 / self.graph.matrix[self.current_node][next_node]  # Inverso de la distancia, proximidad.

        total = sum([tau**self.alpha * eta**self.beta for tau, eta in # La suma de las prob de una hormiga de ir del nodo actual a todos los otros que quedan por visitar
                    [(self.graph.pheromone[self.current_node][n], 1 / self.graph.matrix[self.current_node][n]) for n in set(range(self.graph.num_nodes)) - set(self.tabu)]])
        
        return (tau**self.alpha * eta**self.beta) / total

    def move(self, next_node):
        self.path_len += self.graph.matrix[self.current_node][next_node]
        self.traveled_edges.append((self.current_node, next_node))
        self.tabu.append(next_node)
        self.current_node = next_node                                   #Actualizamos el nodo


    def update_local_pheromone(self):
        rho = 0.1
        for i, j in self.traveled_edges:
            self.graph.pheromone[i][j] = (1 - rho) * self.graph.pheromone[i][j] + rho * self.graph.initial_pheromone
            self.graph.pheromone[j][i] = self.graph.pheromone[i][j]

    


def global_update_pheromone(graph, best_path, best_distance, decay):
    pheromone_added = 1.0 / best_distance
    for i in range(len(best_path) - 1):
        graph.pheromone[best_path[i]][best_path[i + 1]] = (1 - decay) * graph.pheromone[best_path[i]][best_path[i + 1]] + decay * pheromone_added  #disminuye la feromona en el camino recorrido
        graph.pheromone[best_path[i + 1]][best_path[i]] = graph.pheromone[best_path[i]][best_path[i + 1]]                                          #disminuye el camino de vuelta (no direccionado)
    

def ant_colony(graph, num_ants, num_iterations, alpha, beta, decay):
    best_path = None
    best_distance = float('inf')
    for iteration in range(num_iterations):
        ants = [Ant(graph, alpha, beta) for __ in range(num_ants)]
        for ant in ants:
            for __ in range(graph.num_nodes - 1):
                ant.move(ant._select_next())
            ant.path_len += graph.matrix[ant.tabu[-1]][ant.tabu[0]]
            if ant.path_len < best_distance:
                best_path = ant.tabu
                best_distance = ant.path_len
        for ant in ants:
            ant.update_local_pheromone()  # Call the new method after all ants have moved
        global_update_pheromone(graph, best_path, best_distance, decay)
    return best_path, best_distance




def read_coordinates(filename):
    coordinates = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line == 'NODE_COORD_SECTION':
                continue
            if line == 'EOF':
                break
            try:
                _, x, y = line.split()
                coordinates.append((float(x), float(y)))
            except:
                pass  # Ignore lines that don't fit the expected format
    return coordinates

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def create_adjacency_matrix(coordinates):
    n = len(coordinates)
    
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i+1, n):  # Start from i+1 to avoid duplicate calculations
            distance2 = distance(coordinates[i], coordinates[j])
            matrix[i][j] = distance2
            matrix[j][i] = distance2  # Grafo no dirigido         
    return matrix



def main():
    filename = "berlin.txt"
    coords = read_coordinates(filename)
    matrix = create_adjacency_matrix(coords)
    graph = Graph(matrix)
    
    best_path_f, best_distancef = ant_colony(graph, 50, 300, 1, 2, 0.1)

    plt.scatter([coord[0] for coord in coords], [coord[1] for coord in coords], c='r', marker='o')

    for i in range(len(best_path_f) - 1):
        plt.plot([coords[best_path_f[i]][0], coords[best_path_f[i+1]][0]],
                 [coords[best_path_f[i]][1], coords[best_path_f[i+1]][1]],
                 c='g', linestyle='-', linewidth=2, marker='o')

    # Close the path
    plt.plot([coords[best_path_f[0]][0], coords[best_path_f[-1]][0]],
             [coords[best_path_f[0]][1], coords[best_path_f[-1]][1]],
             c='g', linestyle='-', linewidth=2, marker='o')

    # Add the best distance in the plot's title
    plt.title(f"Best path distance: {best_distancef:.2f}")

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()



if __name__ == '__main__':
    main()

if __name__ == '__main__':  # This ensures that main() is called when the script is run directly
    main()










    



    

