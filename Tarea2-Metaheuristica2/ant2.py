import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

class ValidatorBuilder:
    def __init__(self):
        self._validator = lambda x: x

    def type(self, dtype):
        self._validator = lambda x: dtype(x)
        return self

    def start(self, minimum):
        old_validator = self._validator
        self._validator = lambda x: old_validator(x) if x >= minimum else None
        return self

    def end(self, maximum):
        old_validator = self._validator
        self._validator = lambda x: old_validator(x) if x <= maximum else None
        return self

    def build(self):
        return self._validator

class Parameters:
    input_file: str = "berlin.txt"
    seed: int = 0
    colony_size: int = 100
    iterations: int = 250
    alpha: float = 1
    beta: float = 2.5
    q0: float = 0.9
    dont_show_result: bool = False
    dont_track_time: bool = False

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        parser.add_argument("input_file", 
                            help="name of the file that's going to be read",
                            nargs='?', type=str, default=self.input_file)
        parser.add_argument("seed",
                            help="seed to be used for randomic functionality",
                            nargs='?', type=int, default=self.seed)
        parser.add_argument("colony_size",
                            help="colony size or number of ants",
                            nargs='?', type=ValidatorBuilder().type(int).start(0).build(), default=self.colony_size)
        parser.add_argument("iterations",
                            help="number of iterations before stopping algorithm",
                            nargs='?', type=ValidatorBuilder().type(int).start(0).build(), default=self.iterations)
        parser.add_argument("alpha",
                            help="pheromone evaporation factor",
                            nargs='?', type=float, default=self.alpha)
        parser.add_argument("beta",
                            help="the weight of the heuristic value",
                            nargs='?', type=float, default=self.beta)
        parser.add_argument("q0",
                            help="limit probability value",
                            nargs='?', type=ValidatorBuilder().type(float).start(0.0).end(1.0).build(), default=self.q0)
        parser.add_argument("--no-result", dest='dont_show_result',
                            action="store_true",
                            help="hides the result at the end of the algorithm",
                            default=self.dont_show_result)
        parser.add_argument("--no-time", dest='dont_track_time',
                            action="store_true",
                            help="skip tracking algorithm completion time",
                            default=self.dont_track_time)
        
        parser.parse_args(args=args, namespace=self)

class Graph:
    def __init__(self, matrix):
        self.matrix = matrix
        self.num_nodes = matrix.shape[0]
        self.pheromone = np.full((self.num_nodes, self.num_nodes), 1 / (self.num_nodes * self.num_nodes))

class Ant:
    def __init__(self, graph, alpha, beta, q0):
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        self.current_node = random.randint(0, graph.num_nodes - 1)
        self.tabu = [self.current_node]
        self.path_len = 0.0
        self.traveled_edges = []

    def _select_next(self):
        unvisited = np.setdiff1d(np.arange(self.graph.num_nodes), self.tabu)
        if random.uniform(0, 1) <= self.q0:
            # Selección basada en max product of pheromone and heuristic value
            return unvisited[np.argmax(self.graph.pheromone[self.current_node, unvisited] *
                                       (1 / self.graph.matrix[self.current_node, unvisited]) ** self.beta)]
        else:
            # Selección basada en función de probabilidad
            probabilities = np.array([self.probability(n) for n in unvisited])
            return np.random.choice(unvisited, p=probabilities/sum(probabilities))

    def probability(self, next_node):
        tau = self.graph.pheromone[self.current_node, next_node]
        eta = 1 / self.graph.matrix[self.current_node, next_node]
        return (tau ** self.alpha) * (eta ** self.beta)

    def move(self, next_node):
        self.path_len += self.graph.matrix[self.current_node, next_node]
        self.traveled_edges.append((self.current_node, next_node))
        self.tabu.append(next_node)
        self.current_node = next_node

    def update_local_pheromone(self, tau0):
        for i, j in self.traveled_edges:
            self.graph.pheromone[i, j] = (1 - self.alpha) * self.graph.pheromone[i, j] + self.alpha * tau0
            self.graph.pheromone[j, i] = self.graph.pheromone[i, j]

def global_update_pheromone(graph, best_path, best_distance, decay, delta):
    for i in range(len(best_path) - 1):
        graph.pheromone[best_path[i], best_path[i+1]] = (1 - decay) * graph.pheromone[best_path[i], best_path[i+1]] + decay * delta
        graph.pheromone[best_path[i+1], best_path[i]] = graph.pheromone[best_path[i], best_path[i+1]]

def ant_colony(graph, num_ants, num_iterations, alpha, beta, decay, q0):
    best_path = None
    best_distance = np.inf
    tau0 = 1 / (graph.num_nodes * np.mean(graph.matrix))
    for _ in range(num_iterations):
        ants = [Ant(graph, alpha, beta, q0) for _ in range(num_ants)]
        for ant in ants:
            for _ in range(graph.num_nodes - 1):
                ant.move(ant._select_next())
            ant.path_len += graph.matrix[ant.tabu[-1], ant.tabu[0]]
            if ant.path_len < best_distance:
                best_path = ant.tabu
                best_distance = ant.path_len
        for ant in ants:
            ant.update_local_pheromone(tau0)
        delta = 0 if best_distance == np.inf else 1.0 / best_distance
        global_update_pheromone(graph, best_path, best_distance, decay, delta)
    return best_path, best_distance

def read_coordinates(filename):
    coords = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line in ['NODE_COORD_SECTION', 'EOF']:
                continue
            try:
                _, x, y = line.split()
                coords.append([float(x), float(y)])
            except ValueError:
                pass
    return np.array(coords)

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def create_adjacency_matrix(coordinates):
    n = len(coordinates)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = distance(coordinates[i], coordinates[j])
            matrix[i, j] = dist
            matrix[j, i] = dist
    return matrix

def plot_graph(coordinates, best_path):
    fig, ax = plt.subplots()
    ax.plot(coordinates[:, 0], coordinates[:, 1], 'o')
    for i in range(1, len(best_path)):
        start = coordinates[best_path[i-1]]
        end = coordinates[best_path[i]]
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=1.5))
    start = coordinates[best_path[-1]]
    end = coordinates[best_path[0]]
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=1.5))
    plt.show()

def main(filename):

    params = Parameters()
    params.parse_args()
    print("partio")
    coordinates = read_coordinates(params.input_file)
    matrix = create_adjacency_matrix(coordinates)
    graph = Graph(matrix)
    best_path, best_distance = ant_colony(graph, num_ants= params.colony_size, num_iterations=params.iterations, alpha=params.alpha, beta=params.beta, decay=params.alpha, q0=params.q0)
    print(f"Best path: {best_path}")
    print(f"Best distance: {best_distance}")
    plot_graph(coordinates, best_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ant Colony Optimization Algorithm.')
    parser.add_argument('filename', type=str, help='File name with node coordinates.')
    args = parser.parse_args()
    main(args.filename)
