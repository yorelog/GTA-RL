#!/usr/bin/python
import numpy as np
import math
import random
import copy

class Ant():

    def __init__(self, alpha, beta, n, id):
        self.cost = 0
        self.alpha = alpha
        self.beta = beta
        self.visited = np.zeros(shape=n)
        self.current_node = 0
        self.path = []
        self.step_num = 0
        self.n = n
        self.id = id

    def step(self, graph, first_iter, deterministic=False, best_tour=False):
        if self.step_num == 0:
            next_node = 0 #random.randint(0, self.n-1)
        elif first_iter:
            next_node = self.random_next_node()
        else:
            next_node = self.compute_next_node(graph, deterministic, best_tour)

        self.current_node = next_node
        self.visited[self.current_node] = 1
        self.path.append(self.current_node)
        self.step_num += 1

    def compute_next_node(self, graph, deterministic=False, best_tour=False):
        pi = np.zeros(shape=self.n)
        for i in range(self.n):
            if self.visited[i] == 0:
                pi[i] = graph.get_phero(self.current_node, i, self.step_num, best_tour)**self.alpha * \
                        graph.get_reci_dist(self.current_node, i, self.step_num)**self.beta
            else:
                pi[i] = 0

        # Nan is if distance is zero: These nodes should get priority
        pi = np.nan_to_num(pi, nan=1)

        if sum(pi) == 0:
            return self.random_next_node()

        pi = pi/sum(pi)

        if deterministic:
            index = np.argmax(pi)
        else:
            if sum(pi) == 0:
                return self.random_next_node()
            index = np.random.choice(self.n, 1, p=pi)[0]
        return index

    def random_next_node(self):
        pi = np.where(self.visited == 0)[0]
        index = np.random.choice(pi, 1)[0]
        return index

    def is_done(self):
        return self.step_num == self.n

    def get_path_cost(self, graph):
        coords = graph.get_path_coords(self.path)
        cost = np.linalg.norm(coords - np.roll(coords, 1, axis=0), axis=1).sum()
        return cost, self.path

    def reset(self):
        self.step_num = 0
        self.path = []
        self.visited = np.zeros(shape=self.n)

class DynamicGraph():

    def __init__(self, xy, demand, p):



        dist = []
        n = len(xy)
        for t in range(n):
            time = []
            for i in range(n):
                row = []
                for j in range(n):
                    nxt = 0 if (t == n - 1) else t + 1
                    a = math.sqrt(sum((xy[t][i][k] - xy[nxt][j][k]) ** 2 for k in range(2)))
                    row.append(a)
                time.append(row)
            dist.append(time)

        self.dist = np.array(dist)
        self.reciprocal_dist = np.reciprocal(dist)

        self.xy = xy
        self.phero_matrix = np.full(self.dist.shape, 20)
        self.best_phero_matrix = self.phero_matrix
        self.p = p

        self.phero_max = 10
        self.phero_min = 0.1

    def update_phero(self, path_costs):

        cost_matrix = np.zeros(self.dist.shape)

        for cost, path in path_costs:
            for i in range(len(path)-1):
                cost_matrix[i][path[i]][path[i+1]] += 1/cost
            cost_matrix[len(path)-1][path[len(path)-1]][path[0]] += 1/cost

        self.phero_matrix = (1-self.p)*self.phero_matrix + cost_matrix

    def clip_phero(self):
        self.phero_matrix = np.clip(self.phero_matrix, self.phero_min, self.phero_max)

    def get_reci_dist(self, current_node, next_node, time=0):
        return self.reciprocal_dist[time][current_node][next_node]

    def get_phero(self, current_node, next_node, time=0, use_best=False):
        if use_best:
            return self.best_phero_matrix[time][current_node][next_node]
        return self.phero_matrix[time][current_node][next_node]

    def get_path_coords(self, path):
        return np.diagonal(np.take(self.xy, path, axis=1), axis1=0, axis2=1).T

    def save_best_tour(self):
        self.best_phero_matrix = copy.deepcopy(self.phero_matrix)

class Graph():

    def __init__(self, xy, depot, demand, p):

        customers = len(xy)
        depot = np.repeat(np.expand_dims(depot, 0), repeats=customers * 2, axis=0)
        points = np.concatenate((depot, xy), axis=1)

        dist = []
        n = len(points)
        for i in range(n):
            row = []
            for j in range(n):
                row.append(math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2))))
            dist.append(row)

        self.dist = np.array(dist)
        self.reciprocal_dist = np.reciprocal(dist)
        self.demand = demand

        self.xy = points
        self.phero_matrix = np.full(self.dist.shape, 20)
        self.best_phero_matrix = self.phero_matrix
        self.p = p

    def update_phero(self, path_costs):

        cost_matrix = np.zeros(self.dist.shape)

        for cost, path in path_costs:
            for i in range(len(path)-1):
                cost_matrix[path[i]][path[i+1]] += 1/cost
                cost_matrix[path[i+1]][path[i]] += 1/cost

        self.phero_matrix = (1-self.p)*self.phero_matrix + cost_matrix

    def get_reci_dist(self, current_node, next_node, time=0):
        return self.reciprocal_dist[current_node][next_node]

    def get_phero(self, current_node, next_node, time=0, use_best=False):
        if use_best:
            return self.best_phero_matrix[current_node][next_node]
        return self.phero_matrix[current_node][next_node]

    def get_path_coords(self, path):
        return np.take(self.xy, path, axis=0)

    def save_best_tour(self):
        self.best_phero_matrix = self.phero_matrix

class ACO():

    def __init__(self, xy, iterations=1000, k=50, alpha=1.0, beta=5.0, p=0.2, type=None):
        if len(xy.shape) == 2:
            self.graph = Graph(xy, p)
        else:
            self.graph = DynamicGraph(xy, p)

        self.ants = [Ant(alpha, beta, len(xy), i) for i in range(k)]
        self.iterations = iterations
        self.first = True
        self.type = type

        self.test_ant = Ant(alpha, beta, len(xy), -1)
        self.best_tour_cost = np.inf
        self.best_tour = []

    def run(self, log=False):
        for i in range(self.iterations):
            while(not self.all_done()):
                for ant in self.ants:
                    ant.step(self.graph, self.first)

            self.update_phero()
            self.first = False
            result = self.simulate()
            if log:
                print("Iteration: ", i, " Cost: ", result)

    def update_phero(self):

        if self.type == 'min-max':
            lowest_cost = np.inf
            lowest_path = []
            for ant in self.ants:
                path_cost, path = ant.get_path_cost(self.graph)
                if lowest_cost > path_cost:
                    lowest_cost = path_cost
                    lowest_path = path
                ant.reset()
            self.graph.update_phero([[lowest_cost, lowest_path]])
            #self.graph.clip_phero()
        else:
            path_costs = []
            for ant in self.ants:
               path_costs.append(ant.get_path_cost(self.graph))
               ant.reset()
            self.graph.update_phero(path_costs)



    def simulate(self, best_tour=False):
        while not self.test_ant.is_done():
            self.test_ant.step(self.graph, self.first, True, best_tour)
        tour_cost, path = self.test_ant.get_path_cost(self.graph)
        if tour_cost < self.best_tour_cost:
            self.best_tour_cost = tour_cost
            self.best_tour = path
            self.graph.save_best_tour()

        self.test_ant.reset()
        return tour_cost, path

    def get_best_tour(self):
        return  self.best_tour_cost, self.best_tour

    def all_done(self):
        done = True
        for ant in self.ants:
            if not ant.is_done():
                done = False
        return done

from problems.tsp.tsp_gurobi  import plot_tsp, get
import matplotlib.pyplot as plt
import time

def solve_all_aco(dataset):
    results = []
    start_time = time.time()
    for i, instance in enumerate(dataset):
        aco = ACO(instance, iterations=200, alpha=1, beta=5, p=0.2, k=20,
              type='min-max')
        aco.run()
        result = aco.get_best_tour()
        print("Solved instance {} with tour length {} : Solved in {} seconds".format(i, result, time.time()-start_time))
        results.append(result)
    return results

if __name__=="__main__":
    np.random.seed(1234)
    xy = np.random.uniform(0, 1, (20, 2))
    #xy = get(20, 0.1)

    aco = ACO(xy, iterations=1000, alpha=1, beta=5, p=0.2, k=20,
              type='min-max')
    aco.run(log=True)
    cost, tour = aco.get_best_tour()
    print("Best cost", cost, tour)

    fig, ax = plt.subplots(figsize=(10, 10))

    if len(xy.shape) == 2:
        plot_tsp(xy, tour, ax)
    else:
        ordered = np.array([np.arange(len(tour)), tour]).T
        ordered = ordered[ordered[:, 1].argsort()]
        coords = xy[ordered[:, 0], ordered[:, 1]]
        plot_tsp(coords, tour, ax, xy)

    plt.show()