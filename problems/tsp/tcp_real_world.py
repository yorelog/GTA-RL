import math
import os
import time

import numpy as np
import osmnx as ox
import networkx as nx

from utils.data_utils import save_dataset, load_dataset

#G = ox.graph_from_point((-37.814166, 144.961224), dist=1000, dist_type='bbox', network_type="drive", simplify=True)
#ox.save_graphml(G, "melbourne-cbd-bbox")

def load_graph(path="melbourne-cbd"):
    G = ox.load_graphml(path)
    return ox.consolidate_intersections(G, tolerance=4e-4, rebuild_graph=True,
                                  reconnect_edges=True, dead_ends=True)

def generate_coords(G):
    nodes = []
    for index, node in enumerate(G.nodes._nodes):
        nodes.append([G.nodes._nodes[index]['y'], G.nodes._nodes[index]['x']])

    edges = np.zeros(shape=(len(nodes), len(nodes)))
    for u,v,i in G.edges:
        edges[u][v] = 1

    nodes = np.array(nodes)
    return nodes, edges

def normalize_node_locs(nodes):
    nodes = nodes - nodes.min(axis=0)
    nodes = nodes / nodes.max(axis=0)
    return nodes

def select_random_nodes(batch_size, nodes, size):
    locs = []
    idxs = []
    for i in range(batch_size):
        idx = np.random.choice(len(nodes), size=size, replace=False)
        idxs.append(idx)
        locs.append(nodes[idx])
    return np.array(locs), np.array(idxs)

def generate_nx_graph(nodes, edges):

    g = nx.Graph()
    for index, node in enumerate(nodes):
        g.add_node(index, x=node[0], y=node[1])

    for i in range(len(edges)):
        for j in range(len(edges)):
            if edges[i][j] != 0:
                dist = math.sqrt((g.nodes[i]['x'] - g.nodes[j]['x']) ** 2 + (g.nodes[i]['y'] - g.nodes[j]['y']) ** 2)
                g.add_edge(i, j, distance=dist)

    return g

def generate_dynamic_graph(nodes, dynamic_locs, dynamic_locs_idx, edges):
    dynamic_nodes = np.repeat(np.expand_dims(nodes, axis=0), len(dynamic_locs), axis=0)

    for i, loc in  enumerate(dynamic_locs):
        for j, val in enumerate(loc):
            dynamic_nodes[i][dynamic_locs_idx[j]] = val

    graph_list = []
    for t in dynamic_nodes:
        graph_list.append(generate_nx_graph(t, edges))
    return graph_list

def generate_batch_dynamic_graph(nodes, batch_dynamic_locs, batch_dynamic_locs_idx, edges):
    batch_graphs = []
    for locs, idx in zip(batch_dynamic_locs, batch_dynamic_locs_idx):
        batch_graphs.append(generate_dynamic_graph(nodes, locs, idx, edges))

    return batch_graphs


import matplotlib.pyplot as plt
def plot_selection(nodes, locs):
    plt.scatter(nodes[:,0], nodes[:,1], c='b')
    plt.scatter(locs[:,0], locs[:,1], c='r')
    plt.show()

def plot_graph(g, nodes):
    pos = {i:(point[0], point[1]) for i, point in enumerate(nodes)}
    nx.draw(g, pos=pos)
    plt.show()

import itertools

def find_tour_cost(g, tour):
    tour_length = 0
    for i in range(len(tour) - 1):
        tour_length += nx.shortest_path_length(g, tour[i], tour[i + 1], 'distance')
    tour_length += nx.shortest_path_length(g, tour[len(tour) - 1], tour[0], 'distance')
    return tour_length

def find_dynamic_tour_cost(g_list, tour):
    tour_length = 0
    for i in range(len(tour) - 1):
        tour_length += nx.shortest_path_length(g_list[i], tour[i], tour[i + 1], 'distance')
        path = nx.shortest_path(g_list[i], tour[i], tour[i + 1], 'distance')
        tour_length = tour_length - g_list[i][path[-2]][path[-1]]['distance'] + g_list[i+1][path[-2]][path[-1]]['distance']
    tour_length += nx.shortest_path_length(g_list[len(tour) - 1], tour[len(tour) - 1], tour[0], 'distance')
    return tour_length

def select_shortest_combination(g, selection):
    best_cost = np.inf
    best_path = None
    for comb in itertools.permutations(selection):
        tour_length = find_tour_cost(g, comb)
        if best_cost > tour_length:
            best_cost = tour_length
            best_path = comb
            print(tour_length, comb)

    return best_cost, best_path

def select_dynamic_shortest_combination(g_list, selection):
    best_cost = np.inf
    best_path = None
    i = 0
    start = time.time()
    while time.time() - start < 600:
        comb = np.random.choice(len(selection), size=len(selection), replace=False)
        tour_length = find_dynamic_tour_cost(g_list, comb)
        if best_cost > tour_length:
            best_cost = tour_length
            best_path = comb
        i += 1

    print("Iterations: ", i)
    return best_cost, best_path

def select_batch_shortest_combinations(batch_g_list, batch_selection):
    costs = []
    for g_list, selection in zip(batch_g_list, batch_selection):
        c, t = select_dynamic_shortest_combination(g_list, selection)
        print(c, t)
        costs.append(c)
    return sum(costs)/len(costs)

import utm
def convert_to_utm(nodes):
    utm_nodes = np.zeros_like(nodes)
    for i, node in enumerate(nodes):
        x, y, _, _ = utm.from_latlon(node[0], node[1])
        utm_nodes[i][0] = x
        utm_nodes[i][1] = y

    return utm_nodes

def generate_dynamic_tsp_data(data, threshold=0.1):
    stack = []
    dataset_size, num_nodes,_ = data.shape
    init = data
    for i in range(num_nodes):
        stack.append(init)
        init = np.clip(init + np.random.uniform(-threshold, threshold, (dataset_size, num_nodes, 2)), 0, 1)

    np_stack = np.stack(stack, axis=1)

    return np_stack

def map_tours(tours, loc_ids):
    mapped_tours = []
    for tour, loc_id in zip(tours, loc_ids):
        l = []
        for t in tour:
            l.append(loc_id[t])
        mapped_tours.append(l)
    return mapped_tours

def compute_cost(g_list, mapped_tours):
    costs = []
    tours = []
    for g, tour in zip(g_list, mapped_tours):
        costs.append(find_dynamic_tour_cost(g, tour))
        tours.append(tour)
    return costs, tours


def generate_complete_dataset(path, batch_size, num_locs, seed=1234):
    np.random.seed(seed)

    name = path
    G = load_graph()
    nodes, edges = generate_coords(G)
    nodes = convert_to_utm(nodes)
    nodes = normalize_node_locs(nodes)
    num_locs = num_locs
    locs, loc_ids = select_random_nodes(batch_size, nodes, num_locs)

    # Take the batch dimension
    #batch_locs = np.expand_dims(locs, axis=0)
    batch_dynamic_locs = generate_dynamic_tsp_data(locs)

    #plot_selection(nodes, locs)
    data_dir = '../../data'
    datadir = os.path.join(data_dir, 'dynamic_tsp')
    filename = os.path.join(datadir, "{}_{}.pkl".format(name, num_locs))
    filename_g = os.path.join(datadir, "{}_graph_{}.pkl".format(name, num_locs))
    filename_l = os.path.join(datadir, "{}_node_map_{}.pkl".format(name, num_locs))
    g = generate_batch_dynamic_graph(nodes, batch_dynamic_locs[0], loc_ids, edges)

    save_dataset(batch_dynamic_locs, filename)
    save_dataset(g, filename_g)
    save_dataset(loc_ids, filename_l)

if __name__=='__main__':

    generate_complete_dataset('melbourne-cbd-temp', batch_size=4, num_locs=50)
    #filename_g = '../../data/dynamic_tsp/melbourne-cbd_graph_50.pkl'
    #filename_l = '../../data/dynamic_tsp/melbourne-cbd_node_map_50.pkl'
    #g = load_dataset(filename_g)
    #loc_ids = load_dataset(filename_l)
    #cost = select_batch_shortest_combinations(g, loc_ids)
    #print("Cost:", cost)


