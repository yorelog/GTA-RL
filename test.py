import os
import numpy as np
import torch
import time
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from utils import load_model
from problems.tsp.tsp_gurobi import *

  # Put in evaluation mode to not track gradients


def get(num_nodes, threshold):
    stack = []
    init = np.random.uniform(0, 1, (num_nodes, 2))
    for i in range(num_nodes):
        stack.append(init)
        init = np.clip(init + np.random.uniform(-threshold, threshold, (num_nodes, 2)), 0, 1)

    np_stack = np.array(stack)

    return np_stack

def make_oracle(model, xy, temperature=1.0):
    num_nodes = len(xy)

    if len(xy.shape) == 3:
        xyt = torch.tensor(xy[0]).float()[None]  # Add batch dimension
    else:
        xyt = torch.tensor(xy).float()[None]

    with torch.no_grad():  # Inference only
        embeddings, _ = model.embedder(model._init_embed(xyt))

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = model._precompute(embeddings)

    def oracle(tour):
        with torch.no_grad():  # Inference only
            # Input tour with 0 based indices
            # Output vector with probabilities for locations not in tour
            tour = torch.tensor(tour).long()
            if len(tour) == 0:
                step_context = model.W_placeholder
            else:
                step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

            # Compute query = context node embedding, add batch and step dimensions (both 1)
            query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

            # Create the mask and convert to bool depending on PyTorch version
            mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
            mask[tour] = 1
            mask = mask[None, None, :]  # Add batch and step dimension

            log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
            p = torch.softmax(log_p / temperature, -1)[0, 0]
            assert (p[tour] == 0).all()
            assert (p.sum() - 1).abs() < 1e-5
            # assert np.allclose(p.sum().item(), 1)
        return p.numpy()

    return oracle

def make_dynamic_oracle(model, xy, temperature=1.0):
    num_nodes = len(xy)

    xyt = torch.tensor(xy).float()[None]  # Add batch dimension

    with torch.no_grad():  # Inference only
        embeddings_all = model._init_embed(xyt)
        embeddings_all, _ = model.embedder(embeddings_all)
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step

    def oracle(tour):
        with torch.no_grad():  # Inference only
            # Input tour with 0 based indices
            # Output vector with probabilities for locations not in tour
            embeddings = embeddings_all[:, len(tour), :, :]
            fixed = model._precompute(embeddings)

            tour = torch.tensor(tour).long()
            if len(tour) == 0:
                step_context = model.W_placeholder
            else:
                step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

            # Compute query = context node embedding, add batch and step dimensions (both 1)
            query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])

            # Create the mask and convert to bool depending on PyTorch version
            mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
            mask[tour] = 1
            mask = mask[None, None, :]  # Add batch and step dimension

            log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
            p = torch.softmax(log_p / temperature, -1)[0, 0]
            assert (p[tour] == 0).all()
            assert (p.sum() - 1).abs() < 1e-5
            # assert np.allclose(p.sum().item(), 1)
        return p.numpy()

    return oracle




def find_tour(xy, model, test_dynamic):

    model.eval()

    oracle = make_dynamic_oracle(model, xy) if test_dynamic else make_oracle(model, xy)
    sample = False
    tour = []
    tour_p = []
    while (len(tour) < len(xy)):
        p = oracle(tour)

        if sample:
            # Advertising the Gumbel-Max trick
            g = -np.log(-np.log(np.random.rand(*p.shape)))
            i = np.argmax(np.log(p) + g)
            # i = np.random.multinomial(1, p)
        else:
            # Greedy
            i = np.argmax(p)
        tour.append(i)
        tour_p.append(p)
    return tour_p, tour






# Code inspired by Google OR Tools plot:
# https://github.com/google/or-tools/blob/fb12c5ded7423d524fc6c95656a9bdc290a81d4d/examples/python/cvrptw_plot.py

def plot_tsp(xy, tour, ax1, total_xy=None, title=""):
    """
    Plot the TSP tour on matplotlib axis ax1.
    """

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    xs, ys = xy[tour].transpose()
    xs, ys = xy[tour].transpose()
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    d = np.sqrt(dx * dx + dy * dy)
    lengths = d.cumsum()

    # Scatter nodes

    # Starting node
    ax1.scatter([xs[0]], [ys[0]], s=100, color='red')

    if total_xy is not None:
        time, nodes, coords = total_xy.shape
        flatten_xy =  total_xy.reshape((time*nodes, coords))
        colors = cm.rainbow(np.linspace(0, 1, nodes))

        for i in range(nodes):
            ax1.scatter(total_xy[:, i, 0], total_xy[:, i, 1], s=20, color=colors[i])
            ax1.scatter(xy[i][0], xy[i][1], s=100, color=colors[i])

    else:
        ax1.scatter(xs, ys, s=100, color='blue')

    # Arcs
    qv = ax1.quiver(
        xs, ys, dx, dy,
        scale_units='xy',
        angles='xy',
        scale=1,
    )

    ax1.set_title('Algorithm {}, {} nodes, total length {:.2f}'.format(title, len(tour), lengths[-1]))

def plot_tsp_with_data(xy, tour, title=""):
    fig, ax = plt.subplots(figsize=(10, 10))
    if len(xy.shape) == 3:
        ordered = np.array([np.arange(len(tour)), tour]).T
        ordered = ordered[ordered[:, 1].argsort()]
        coords = xy[ordered[:, 0], ordered[:, 1]]
        plot_tsp(coords, tour, ax, xy, title)
    else:
        plot_tsp(xy, tour, ax)
    plt.show()


def run_test(opts):

    if opts.dynamic:
        xy = get(opts.graph_size, opts.intensity)
    else:
        xy = np.random.uniform(0, 1, (opts.graph_size, 2))

    model, _ = load_model(path=opts.load_path)

    tour_p, tour = find_tour(xy, model, not opts.baseline)

    ### Plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    if len(xy.shape) == 3:
        ordered = np.array([np.arange(len(tour)), tour]).T
        ordered = ordered[ordered[:, 1].argsort()]
        coords = xy[ordered[:, 0], ordered[:, 1]]
        plot_tsp(coords, tour, ax, xy, "RL: ")
    else:
        plot_tsp(xy, tour, ax)

    print("RL: ", tour)

    if opts.use_gurobi:

        fig1, ax1 = plt.subplots(figsize=(10, 10))
        start_time = time.time()
        tour_length, tour_gb = solve_dynamic_euclidian_tsp(xy)
        print("Gurobi Time: ", time.time() - start_time)
        if len(xy.shape) == 3:
            ordered = np.array([np.arange(len(tour_gb)), tour_gb]).T
            ordered = ordered[ordered[:, 1].argsort()]
            coords = xy[ordered[:, 0], ordered[:, 1]]
            plot_tsp(coords, tour_gb, ax1, xy, "LP: ")
        else:
            plot_tsp(xy, tour_gb, ax1)

        print("LP: ", tour_gb)
    plt.show()

np.random.seed(1234)

def test_gurobi(filename, dynamic, time=60):

    data = load_from_path(filename)
    results = solve_all_gurobi(data, dynamic, time)
    lengths = np.array(results)[:, 0]
    print("Results: ", np.mean(lengths))
    return results


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic", action='store_true', help="Solve the Dynamic TSP")
    parser.add_argument('--graph_size', type=int, default=20, help="The size of the problem graph")
    parser.add_argument('--intensity', type=int, default=0.1, help="How much dynamic nodes should change over time")
    parser.add_argument("--use_gurobi", action='store_true', help="Use gurobi optimizer to solve the TSP")
    parser.add_argument("--baseline", action='store_true', help="Use static baseline")
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--load_data', help='Path to load dataset')
    parser.add_argument('--gurobi_time', type=int, default=6000, help="Time limit for Gurobi Solver")
    parser.add_argument('--problem', type=str, default=20, help="Problem to solve")

    opts = parser.parse_args(args=None)
    opts.use_gurobi = True
    opts.dynamic = True
    opts.baseline = False

    if not opts.baseline:
        opts.load_path = 'outputs/order/dynamic_tsp_20/run_4'
    else:
        opts.load_path = 'pretrained/tsp_20/'

    if opts.dynamic:
        opts.load_data = 'data/dynamic_tsp/dynamic_tsp10_validation_seed4321.pkl'
    else:
        opts.load_data = 'data/tsp/tsp20_test_seed1234.pkl'

    if opts.use_gurobi:
        test_gurobi(opts.load_data, opts.dynamic, opts.gurobi_time)
    else:
        run_test(opts)

# %%