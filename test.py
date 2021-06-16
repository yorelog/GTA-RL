import os
import numpy as np
import torch

from utils import load_model
from problems.tsp.tsp_gurobi import *

test_dynamic = True

if test_dynamic:
    model, _ = load_model('outputs/dynamic_tsp_20/run_20210613T202716')
else:
    model, _ = load_model('outputs/tsp_20/run_20210608T170039')
    model, _ = load_model('pretrained/tsp_20/')


model.eval()  # Put in evaluation mode to not track gradients

num_nodes = 20
seed = None

np.random.seed(1234)

def get(num_nodes, threshold):
    stack = []
    init = np.random.uniform(0, 1, (num_nodes, 2))
    for i in range(num_nodes):
        stack.append(init)
        init = np.clip(init + np.random.uniform(-threshold, threshold, (num_nodes, 2)), 0, 1)

    np_stack = np.array(stack)

    return np_stack

xy = get(num_nodes, 0.1)
#xy = np.random.uniform(0, 1, (20, 2))



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



from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


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



fig, ax = plt.subplots(figsize=(10, 10))

if len(xy.shape) == 3:
    ordered = np.array([np.arange(len(tour)), tour]).T
    ordered = ordered[ordered[:,1].argsort()]
    coords = xy[ordered[:,0], ordered[:,1]]
    plot_tsp(coords, tour, ax, xy, "RL: ")
else:
    plot_tsp(xy, tour, ax)

print("RL: ", tour)

fig1, ax1 = plt.subplots(figsize=(10, 10))

tour_length, tour = solve_dynamic_euclidian_tsp(xy)
if len(xy.shape) == 3:
    ordered = np.array([np.arange(len(tour)), tour]).T
    ordered = ordered[ordered[:,1].argsort()]
    coords = xy[ordered[:,0], ordered[:,1]]
    plot_tsp(coords, tour, ax1, xy, "LP: ")
else:
    plot_tsp(xy, tour, ax1)

print("LP: ", tour)
plt.show()
# %%