import argparse

import matplotlib.pyplot as plt
import numpy as np
from utils.data_utils import load_dataset, save_dataset
from gurobipy import *
import pickle


def solve_euclidian_vrp(xy, depot, dem, timeout=None, gap=None):

    dist = {}
    demand = {}
    points = np.concatenate((depot, xy))

    CAPACITY = 1

    n = len(points)

    for i in range(len(points)):
        for j in range(len(points)):
            if i != j:
                dist[i, j] = math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(2)))
        if i != 0:
            demand[i] = dem[i-1]

    m = Model()
    m.Params.outputFlag = False

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    cap_remain = m.addVars(n, vtype=GRB.CONTINUOUS)

    for i in range(1, n):
        m.addConstr(sum(vars[j, i] for j in range(n) if i != j) == 1)
        m.addConstr(sum(vars[i, j] for j in range(n) if i != j) == 1)

    m.addConstrs(cap_remain[i] >= demand[i] for i in range(1, n))
    m.addConstrs(cap_remain[i] <= CAPACITY for i in range(1, n))

    m.addConstrs((vars[i,j] == 1) >> (cap_remain[i] + demand[i] == cap_remain[j])
                 for i in range(1, n) for j in range(1, n) if i != j)

    m._vars = vars
    m.Params.lazyConstraints = 1
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize()

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    return m.objVal, selected

def solve_dynamic_euclidian_vrp(xy, depot, dem, timeout=None, gap=None):

    dist = {}
    demand = {}
    customers = len(dem)
    depot = np.repeat(np.expand_dims(depot, 0), repeats=customers*2, axis=0)

    points = np.concatenate((depot, xy), axis=1)

    CAPACITY = 1

    time, n, _ = points.shape

    for t in range(time):
        for i in range(n):
            for j in range(n):
                nxt = 0 if (t == time - 1) else t + 1
                if i != j:
                    dist[t, i, j] = math.sqrt(sum((points[t][i][k] - points[nxt][j][k]) ** 2 for k in range(2)))
        dist[t, 0, 0] = 0.0

    for i in range(1, n):
        demand[i] = dem[i-1]

    m = Model()
    m.Params.outputFlag = False

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    cap_remain = m.addVars(n, vtype=GRB.CONTINUOUS)

    for i in range(1, n):
        m.addConstr(sum(vars[t, j, i] for t in range(time) for j in range(n) if i != j) == 1)
        m.addConstr(sum(vars[t, i, j] for t in range(time) for j in range(n) if i != j) == 1)

    for t in range(time): # At one time step there should be only one connection
        m.addConstr(sum(vars[t, i, j] for i in range(n) for j in range(n)  if i != j or (i == 0) ) == 1)

    m.addConstrs(cap_remain[i] >= demand[i] for i in range(1, n))
    m.addConstrs(cap_remain[i] <= CAPACITY for i in range(1, n))

    m.addConstrs((vars[t, i, j] == 1) >> (cap_remain[i] + demand[i] == cap_remain[j])
                 for t in range(time) for i in range(1, n) for j in range(1, n) if i != j)

    for t in range(time):
        nxt = 0 if (t == time - 1) else t + 1
        m.addConstrs((vars[t, j, i] == 1) >> (sum(vars[nxt, i, k]  for k in range(n) if k != i ) == 1)
                for i in range(1, n) for j in range(n) if i != j)

    for t in range(time):
        m.addConstr((vars[t, 0, 0] == 1) >> (sum(vars[k, 0, 0] for k in range(t,time)) == time-t))

    m._vars = vars
    m.Params.lazyConstraints = 1
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize()

    vals = m.getAttr('x', vars)
    selected = tuplelist((t, i, j) for t, i, j in vals.keys() if vals[t, i, j] > 0.5)

    return m.objVal, selected

def load_from_path(filename):

    assert os.path.splitext(filename)[1] == '.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data

def solve_all_vrp(dataset, dynamic, time):
    total_length = 0
    i = 0
    for depot, xy, demand, cap in load_from_path(dataset):
        xy = np.array(xy)
        depot = np.array([depot])
        demand = np.array(demand) / cap
        length, tour = solve_dynamic_euclidian_vrp(xy, depot, demand, timeout=time)
        total_length += length
        print("Solved instance {} with tour length {}".format(i, length))
        i += 1

    print("Average Legnth {}".format(total_length/i))

import matplotlib.cm as cm
if __name__=="__main__":

    solve_all_vrp("../../data/dynamic_vrp/dynamic_vrp20_validation_seed4321.pkl", dynamic=True, time=300)

    size = 10

    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }

    np.random.seed(1234)

    xy = np.random.uniform(0, 1, (size*2, size, 2))
    depot = np.random.uniform(0, 1, (1, 2))
    demand = (np.random.uniform(0, 9, (size)) + 1) /  CAPACITIES[size]

    obj, tour = solve_dynamic_euclidian_vrp(xy, depot, demand, timeout=600)

    depot = np.repeat(np.expand_dims(depot, 0), repeats=size*2, axis=0)
    xy = np.concatenate((depot, xy), axis=1)

    #xy = np.concatenate((depot, xy))
    for t, i, j in tour:
        next = 0 if t == size*2-1 else t+1
        plt.plot([xy[t][i][0], xy[next][j][0]], [xy[t][i][1], xy[next][j][1]], c='g', zorder=0)
    plt.plot(xy[0][0][0], xy[0][0][1], c='r', marker='s')

    colors = cm.rainbow(np.linspace(0, 1, size))
    for i in range(size):
        plt.scatter(xy[:, i, 0], xy[:, i, 1], c=colors[i])
    plt.show()


