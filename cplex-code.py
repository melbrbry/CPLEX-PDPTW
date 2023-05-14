#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 22:26:49 2023

@author: elbarbari
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance_matrix
import sys
import docplex.mp
from docplex.mp.model import Model
import pickle 

no_of_packages = 10
packages = [i for i in range(1, no_of_packages+1)]
machines = [i for i in range(no_of_packages+1, (no_of_packages)*2 +1)]
nondepot =  packages + machines
nodes = [0] +nondepot + [(2 * no_of_packages) + 1]
M = 10000  #Big M
time_limit = 500

def get_constraints(no_of_packages):
    ReadyTime = []
    DueTime = []
    constraints = pickle.load(open("./grids/constraints_10", "rb"))
    
    for it in constraints:
        print(it)
        
    ReadyTime.append(0)
    DueTime.append(time_limit)
    
    for const_id, const in enumerate(constraints):
        if const_id < no_of_packages:
            ReadyTime.append(const[0][0])
            DueTime.append(const[0][1])
    
    for const_id, const in enumerate(constraints):
        if const_id < no_of_packages:
            ReadyTime.append(const[1][0])
            DueTime.append(const[1][1])    
    
    ReadyTime.append(0)
    DueTime.append(time_limit)
    
    return ReadyTime, DueTime

def get_coordinates(no_of_packages):
    XCOORD = []
    YCOORD = []
    pos = dict()
    grid = pickle.load(open("./grids/grid_10", "rb"))
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] not in [".", "*"]:
                pos[grid[i][j]] = (i, j)
    XCOORD.append(pos["S"][0])
    YCOORD.append(pos["S"][1])
    for i in range(1, no_of_packages+1):
        XCOORD.append(pos["p" + str(i)][0])
        YCOORD.append(pos["p" + str(i)][1])
    for i in range(1, no_of_packages+1):
        XCOORD.append(pos["m" + str(i)][0])
        YCOORD.append(pos["m" + str(i)][1])
    XCOORD.append(pos["S"][0])
    YCOORD.append(pos["S"][1])
    return XCOORD, YCOORD

requests = [0] + [1] * no_of_packages + [-1] * no_of_packages + [0]
ReadyTime, DueTime = get_constraints(no_of_packages)
XCOORD, YCOORD = get_coordinates(no_of_packages)

print(XCOORD)
print(YCOORD)
print(ReadyTime)
print(DueTime)


arcs = [(i, j) for i in nodes for j in nodes]

# Manhattan Distance
distance = {(i, j): abs(XCOORD[i]-XCOORD[j]) + abs(YCOORD[i]-YCOORD[j]) for i, j in arcs}
travel_time = distance

time_start = time.time() # for loging
mdl = Model('PDPTW')

# indicates whether the agent travels from node 𝑖 to node 𝑗 (exluding traveling to the same node)
X = [(i, j) for i in nodes for j in nodes if i != j] 
# indicates the capacity after visiting node 𝑖
Q = [i for i in nodes] 
# indicates the time of visiting node 𝑖
S = [i for i in nodes] 

x = mdl.binary_var_dict(X, name='x')
q = mdl.binary_var_dict(Q, name='q')
s = mdl.integer_var_dict(S, name='s')

# Objective Function: finish the task asap
obj_function = mdl.sum(s[no_of_packages*2+1])

# each nondepot should appear as i once (with other conditions, this ensures that each node is visited exactly once)
mdl.add_constraints(mdl.sum(x[i,j] for j in nodes if i!=j)==1 for i in nondepot)

# the start depot should appear as i exactly once (the route should start at the start depot)
mdl.add_constraint(mdl.sum(x[0,j] for j in nodes if j!=0)==1)

# the terminal depot should appear as j exactly once (the route should end at the terminal depot)
mdl.add_constraint(mdl.sum(x[i, 2 * no_of_packages + 1] for i in nodes if i!=2*no_of_packages+1) == 1)

# the agent cannot go from the start depot directly to the terminal depot
mdl.add_constraint(x[0, 2 * no_of_packages + 1] == 0)

# the agent should leave the node it visits 
mdl.add_constraints((mdl.sum(x[i,j] for j in nodes if j!=i) - mdl.sum(x[j,i] for j in nodes if j!=i) == 0) for i in nondepot)

# time constraints
mdl.add_constraints((s[j] >= s[i] + travel_time[i,j] - M * (1- x[i,j])) for i , j in X)

# capacity constraints
mdl.add_constraints(q[j] >= q[i] + requests[j] - M * (1- x[i,j]) for i, j in X)

# precedence constraints
mdl.add_constraints((s[i] + travel_time[i,no_of_packages+i] <= s[no_of_packages+i]) for i in packages)

# time windows
mdl.add_constraints(ReadyTime[i] <= s[i] for i in S)
mdl.add_constraints(s[i] <= DueTime[i] for i in S)


# Set time limit
mdl.parameters.timelimit.set(1000)

# Solve
mdl.minimize(obj_function)

time_solve = time.time() # for log

solution = mdl.solve(log_output = True)

time_end = time.time() # for log

def generate_ordering(node_id):
    if node_id == 0:
        print("start depot", times[node_id])
    elif node_id <= no_of_packages: 
        print("package", node_id, times[node_id])
    elif node_id == no_of_packages * 2 + 1:
        print("terminal depot", obj)
        return     
    elif node_id > no_of_packages:
        print("machine", node_id - no_of_packages, times[node_id])
    
    for arc in active_arcs:
        if arc[0] == node_id:
            generate_ordering(arc[1])


if __name__ == '__main__':
    if solution != None:
        obj = round(obj_function.solution_value, 2) # to access the obj
    running_time = round(time_end - time_solve, 2)
    elapsed_time = round(time_end - time_start, 2)
    print('----------------------------------------------------------') 
    print('Best_Solution:',obj,' RunningTime: ',running_time,' ElapsedTime: ',elapsed_time)
    print('----------------------------------------------------------')      
    active_arcs = [a for a in X if x[a].solution_value > 0.9]
    times = [int(s[i].solution_value) for i, j in X if x[i, j].solution_value > 0.9]
    print('Routes: ',active_arcs)
    print('----------------------------------------------------------')  
    print('Ordering:')
    generate_ordering(0)
