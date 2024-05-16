#Uses formulation 6, even though we don't have a probabalistic interpretation
#Local constraint only
import numpy as np
import pandas as pd
import os

from gurobipy import quicksum
from gurobipy import Model
from gurobipy import GRB

def solve_adversary(alpha,x,p,B):
    M=10000 #TODO change

    #set up model
    m = Model()
    vtype = GRB.BINARY
    I,J=alpha.shape
    print(I,J)

    #define variables
    z = {} #1 if station i is made feasible, 0 else
    for i in range(I):
        z[i] = m.addVar(vtype=vtype, name="z(%s)" % i)
    s = {} #1 if block j has demand met, 0 else
    for j in range(J):
        s[j] = m.addVar(vtype=vtype, name="s(%s)" % j)

    #add constraints
    for j in range(J):
        m.addConstr(quicksum(alpha[i,j]*x[i]*z[i] for i in range(I)) <= M*s[j],
                        name="satisfied_%s" % j)
    #m.addConstr(I-quicksum(z[i] for i in range(I)) <= -1*np.log(B)*quicksum(1-p[i] for i in range(I)),
    #                    name="knapsack")
    #m.addConstr(quicksum(z[i] *np.log(1-p[i])for i in range(I)) <= quicksum(1-p[i] for i in range(I))-quicksum(np.log(B) for i in range(I)),
    #                    name="knapsack2")
    m.addConstr(quicksum((1-z[i]) *np.log(1-p[i])for i in range(I)) >= quicksum(x[i]*np.log(B) for i in range(I)),
                        name="knapsack3")
    
    #set objective
    m.setObjective(quicksum(1-s[j] for j in range(J)), GRB.MAXIMIZE)

    #solve
    m.optimize()
    infeas_stations=[]
    unmatched_blocks=[]
    for v in m.getVars():
        if v.x<1 and 'z' in v.varName:
            infeas_stations.append(int(v.varName[2:-1]))
        if v.x<1 and 's' in v.varName:
            unmatched_blocks.append(int(v.varName[2:-1]))
    print('Max utility:',  m.objVal)
    infeas_stations=np.array(infeas_stations)
    unmatched_blocks=np.array(unmatched_blocks)
    print(unmatched_blocks)
    print(infeas_stations)
    #print(z.X)

    return(0)

def adversarial_soln(max_dist):
    alpha=np.load('Robust/alpha%s.npy' %max_dist)
    charger_df=pd.read_csv('Robust/chargers_df.csv')
    p=charger_df['prob'].to_numpy()

    selected_df=pd.read_csv('Robust/selected_df%s.csv' %max_dist)
    selected_ids=selected_df['id2']
    charger_df=charger_df.set_index('id2')
    charger_df['selected']=0
    print(len(selected_ids))
    for id in selected_ids: #TODO inefficient
        charger_df.loc[id, 'selected']=1
        x=charger_df['selected'].to_numpy()
    x=charger_df['selected'].to_numpy()
    print(sum(x))

    B=.95 #Budget

    solve_adversary(alpha,x,p,B)

if __name__ == "__main__":
    adversarial_soln(300)