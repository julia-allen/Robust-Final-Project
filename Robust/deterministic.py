import numpy as np
import pandas as pd
import os

from gurobipy import quicksum
from gurobipy import Model
from gurobipy import GRB

from sampling import check_feas

def basic_model(alpha,c,lamb,z):
    #set up the model
    m = Model()
    vtype = GRB.BINARY
    I,J=alpha.shape
    print(I,J)

    #define variables
    x = {} #1 if station i is opened, 0 else
    for i in range(I):
        x[i] = m.addVar(vtype=vtype, name="x(%s)" % i)
    s = {} #1 if block j has demand met, 0 else
    for j in range(J):
        s[j] = m.addVar(vtype=vtype, name="s(%s)" % j)

    #add constraints
    for j in range(J):
        m.addConstr(quicksum(alpha[i,j]*x[i]*z[i] for i in range(I)) >= s[j],
                        name="satisfied_%s" % j)
        
    #set objective
    m.setObjective(c*quicksum(x[i] for i in range(I))+lamb*quicksum(1-s[j] for j in range(J)), GRB.MINIMIZE)
    #TODO should penalty be proportional to population?

    #solve
    m.optimize()
    selected_stations=[]
    unmatched_blocks=[]
    for v in m.getVars():
        if v.x>0 and 'x' in v.varName:
            selected_stations.append(int(v.varName[2:-1]))
        if v.x<1 and 's' in v.varName:
            unmatched_blocks.append(int(v.varName[2:-1]))
    print('Min cost:',  m.objVal)
    selected_stations=np.array(selected_stations)
    unmatched_blocks=np.array(unmatched_blocks)
    print(unmatched_blocks)

    x_star=[]
    s_star=[]
    for v in m.getVars():
        if 'x' in v.varName:
            x_star.append(int(v.x))
        if 's' in v.varName:
            s_star.append(int(v.x))

    return selected_stations, unmatched_blocks,x_star,s_star

def display_solution(max_dist):
    block_df=pd.read_csv('Robust/block_df.csv')
    charger_df=pd.read_csv('Robust/chargers_df_05p.csv')

    alpha=np.load('Robust/alpha%s_05p.npy' %max_dist)
    c=1
    lamb=100
    z=charger_df['prediction']
    p=charger_df['prob']

    selected_stations,unmatched_blocks,x_star,s_star=basic_model(alpha,c,lamb,z)
    #selected_df=charger_df.iloc[selected_stations]
    #unmatched_df=block_df.iloc[unmatched_blocks]

    #print(unmatched_df)
    print(str(np.sum(x_star))+" stations opened and "+str(len(s_star)-np.sum(s_star))+" neighborhoods unsatisfied")

    #y=sample_selected(x_star,p)
    s_new=check_feas(p,alpha,x_star)
    print("Sampling terminated with average "+str(np.sum(x_star*p))+" stations opened and average "+str(len(s_new)-np.sum(s_new)) +" neighborhoods unsatisfied")

    #selected_df.to_csv(os.path.join('Robust', 'selected_df%s.csv' %max_dist), index=False)

    print(len(x_star))
    print(len(charger_df))
    print([x_star == 1])
    idxs=[]
    for i in range(len(x_star)):
        if x_star[i]==1:
            idxs.append(i)
    idxs=np.array(idxs)
    print(idxs)
    selected_df=charger_df.loc[idxs]

    block_df['Initially feas']=s_star
    block_df['Finally feas']=s_new

    selected_df.to_csv(os.path.join('Robust/preprocessed', 'selected_df_det_%s_%s.csv' %('05p',300)), index=False)
    #selected_infeas_df.to_csv(os.path.join('Robust/preprocessed', 'selected_infeas_df_%s_%s.csv' %(dist,B0new)), index=False)
    block_df.to_csv(os.path.join('Robust/preprocessed', 'satisfied_block_df_det_%s_%s.csv' %('05p',300)), index=False)

    results_row=pd.DataFrame()
    results_row.loc[0,'init_stat']=np.sum(x_star)
    results_row.loc[0,'sample_stat']=np.sum(x_star*p)
    results_row.loc[0,'init_unsat']=len(s_star)-np.sum(s_star)
    results_row.loc[0,'sample_unsat']=len(s_new)-np.sum(s_new)
    results_row.loc[0,'min_unsat']=np.min(s_new)
    results_row.to_csv('Robust/preprocessed/det_results.csv', index=False)
    print(results_row)

if __name__ == "__main__":
    display_solution(300)