import numpy as np
import pandas as pd
import os
import time

from gurobipy import quicksum
from gurobipy import Model
from gurobipy import GRB

from sampling import sample_selected, check_feas

def master_problem(alpha,c,lamb,warm_start=True,add_z=[],add_j=[],start_z=[]):
    #add_z: numpy array of numpy arrays. z arrays to add as constraints
    #add_j: j index associated with the corresponding add_z
    #start z: nominal z if doing a warm start

    I,J=alpha.shape
    #print(I,J)
    if len(start_z)==0:
        z_nom=np.ones(I)
    else:
        z_nom=start_z

    #set up the model
    m = Model()
    vtype = GRB.BINARY

    #define variables
    x = {} #1 if station i is opened, 0 else
    for i in range(I):
        x[i] = m.addVar(vtype=vtype, name="x(%s)" % i)
    s = {} #1 if block j has demand met, 0 else
    for j in range(J):
        s[j] = m.addVar(vtype=vtype, name="s(%s)" % j)


    #add nominal start constraints
    for j in range(J):
        m.addConstr(quicksum(alpha[i,j]*x[i]*z_nom[i] for i in range(I)) >= s[j],
                        name="warm_%s" % j)
        
    #add cutting plane constraints
    num_cps=len(add_j)
    for ix in range(num_cps):
        z_cp=add_z[ix]
        j=add_j[ix]
        m.addConstr(quicksum(alpha[i,j]*x[i]*z_cp[i] for i in range(I)) >= s[j],
                        name="cuttingplane%s" % ix)
        
    #set objective
    m.setObjective(c*quicksum(x[i] for i in range(I))+lamb*quicksum(1-s[j] for j in range(J)), GRB.MINIMIZE)

    #solve
    m.Params.LogToConsole = 0
    m.optimize()
    selected_stations=[]
    unmatched_blocks=[]
    #for v in m.getVars():
    #    if v.x>0 and 'x' in v.varName:
    #        selected_stations.append(int(v.varName[2:-1]))
    #    if v.x<1 and 's' in v.varName:
    #        unmatched_blocks.append(int(v.varName[2:-1]))

    x_star=[]
    s_star=[]
    for v in m.getVars():
        if 'x' in v.varName:
            x_star.append(int(v.x))
        if 's' in v.varName:
            s_star.append(int(v.x))

    print('Min cost:',  m.objVal)
    #selected_stations=np.array(selected_stations)
    #unmatched_blocks=np.array(unmatched_blocks)
    #print(unmatched_blocks)
    x_star=np.array(x_star)
    s_star=np.array(s_star)

    #for constr in m.getConstrs():
        #print(constr)
        #print(constr.slack)

    return x_star,s_star

def subproblem(alpha,x,p,B0,j):
    #set up model
    m = Model()
    vtype = GRB.BINARY
    I,J=alpha.shape

    #define variables
    z = {} #1 if station i is made feasible, 0 else
    for i in range(I):
        z[i] = m.addVar(vtype=vtype, name="z(%s)" % i)

    #add constraints
    m.addConstr(quicksum((1-z[i]) *np.log(1-p[i])for i in range(I)) >= quicksum(x[i]*np.log(B0) for i in range(I)),
                        name="knapsack")
    
    #test constraint with higher N
    #m.addConstr(quicksum((1-z[i]) *np.log(1-p[i])for i in range(I)) >= 3*np.log(B0),
    #                    name="knapsack")

    #set objective
    m.setObjective(quicksum(alpha[i,j]*x[i]*z[i] for i in range(I)), GRB.MINIMIZE)

    #solve
    m.Params.LogToConsole = 0
    m.optimize()
    z_hatj=[]
    for v in m.getVars(): #TODO maybe bad runtime
        z_hatj.append(int(v.x))
    #print(z_hatj)
    
    #print(m.objval)
    return m.objVal,z_hatj

#def warm_start(p,x,B0):
#for a given set of p_i and x values and B0, implement the algorithm to find optimal p_min and warm start
#    rhs=sum()

def cutting_planes(alpha,p,c,lamb,B0):
    I,J=alpha.shape
    print(I,J)

    add_z=[]
    add_j=[]

    terminate=False
    it1=0

    #start_z = np.where(p < 0.01, 0, 1)
    start_z=np.ones(len(p))
    print(sum(start_z))
    print(len(start_z))

    #to track runtime
    MP_times=[]
    SP_times=[]

    warm_start=False

    while terminate==False:
        print("Starting Iteration "+str(it1))

        mp_start=time.time()
        x_star,s_star=master_problem(alpha,c,lamb,warm_start,add_z,add_j,start_z) #TODO one more MP?
        #selected_df=charger_df.iloc[selected_stations]
        #unmatched_df=block_df.iloc[unmatched_blocks]
        #print(x_star)
        #print(s_star)
        mp_end=time.time()
        MP_times.append(mp_end-mp_start)

        sp_start=time.time()
        new_adds=0
        for j in range(J):
            obj,z_hatj=subproblem(alpha,x_star,p,B0,j)
            #print(sum(z_hatj))
            if obj-s_star[j]<0:
                #print(z_hatj)
                add_z.append(z_hatj)
                add_j.append(j)
                #print(add_z,add_j)
                new_adds=new_adds+1
        sp_end=time.time()
        SP_times.append(sp_end-sp_start)
        print("Adding "+ str(new_adds)+ " constraints to Master")
        if new_adds==0:
            terminate=True
        it1=it1+1

    it2=0
    warm_start=False
    terminate=True
    while terminate==False:
        print("Starting Round 2 Iteration "+str(it2))

        x_star,s_star=master_problem(alpha,c,lamb,warm_start,add_z,add_j,start_z)
        #selected_df=charger_df.iloc[selected_stations]
        #unmatched_df=block_df.iloc[unmatched_blocks]
        #print(x_star)
        #print(s_star)

        new_adds=0
        for j in range(J):
            obj,z_hatj=subproblem(alpha,x_star,p,B0,j)
            #print(sum(z_hatj))
            if obj-s_star[j]<0:
                #print(z_hatj)
                add_z.append(z_hatj)
                add_j.append(j)
                #print(add_z,add_j)
                new_adds=new_adds+1
        print("Adding "+ str(new_adds)+ " constraints to Master")
        if new_adds==0:
            terminate=True
        it2=it2+1

    print("Optimal Solution Found")
    #print(x_star)
    #print(s_star)
    #print(add_z)
    #print(add_j)
    MP_times=np.array(MP_times)
    SP_times=np.array(SP_times)
    print(MP_times)
    print(SP_times)

    return x_star,s_star,add_j,it1,it2,MP_times,SP_times

def output_solution(x_star,y,s_star,s_new,dist,B0,block_df,p_cut,MP_times,SP_times):
    print(x_star)
    print(charger_df)
    selected_df=charger_df[(x_star == 1)]
    #selected_infeas_df=charger_df[(x_star==1) & (y == 0)]
    #print(selected_infeas_df)

    block_df['Initially feas']=s_star
    block_df['Finally feas']=s_new
    #print(block_df)

    #init_infeas_df=block_df[(s_star==0)]
    #final_infeas_df=block_df[(s_new==0)]

    #selected_df=charger_df.iloc[selected_stations]
    #unmatched_df=block_df.iloc[unmatched_blocks]

    #print(unmatched_df)

    B0new=str(B0).replace(".", "")

    selected_df.to_csv(os.path.join('Robust/preprocessed', 'selected_df_%s_%s_%s.csv' %(p_cut,dist,B0new)), index=False)
    #selected_infeas_df.to_csv(os.path.join('Robust/preprocessed', 'selected_infeas_df_%s_%s.csv' %(dist,B0new)), index=False)
    block_df.to_csv(os.path.join('Robust/preprocessed', 'satisfied_block_df_%s_%s_%s.csv' %(p_cut,dist,B0new)), index=False)

    np.save('Robust/preprocessed/MP_times_%s_%s_%s.npy' %(p_cut,dist,B0new), MP_times)
    np.save('Robust/preprocessed/SP_times_%s_%s_%s.npy' %(p_cut,dist,B0new), SP_times)

def make_results_df(B0,iter,constr,init_stat,sample_stat,init_unsat,sample_unsat,min_unsat,runtime):
    results_df=pd.read_csv('Robust/preprocessed/results_df.csv')
    if B0 not in results_df['B0'].unique():
        results_row=pd.DataFrame()
        results_row.loc[0,'B0']=B0
        results_row.loc[0,'iter']=iter
        results_row.loc[0,'constr']=constr
        results_row.loc[0,'init_stat']=init_stat
        results_row.loc[0,'sample_stat']=sample_stat
        results_row.loc[0,'init_unsat']=init_unsat
        results_row.loc[0,'sample_unsat']=sample_unsat
        results_row.loc[0,'min_unsat']=min_unsat
        results_row.loc[0,'runtime']=runtime
        results_df=pd.concat([results_df,results_row])
        results_df.to_csv('Robust/preprocessed/results_df.csv', index=False)

if __name__ == "__main__":
    #dummy test
    #alpha=np.array([[0,1,0,1,0,1],[1,0,0,0,1,0],[0,1,1,0,0,0],[0,0,1,0,0,0],[0,1,1,0,1,0]])
    #p=[.2,.2,.3,.1,.1]
    start_time=time.time()
    c=1
    lamb=100
    B0=1
    dist=300
    p_cut='05p'

    #real data test
    block_df=pd.read_csv('Robust/block_df.csv')
    charger_df=pd.read_csv('Robust/chargers_df_05p.csv')
    alpha=np.load('Robust/alpha%s.npy' %dist)
    p=charger_df['prob'].to_numpy()
    #alpha=alpha[p > 0.1]
    print(alpha.shape)
    #p=p[p>0.1]
    print(len(p))
    #print(p)

    start_cp_time=time.time()
    x_star,s_star,add_j,num_iters,num_iters2,MP_times,SP_times=cutting_planes(alpha,p,c,lamb,B0)
    end_cp_time=time.time()

    num_constrs=len(add_j)
    print("Cutting planes terminated with "+str(num_iters)+" iterations and "+str(num_constrs) +" added constraints")
    print(str(np.sum(x_star))+" stations opened and "+str(len(s_star)-np.sum(s_star))+" neighborhoods unsatisfied")

    #y=sample_selected(x_star,p)
    s_new=check_feas(p,alpha,x_star)
    min_unsat=np.min(s_new)
    #print(s_new)
    print("Sampling terminated with average "+str(np.round(np.sum(x_star*p),3))+" stations opened and average "+str(np.round(len(s_new)-np.sum(s_new),3)) +" neighborhoods unsatisfied")

    output_solution(x_star,p,s_star,s_new,dist,B0,block_df,p_cut,MP_times,SP_times)
    end_time=time.time()
    total_time=end_time-start_time

    make_results_df(B0,num_iters,num_constrs,np.sum(x_star),np.round(np.sum(x_star*p),3),len(s_star)-np.sum(s_star),np.round(len(s_new)-np.sum(s_new),3),min_unsat,total_time)