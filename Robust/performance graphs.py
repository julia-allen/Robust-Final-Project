import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#B0=[.999,.99,.98,.97,.96,.95,.94]
#iters=[1,5,4,8,9,14,39]
#constrs=[0,60,75,153,181,348,576]
#init_stats=[17,17,18,18,18,20,22]
#final_stats=[9,10,15,16,18,19,21] s=marker_size
#init_unsatis=[0,0,0,0,0,0,0]
#final_unsatis=[51,41,23,14,0,0,0]

results_df=pd.read_csv('Robust/preprocessed/results_df.csv')
B0=results_df['B0']
iters=results_df['iter']
constrs=results_df['constr']
init_stats=results_df['init_stat']
final_stats=results_df['sample_stat']
init_unsatis=results_df['init_unsat']
final_unsatis=results_df['sample_unsat']
min_unsatis=results_df['min_unsat']
runtimes=results_df['runtime']

det_results=pd.read_csv('Robust/preprocessed/det_results.csv')
det_init_stat=det_results['init_stat']
det_final_stat=det_results['sample_stat']
det_init_unsatis=det_results['init_unsat']
det_final_unsatis=det_results['sample_unsat']
det_min_unsatis=det_results['min_unsat']
print(det_init_stat.loc[0])

MP_times_099=np.load('Robust/preprocessed/MP_times_05p_300_099.npy')
SP_times_099=np.load('Robust/preprocessed/SP_times_05p_300_099.npy')
MP_times_093=np.load('Robust/preprocessed/MP_times_05p_300_093.npy')
SP_times_093=np.load('Robust/preprocessed/SP_times_05p_300_093.npy')
MP_times_092=np.load('Robust/preprocessed/MP_times_05p_300_092.npy')
SP_times_092=np.load('Robust/preprocessed/SP_times_05p_300_092.npy')

plt.plot(MP_times_093,label='Master problem')
plt.plot(SP_times_093,label='Subproblem')
plt.xlabel('Iteration')
plt.ylabel('Runtime')
plt.title("Runtime vs. Iteration for Master and Sub-Problems, B0=0.93")
plt.show()

plt.plot(MP_times_099,label='Master problem')
plt.plot(SP_times_099,label='Subproblem')
plt.xlabel('Iteration')
plt.ylabel('Runtime')
plt.title("Runtime vs. Iteration for Master and Sub-Problems, B0=0.99")
plt.show()

plt.plot(MP_times_092,label='Master problem')
plt.plot(SP_times_092,label='Subproblem')
plt.xlabel('Iteration')
plt.ylabel('Runtime')
plt.title("Runtime vs. Iteration for Master and Sub-Problems, B0=0.92")
plt.show()





df_09999=pd.read_csv('Robust/preprocessed/selected_df_05p_300_09999.csv')
ps_09999=df_09999['prob']
df_093=pd.read_csv('Robust/preprocessed/selected_df_05p_300_093.csv')
ps_03=df_093['prob']
plt.hist(ps_09999,histtype='step',linewidth=2,label='B0=1')
plt.hist(ps_03,histtype='step',linewidth=2,label='B0=0.93')
plt.legend()
plt.title("Feasibility Probability of Selected Stations")
plt.xlabel('Feasibility probability')
plt.xlabel('Frequency')
plt.show()

det_df=pd.read_csv('Robust/preprocessed/selected_df_det_05p_300.csv')
ps_det=det_df['prob']
plt.hist(ps_det,histtype='step',linewidth=2,label='Deterministic')
plt.hist(ps_03,histtype='step',linewidth=2,label='Robust, B0=0.93')
plt.legend()
plt.title("Feasibility Probability of Selected Stations")
plt.xlabel('Feasibility probability')
plt.xlabel('Frequency')
plt.show()

plt.plot(B0,runtimes)
plt.xlabel("B_0")
plt.ylabel("Runtime")
plt.title("Runtime vs. B_0")
plt.show()

plt.plot(B0,iters)
plt.xlabel("B_0")
plt.ylabel("Number of iterations")
plt.title("Number of iterations vs. B_0")
plt.show()

plt.plot(B0,constrs)
plt.xlabel("B_0")
plt.ylabel("Number of added constraints")
plt.title("Number of added constraints vs. B_0")
plt.show()

plt.plot(B0,init_stats,label="Pre-sampling")
plt.plot(B0,final_stats,label="Post-sampling")
plt.hlines(det_init_stat.iloc[0],0.93,1, colors='blue',linestyles='dotted', label='Deterministic pre-sampling')
plt.hlines(det_final_stat.iloc[0],0.93,1, colors='orange',linestyles='dotted', label='Deterministic post-sampling')
plt.xlabel("B_0")
plt.ylabel("Number of stations opened")
plt.title("Pre- and post-sampling number of stations opened vs. B_0")
plt.legend()
plt.show()

plt.plot(B0,init_unsatis,label="Pre-sampling")
plt.plot(B0,final_unsatis,label="Post-sampling")
plt.hlines(det_init_unsatis.iloc[0],0.93,1, colors='blue',linestyles='dotted', label='Deterministic pre-sampling')
plt.hlines(det_final_unsatis.iloc[0],0.93,1, colors='orange',linestyles='dotted', label='Deterministic post-sampling')
plt.xlabel("B_0")
plt.ylabel("Number of neighborhoods unsatisfied")
plt.title("Pre- and post-sampling number of neighborhoods unsatisfied vs. B_0")
plt.legend()
plt.show()

plt.plot(B0,min_unsatis,label="Robust")
plt.hlines(det_min_unsatis.iloc[0],0.93,1, colors='blue',linestyles='dotted', label='Deterministic')
plt.xlabel("B_0")
plt.ylabel("Minimum probability of satisfaction")
plt.title("Minimum probability of satisfaction vs. B_0")
plt.legend()
plt.show()