# -*- coding: utf-8 -*-
"""
@author: Valeria Fascianelli, Columbia University, 2023
"""

import numpy as np
from main_functions import get_HMM, get_clustering, get_new_clustered_hmm, get_new_states_sequence

## Example main to fit HMM and run clustering of HMM states of neuronal activity

if __name__=='__main__':    
    
     
    states  = np.linspace(2,50,num=49,dtype=int)   # total number of hidden states
    nModels = 5 # total number of models/state    
    
    #### FIT HMM #####
    ## input:
    #data_area[nMice][nTimeSteps, nNeurons]: neural activity during baseline
    #data_area[nMice][nTimeStepsxnTrials, nNeurons]: neural activity during task
    ## ouptut:
    #AIC_measure[#states,#mice,#nModels]: AIC measure for penalized likelihood
    #results[(iState, iMice, iModel)]: dict containing all the parameters of each model, including the likelihood    
    
    AIC_measure, results   = get_HMM(data_area,  states, nMice, nModels)

    state_activity, id_states_sequence, 
    transition, best_model, state_activity_time = get_analysis_states(AIC_measure,data_area,results,idMice)    
    
    
    ## Analysis of states actitivy, clustering, and defining the new state sequence after states clustering at fixed correlation threshold value     
    general_threshold = 0.1 #correlation value where to cut the clustering dendrogram (minimum=0, maximum=1)

    labels_mice, correlation_output, number_clusters = get_clustering(state_activity, best_model, general_threshold)       
    new_activity_cluster_states, clust_dict          = get_new_clustered_hmm(labels_mice, state_activity_control)
    new_id_states_sequence, new_cluster_states_time  = get_new_states_sequence(nBins,id_states_sequence,labels_mice,new_activity_cluster_states, clust_dict)

    
