# -*- coding: utf-8 -*-
"""
@author: Valeria Fascianelli, Columbia University, 2023
"""
import numpy as np
import pandas as pd
import scipy
import random
import PCA

## HMM functions: please, download it from https://github.com/lindermanlab
import ssm  # for hmm

# for clustering
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import squareform


####################################################################################
## fit HMM model
def get_HMM(data_area, states, nMice, nModels):
    
    #Fit HMM with Poisson statistics, and E-M algorithm (see for details  https://github.com/lindermanlab)
    
    #data_area[nMice][nTimeSteps, nNeurons]: neural activity during baseline
    #data_area[nMice][nTimeStepsxnTrials, nNeurons]: neural activity during task
    #states: list of # states
    #nMice: total number of mice
    #nModels: total number of models to fit
    
    
    results = {}
    aic = np.nan*np.ones((len(states), nMice, nModels))    
    LL_control = np.nan*np.ones((len(states), nMice, nModels))
    results = {}
    break_var=False
    for iState,jState in enumerate(states):
        print('[state] ', jState)
        for iMiceTrain in range(nMice): 
            if break_var:                
                break_var=False
                continue
            print('[mice]  ', iMiceTrain)
            m=jState
            k = 1 # because Poisson process only has 1 parameter
            param = m*m + k*m -1  
                           
        
            if break_var:
                break
            #print('cross ', iCross)
          
            data_hmm_control = data_area[iMiceTrain]#[:,idNeurons]                 
                       
            data = data_hmm_control
            D = np.size(data,axis=1)
            for iModel in range(nModels):                                              
                  
                model_hmm = ssm.HMM(jState, D, observations="poisson")
               
                try:
                    train_ll  = model_hmm.fit(data, method='em')                  
                except:
                    print('Not convergence anymore')
                    break_var = True
                    break
                
                LL = model_hmm.log_likelihood(data)   
                LL_control[iState, iMiceTrain, iModel] = LL
                aic[iState, iMiceTrain, iModel] = 2*param-2*LL                    
                results[(iState, iMiceTrain, iModel)] = (model_hmm, train_ll, LL)

    return aic, results                


def get_analysis_states(aic,data_area,results_hmm,idMice):
   
    ### Get best HMM models/state, average inferred activity/state, sequence of most probable states along time 
    
    ## INPUT:
    #data_area[nMice][nTimeSteps, nNeurons]: neural activity during baseline
    #data_area[nMice][nTimeStepsxnTrials, nNeurons]: neural activity during task 
    #aic[#states,#mice,#nModels]: AIC measure for penalized likelihood
    #results_hmm[(iState, iMice, iModel)]: dict containing all the parameters of each model, including the likelihood            
    #idMice: indices list of mice to analyze
    
    ## OUTPUT:
    #state_activity[nMice][#states,#neurons]: inferred neuronal activity in each hidden states
    #id_states_sequence: states id in each consecutive time bin
    #transition[nMice][#states,#states]: transition matrix between each hidden state
    #best_model[nMice]: array of HMM object of ssm.hmm module indicating the best HMM model (according to AIC)
    #state_activity_time[nMice][#nBins,#neurons]: inferred states activity in each consecutive time bin

    aic = np.squeeze(aic,)
  
    state_activity         = []   
    best_model             = []       
    id_states_sequence       = []
    transition             = []
    state_activity_time = []
       
    for iMice,jMice in enumerate(idMice):
             
        index_mice      = []
        best_model_index = np.unravel_index(np.nanargmin(aic[:,jMice,:]), aic[:,jMice,:].shape)
        iModel = best_model_index[1]
        iState = best_model_index[0]
        best_model.append(results_hmm[iState, jMice,0,iModel][0])
        most_states   = best_model[iMice].most_likely_states(data_area[jMice])
        id_states_sequence.append(best_model[iMice].most_likely_states(data_area[jMice]))
        transition_matrix = best_model[iMice].transitions.transition_matrix  
        transition.append(transition_matrix)
       
        smoothed_data = best_model[iMice].smooth(data_area[jMice])
        state_activity_time.append(smoothed_data)
        
        ## reorganize smoothed data for states
        a = pd.Series(range(len(most_states))).groupby(most_states, sort=False).apply(list).tolist()
        smoothed_mice_sorted = np.nan*np.ones(np.shape(smoothed_data))
        xx=0
        state_activity_mice_app = np.empty((0,np.size(data_area[jMice],axis=1)), float)
        for jj,kk in enumerate(a):
            smoothed_mice_sorted[xx:xx+len(kk),:] = smoothed_data[kk,:]      
            state_activity_mice_app = np.vstack((state_activity_mice_app, np.mean(smoothed_data[kk,:],axis=0)))
            xx+=len(kk)
            index_mice.append(xx)
        state_activity.append(state_activity_mice_app)                           
        
    return state_activity, id_states_sequence, transition, best_model, state_activity_time





def get_clustering(state_activity, best_model, threshold):
    
    ## get analysis of inferrred state activity and clustering of the states
    
    ##INPUT:
    #state_activity[nMice][#states,#neurons]: inferred neuronal activity in each hidden states
    #best_model[nMice]: array of HMM object of ssm.hmm module indicating the best HMM model (according to AIC)
    #threshold: correlation value where to cut the clustering dendrogram (minimum=0, maximum=1)

    ##OUTPUT:
    #labels_mice: indices identifyinc the new clusters of old states (direct output from shc.fcluster(), visit https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html)    
    #correlation_output:states correlation [#states,#states]
    #number_clusters: #number of total unique clusters at given correlation threshold
        
        
    number_clusters = []   
    labels_mice=[]
    correlation_output = []
    
    ## loop over mice
    for imice, state_mice in enumerate(state_activity): 
        print(imice)
        
        ## compute dissimilarity
        pandas_control = pd.DataFrame(state_mice.transpose())    
        correlations   = pandas_control.corr()
        correlation_output.append(correlations)
        dissimilarity = 1 - abs(correlations)  
        
        ## start clustering
        Z = shc.linkage(squareform(dissimilarity), 'complete')   
        
        labels = shc.fcluster(Z, threshold, criterion='distance')
        number_clusters.append(len(np.unique(labels)))
        labels_mice.append(labels)
              
    return labels_mice, correlation_output, number_clusters


def get_new_clustered_hmm(labels, state_activity):    
    ### compute the actiticy of the new clustered states    
    
    ##INPUT: 
    #labels_mice: indices identifyinc the new clusters of old states (direct output from shc.fcluster(), visit https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html)    
    #state_activity[nMice][#states,#neurons]: inferred neuronal activity in each hidden states

    ##OUTPUT:
    #new_activity_cluster_states: new average actitivy of the new clusters   
    #clust_dict: dictionary matching the new cluster ID with all the former states ID
            
    new_activity_cluster_states = []    
    clust_dict = []
   
    for i in range(len(labels)):
        #print(i)
        clusters = [] 
        val = np.unique(labels[i])
        new_cluster_states_app = np.nan*np.ones((len(val), np.size(state_activity[i],axis=1)))
        clus_dict_app = dict()
        for iv,v in enumerate(val):
            #print(v)         
            index = list(np.where(labels[i]==v)[0])
            clusters.append(index)
            
            new_cluster_states_app[iv,:]=np.mean(state_activity[i][index,:],axis=0)
            clus_dict_app[v] = iv
        new_activity_cluster_states.append(new_cluster_states_app)  
        clust_dict.append(clus_dict_app)
            
    return new_activity_cluster_states, clust_dict

def get_new_states_sequence(nBins,id_states_sequence,labels_mice,new_activity_cluster_states, clust_dict):
    #### define the new states ID sequence along time after clustering           
    
    ##INPUT:
    #nBins: #total time bins
    #id_states_sequence: states id in each consecutive time bin
    #labels_mice: indices identifyinc the new clusters of old states (direct output from shc.fcluster(), visit https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html)    
    #new_activity_cluster_states: new average actitivy of the new clusters   
    #clust_dict: dictionary matching the new cluster ID with all the former states ID

    ##OUTPUT:
    #new_id_states_sequence: cluster id in each consecutive time bin (after clustering)
    #new_cluster_states_time: new average actitivy of the new clusters in each consecutive time bins
    
    
    new_id_states_sequence = []
    new_cluster_states_time = []
    new_duration_states = []
    for iMice in range(len(id_states_sequence)):
        print(iMice)
        new_id_states_sequence_app = np.nan*np.ones((len(id_states_sequence[iMice])))
        new_duration_states_app = []
        new_cluster_states_time_app = np.nan*np.ones((len(id_states_sequence[iMice]),np.size(new_activity_cluster_states[iMice],axis=1)))
        for iState,jState in enumerate(id_states_sequence[iMice]):           
            new_id_states_sequence_app[iState] = labels_mice[iMice][jState]
            clusID_pos = clust_dict[iMice][labels_mice[iMice][jState]]
            new_cluster_states_time_app[iState,:] = new_activity_cluster_states[iMice][clusID_pos,:]
        new_cluster_states_time.append(new_cluster_states_time_app)   
        new_id_states_sequence.append(new_id_states_sequence_app)
        
        n=1
        for i,j in enumerate(new_id_states_sequence[iMice][:-1]):                
            if new_id_states_sequence[iMice][i] == new_id_states_sequence[iMice][i+1]:                
                n+=1                
            else:
                new_duration_states_app.append(n)                      
                n=1   
        if n == nBins:
            new_duration_states.append([n])
        else:   
            new_duration_states.append(new_duration_states_app)        
    return new_id_states_sequence, new_cluster_states_time


############## for PCA dimensionality analysis
def get_spectrum_pca(data_pca, idMice, bin_size, nSubSamples, nRndNeurons):
    spectrum      = np.nan*np.ones((len(idMice), nSubSamples, nRndNeurons))
    spectrum_cum  = np.nan*np.ones((len(idMice), nRndNeurons))

    for iMouse, jMouse in enumerate(idMice):
        data = data_pca[jMouse]/bin_size
        data_zscore = scipy.stats.zscore(data,axis=0)
        nNeurons = np.size(data,axis=1)
        for isubsample in range(nSubSamples):
            rnd_neurons = random.sample(range(nNeurons), nRndNeurons)            
            data_app = data_zscore[:,rnd_neurons]            
            pca = PCA()
            pca.fit(data_app)            
            spectrum[iMouse,isubsample,:] = pca.explained_variance_ratio_ 
        spectrum_cum[iMouse, :] = np.cumsum(np.nanmean(spectrum[iMouse,:,:],axis=0))                        
            
    return spectrum, spectrum_cum    

def get_participation_ratio(eigenvalues):
    eigenvalues = np.array(eigenvalues)
    a = np.sum(eigenvalues)**2
    b = np.sum(eigenvalues**2)
    return a/b

# mahalnobis decoder
def mahalanobis(x=None, data=None, cov=None):
   
    x_minus_mu = scipy.spatial.distance.euclidean(x,np.mean(data,axis=0))  
    vect_dir = (x-np.mean(data,axis=0))/x_minus_mu #subtract the overall mean and normalize vector length to 1
    std_data = np.std(data,axis=0)    
    sigma    = np.abs(np.dot(vect_dir,std_data))
        
    return x_minus_mu/sigma #mahal.diagonal()


class MahalanobisBinaryClassifier():
    
    def __init__(self, xtrain, ytrain):
        self.xtrain1 = xtrain[ytrain == 0, :]
        self.xtrain2 = xtrain[ytrain == 1, :]
        
    def score(self, xtest,ytest):
                                
        ytrain = [0,1]
        dist1 = mahalanobis(xtest, self.xtrain1)
        dist2 = mahalanobis(xtest, self.xtrain2)
        
        min_dist = np.argmin((dist1,dist2))
        
        if ytrain[min_dist] == ytest :
            return 1
        else:
            return 0
        
   
