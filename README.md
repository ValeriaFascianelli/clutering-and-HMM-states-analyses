# stress_susceptibility

Functions to fix HMM model on neuronal data and to run the clustering of the inferred states using the Pearson correlation as a distance measure between inferred states. Note that the HMM is fit using the model from Linderman Lab https://github.com/lindermanlab.

main_functions.py contains the main functions to fit the HMM, to run the clustering on the inferred hidden states, the assess the participation ratio, and the mahalnobis decoder class
main_HMM_clustering is the main file to run the HMM and clustering analyses using the functions defined in the main_functions.py file 
