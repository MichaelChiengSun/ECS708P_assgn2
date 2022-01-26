import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:, 0] = f1
X_full[:, 1] = f2
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full
X_phoneme_1 = np.zeros((np.sum(phoneme_id==1), 2))
X_phoneme_2 = np.zeros((np.sum(phoneme_id==2), 2))
# X_phoneme = ...
X_phoneme_1[:,0] = f1[phoneme_id==1]
X_phoneme_1[:,1] = f2[phoneme_id==1]
X_phoneme_2[:,0] = f1[phoneme_id==2]
X_phoneme_2[:,1] = f2[phoneme_id==2]
X_phonemes_1_2 = np.vstack((X_phoneme_1, X_phoneme_2))
########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)

#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"
GMMp1_npy_file = 'data/GMM_params_phoneme_01_k_03.npy'
GMMp2_npy_file = 'data/GMM_params_phoneme_02_k_03.npy'
GMMp3_npy_file = 'data/GMM_params_phoneme_01_k_06.npy'
GMMp4_npy_file = 'data/GMM_params_phoneme_02_k_06.npy'
GMMp1 = np.load(GMMp1_npy_file, allow_pickle=True)
GMMp1 = np.ndarray.tolist(GMMp1)
GMMp2 = np.load(GMMp2_npy_file, allow_pickle=True)
GMMp2 = np.ndarray.tolist(GMMp2)
GMMp3 = np.load(GMMp3_npy_file, allow_pickle=True)
GMMp3 = np.ndarray.tolist(GMMp3)
GMMp4 = np.load(GMMp4_npy_file, allow_pickle=True)
GMMp4 = np.ndarray.tolist(GMMp4)
mu1 = GMMp1['mu']
mu2 = GMMp2['mu']
mu3 = GMMp3['mu']
mu4 = GMMp4['mu']
s1 = GMMp1['s']
s2 = GMMp2['s']
s3 = GMMp3['s']
s4 = GMMp4['s']
p1 = GMMp1['p']
p2 = GMMp2['p']
p3 = GMMp3['p']
p4 = GMMp4['p']
prediction1 = get_predictions(mu1,s1,p1,X_phonemes_1_2)
prediction2 = get_predictions(mu2,s2,p2,X_phonemes_1_2)
prediction3 = get_predictions(mu3,s3,p3,X_phonemes_1_2)
prediction4 = get_predictions(mu4,s4,p4,X_phonemes_1_2)
GMM_phoneme1 = 0
GMM_phoneme2 = 0

for i in range(int(len(X_phonemes_1_2)/2)):
    if np.sum(prediction1[i]) > np.sum(prediction2[i]):
    #if np.sum(prediction3[i]) > np.sum(prediction4[i]):
        GMM_phoneme1 += 1

for i in range(int(len(X_phonemes_1_2)/2), len(X_phonemes_1_2)):
    if np.sum(prediction2[i]) > np.sum(prediction1[i]):
    #if np.sum(prediction4[i]) > np.sum(prediction3[i]):
        GMM_phoneme2 += 1

accuracy = ((GMM_phoneme1+GMM_phoneme2)/len(X_phonemes_1_2))*100
########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()