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
data_npy_phoneme_1 = 'data/GMM_params_phoneme_01_k_03.npy'
data_npy_phoneme_2 = 'data/GMM_params_phoneme_02_k_03.npy'


data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)
# Loading data from .npy file
data_1 = np.load(data_npy_phoneme_1, allow_pickle=True)
data_1 = np.ndarray.tolist(data_1)

data_2 = np.load(data_npy_phoneme_2, allow_pickle=True)
data_2 = np.ndarray.tolist(data_2)

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

########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here
X_full[:,0]=f1
X_full[:,1]=f2
# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

# X_phonemes_1_2 = ...
X_phonemes_1_2 = np.array([])

for i in range(len(f1)):
    if phoneme_id[i] == 1:
        if np.size(X_phonemes_1_2) == 0:
            X_phonemes_1_2 = [X_full[i][0], X_full[i][1]]
        else:
            row = [X_full[i][0], X_full[i][1]]
            X_phonemes_1_2 = np.vstack((X_phonemes_1_2, row))
for i in range(len(f1)):
    if phoneme_id[i] == 2:
        if np.size(X_phonemes_1_2) == 0:
            X_phoneme_1_2 = [X_full[i][0], X_full[i][1]]
        else:
            row = [X_full[i][0], X_full[i][1]]
            X_phonemes_1_2 = np.vstack((X_phonemes_1_2, row))
########################################/
print(X_phonemes_1_2)
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

########################################/
mu1= data_1['mu']
s1=data_1['s']
p1=data_1['p']

mu2= data_2['mu']
s2=data_2['s']
p2=data_2['p']

prediction1=get_predictions(mu1,s1,p1,X_phonemes_1_2)

prediction2=get_predictions(mu2,s1,p2,X_phonemes_1_2)
print(len(prediction1)+len(prediction2))
data1=int(len(X_phonemes_1_2)/2)
GMM1=0
GMM2=0
for i in range(data1):
    if(np.sum(prediction1[i])>np.sum(prediction2[i])):
        GMM1+=1
for j in range(data1,len(X_phonemes_1_2)):
    if(np.sum(prediction1[j])<=np.sum(prediction2[j])):
        GMM2+=1
print(GMM1+GMM2)
accuracy=((GMM1+GMM2)/(len(X_phonemes_1_2))*100)

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()