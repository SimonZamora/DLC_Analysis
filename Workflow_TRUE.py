import pandas as pd

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

import csv

from os import listdir

mypath = r"D:/Simon/Analyses/449_FirstTry/Data"
data_path = r"D:/Simon/Analyses/449_FirstTry/FullDataSet"

files_list = [f.split('_')[0]+'_'+f.split('_')[1] for f in listdir(mypath)]
data_list = [f for f in listdir(data_path) if f.endswith('.csv')]

dico = {}

for elem in files_list:
    dico[elem]= [f for f in data_list if elem in f]
##
def return_all_trials_from_day(day,full_data):
    return [trial for trial in full_data if day in trial]
##

def trial_struct(trial_name,bodyparts,path = "D:/Simon/Analyses/449_FirstTry/FullDataSet/"):
    trial = {}
    print("Extracting data from this trial:",trial_name)
    with open(path+trial_name, 'r') as file:
        csvreader = csv.reader(file)
        line_count = 0
        for part in bodyparts:
            trial[part] = {'x': [], 'y': [], 'likelihood': []}
        trial["Name"]=trial_name.split('_')[1]+'_'+trial_name.split('_')[2]+'_'+trial_name.split('_')[3]
        for row in csvreader:

            if line_count > 2:
                iterat = 1
                for i in range(len(bodyparts)):
                    trial[bodyparts[i]]['x'].append(float(row[iterat]))
                    trial[bodyparts[i]]['y'].append(float(row[iterat + 1]))
                    trial[bodyparts[i]]['likelihood'].append(float(row[iterat + 2]))
                    iterat += 3

            line_count += 1
    print("Done!")
    return trial

##
def remove_unlikely(trial,treshold = 0.9,bodyparts=['LeftPaw', 'RightPaw', 'Nose', 'Tongue', 'Droplet']):
    for parts in bodyparts:
        trial[parts]['x']= [np.nan if j < 0.9 else i for i,j in zip(trial[parts]['x'],trial[parts]['likelihood'])]
        trial[parts]['y']= [np.nan if j < 0.9 else i for i,j in zip(trial[parts]['y'],trial[parts]['likelihood'])]

##

def plot_position_in_time(trial,figure,position_in_grid,axis):
    ax = figure.add_subplot(position_in_grid)
    lin = np.linspace(1, len(trial["RightPaw"][axis]), len(trial["RightPaw"][axis]))

    ax.plot(lin, trial["RightPaw"][axis], label="RightPaw")
    ax.plot(lin, trial["LeftPaw"][axis], label="LeftPaw")
    ax.legend()
    ax.set_title(axis+" position of paws")
    ax.set_xlabel("time")
    ax.set_ylabel(axis)

def plot_current_trial(trial,cam,bodyparts=['Nose', 'Tongue', 'Droplet','LeftPaw', 'RightPaw']):
    markers = ['<','<','v','o','o']
    colors = ['Greens','Wistia','Greys','Reds','Blues']
    colors_ = ['green', 'orange', 'gray','red', 'blue']

    fig = plt.figure(tight_layout=True)
    grid = plt.GridSpec(ncols=2, nrows=2, width_ratios=[3, 3])

    ax1 = fig.add_subplot(grid[0:3,0])

    for k in range(len(bodyparts)):
        x = trial[bodyparts[k]]['x'][0]
        y = trial[bodyparts[k]]['y'][0]
        ax1.scatter(x, y, marker=markers[k], label=bodyparts[k], c=colors_[k])

    for i in range(len(bodyparts)):
        x = trial[bodyparts[i]]['x']
        y = trial[bodyparts[i]]['y']
        lin = np.linspace(1, len(x), len(x))
        ax1.scatter(x,y,marker = markers[i],c = lin,cmap=colors[i])

    ax1.set_ylim(380, 50)
    ax1.set_xlim(50,350)
    ax1.legend()
    ax1.set_title("Trajectory of each object")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Plot traces of paws x
    plot_position_in_time(trial,fig,grid[0,1],'x')

    # Plot traces of paws y
    plot_position_in_time(trial, fig, grid[1,1], 'y')

    plt.show()

##
def slice_trajectory(trial,bodypart,interval):
    traj_sliced = {'x':trial[bodypart]['x'][interval[0]:interval[1]],
                   'y':trial[bodypart]['y'][interval[0]:interval[1]],
                   'likelihood':trial[bodypart]['likelihood'][interval[0]:interval[1]]}
    return traj_sliced
def find_reaches(trial,reaching_paw,interval=[-150,50]):
    x = trial[reaching_paw]['x']
    maxi = min((v, i) for i, v in enumerate(x))[1]
    trial_cutted={}
    print(maxi)
    for entry in trial:
        if len(trial[entry])>3:
            trial_cutted[entry]=trial[entry]
        else:
            inter = [maxi+interval[0],maxi+interval[1]]
            trial_cutted[entry]=slice_trajectory(trial,entry,inter)
    return trial_cutted