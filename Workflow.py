import pandas as pd

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')
import csv
from os import listdir

from scipy.signal import savgol_filter
from scipy import interpolate

##
def return_all_trials_name_from_day(day,full_data):
    return [trial for trial in full_data if day in trial]

##
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


##
import difflib
def return_triplet_name(full_data):


    full_data.sort()
    chunked = chunks(full_data,3)
    full_mat = []
    good = True

    for triplet in chunked:
        if (triplet[0][:-62] == triplet[1][:-62]==triplet[2][:-62] and triplet[0][-58:]==triplet[1][-58:]==triplet[2][-58:]):
            full_mat.append(triplet)
        else:
            print("Wrong triplet",triplet[0],triplet[1],triplet[2])
            good = False
    if good:
        print("All triplet well formed", len(full_mat),"Triplet found")
    return full_mat

##

def read_csv(csv_path,bodyparts):
    trial = {}
    print("Extracting data from this trial:", csv_path)
    with open(csv_path, 'r') as file:
        csvreader = csv.reader(file)
        line_count = 0
        for part in bodyparts:
            trial[part] = {'x': [], 'y': [], 'likelihood': []}
        slit_name = csv_path.split('_')
        trial["Name"] = slit_name[1][-5:] + '_' +slit_name[2] + '_' + slit_name[4] + '_' + slit_name[5]
        trial["Cam"]=slit_name[6]
        trial["T"]=slit_name[7] + '_'+slit_name[8]+'_'+slit_name[9][:-4]
        for row in csvreader:

            if line_count > 2:
                iterat = 1
                for i in range(len(bodyparts)):
                    trial[bodyparts[i]]['x'].append(float(row[iterat]))
                    trial[bodyparts[i]]['y'].append(float(row[iterat + 1]))
                    trial[bodyparts[i]]['likelihood'].append(float(row[iterat + 2]))
                    iterat += 3

            line_count += 1
    print("Done extracting 1 trial!")
    return trial

## Work on removing likelihood

def remove_unlikely(trial,treshold = 0.9,bodyparts=['LeftPaw', 'RightPaw', 'Nose', 'Tongue', 'Droplet']):
    for parts in bodyparts:
        trial[parts]['x']= [np.nan if j < 0.9 else i for i,j in zip(trial[parts]['x'],trial[parts]['likelihood'])]
        trial[parts]['y']= [np.nan if j < 0.9 else i for i,j in zip(trial[parts]['y'],trial[parts]['likelihood'])]
##
def trial_struct(list_of_triplet,bodyparts,path):
    final_list_of_trials = []
    for trip in list_of_triplet:
        #print("Reading this triplet",trip)
        trials = {}
        for trial in trip:
            extract = read_csv(path+trial,bodyparts)
            remove_unlikely(extract)
            trials[extract['Cam']]=extract
        final_list_of_trials.append(trials)
        #print('')
    print("Done extracting all trials!")

    return final_list_of_trials

##
def construct_body_struct(list_of_trials,bodyparts = ['LeftPaw', 'RightPaw', 'Nose', 'Tongue', 'Droplet'],main_cam='camA'):
    body_struct = []
    for trial in list_of_trials:
        body = {}
        body["Name"]=trial["camA"]["Name"]
        body["T"]=trial["camA"]["T"]
        for part in bodyparts:
            body[part] = {'x': trial["camA"][part]["x"], 'y': trial["camB"][part]["x"], 'z': trial["camB"][part]["y"]}

        body['RightPaw']['x'] = trial["camC"]["RightPaw"]["x"]
        body_struct.append(body)
    return body_struct

##
def create_dataset_from_list(list_of_trial,body_part,data_path, day = ''):
    if day != '':

        day_list = return_all_trials_name_from_day(day,list_of_trial)
        triplets = return_triplet_name(day_list)

    else:
        triplets = return_triplet_name(list_of_trial)
    trials = trial_struct(triplets, body_part, data_path + '/')
    bodies = construct_body_struct(trials)

    return bodies
##

mypath = r"D:/Simon/Analyses/449_FirstTry/Data"
data_path = r"D:/Simon/Analyses/449_FirstTry/FullDataSet"

#data_path = r"D:/Simon/Analyses/449_FirstTry/FirstTryTest"

files_list = [f.split('_')[0]+'_'+f.split('_')[1] for f in listdir(mypath)]
data_list = [f for f in listdir(data_path) if f.endswith('.csv')]
body_parts = ['LeftPaw', 'RightPaw', 'Nose', 'Tongue', 'Droplet']

#data = create_dataset_from_list(data_list,body_parts,data_path,day="2023-02-08_16-53-13")
data = create_dataset_from_list(data_list,body_parts,data_path)

##

def plot_position_in_time(trial,figure,position_in_grid,axis):
    ax = figure.add_subplot(position_in_grid)
    lin = np.linspace(1, len(trial["RightPaw"][axis]), len(trial["RightPaw"][axis]))

    ax.plot(lin, trial["RightPaw"][axis], '-o',markersize = 3,label="RightPaw")
    ax.plot(lin, trial["LeftPaw"][axis], '-o',markersize = 3,label="LeftPaw")
    ax.legend()
    ax.set_title(axis+" position of paws")
    ax.set_xlabel("time")
    ax.set_ylabel(axis)

def plot_current_trial(trial,bodyparts=['Nose', 'Tongue', 'Droplet','LeftPaw', 'RightPaw']):
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

    ax1.set_ylim(500, 250)
    ax1.set_xlim(50,350)
    ax1.legend()
    ax1.set_title("Trajectory of each object")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Plot traces of paws x
    plot_position_in_time(trial,fig,grid[0,1],'x')

    # Plot traces of paws y
    plot_position_in_time(trial, fig, grid[1,1], 'y')
    fig.suptitle(trial["Name"]+trial["T"])
    plt.show()

plot_current_trial(data[1])
##

def slice_trajectory(trial,bodypart,interval):
    traj_sliced = {'x':trial[bodypart]['x'][interval[0]:interval[1]],
                   'y':trial[bodypart]['y'][interval[0]:interval[1]]}
    return traj_sliced
def find_reaches(trial,reaching_paw,interval=[-150,50]):
    x = trial[reaching_paw]['x']
    maxi = np.nanargmin(x)
    trial_cutted={}
    for entry in trial:
        if len(trial[entry])>3:
            trial_cutted[entry]=trial[entry]
        else:
            inter = [maxi+interval[0],maxi+interval[1]]
            trial_cutted[entry]=slice_trajectory(trial,entry,inter)
    trial_cutted["DetectedReach"]=maxi+interval[0]
    return trial_cutted

##
cut = find_reaches(data[1],"LeftPaw")
plot_current_trial(cut)
plot_current_trial(data[1])
##
from scipy.signal import find_peaks
tr = data[0]["RightPaw"]['x']
x = np.abs(np.diff(tr))
t2 = [y - x for x,y in zip(tr,tr[1:])]
plt.plot(t)
plt.show()
peaks, properties = find_peaks(x, prominence=10)
plt.plot(tr)
plt.plot(np.abs(np.diff(tr)))
plt.hlines(y=5,xmin=0,xmax=600)
plt.plot(peaks, x[peaks], "x")
plt.show()
##
t = np.abs(t2)
plt.plot(np.abs(t))
plt.plot(np.abs(np.diff(tr)))
peaks, properties = find_peaks(np.abs(t), prominence=10)
plt.plot(peaks, t[peaks], "x")
##
x = cut["LeftPaw"]['x']
y = cut["LeftPaw"]['y']

print(len([elem for elem in y if np.isnan(elem)]))
##
#todo add here future exclusion of jump
def correct_n_interpolate(trial,parts_to_interpolate,factor_of_point=1):
    dim = ['x','y']
    for d in dim:
        for part in parts_to_interpolate:
            values = trial[part][d]
            time = np.arange(1,len(values)+1,1)

            values_wo_nan = [elem for elem in values if not np.isnan(elem)]
            time_wo_nan = [i for i, j in zip(time, values) if not np.isnan(j)]

            f = interpolate.PchipInterpolator(time_wo_nan,values_wo_nan, axis=0, extrapolate=None)

            time_new = np.arange(1, len(values)+1,factor_of_point)
            values_new= f(time_new)

            trial[part][d+'-interpolated']=values_new

correct_n_interpolate(cut,["LeftPaw","RightPaw"])

def return_vel_during_reach(trial,parts_to_vel):
    dim = ['x','y']
    for d in dim:
        for part in parts_to_vel:
            trial[part][d+'-vel']= savgol_filter(trial[part][d+'-interpolated'],11,3,1)

return_vel_during_reach(cut,["LeftPaw","RightPaw"])

##
def find_movement_initiation(trial,reaching_paw):
    maxi_vel = np.nanargmin(trial[reaching_paw]['x-vel'])
    span_to_see = trial[reaching_paw]['x-interpolated'][:maxi_vel]

    mean_before_init = np.mean(span_to_see)
    std_before_init = np.std(span_to_see)

    init = next(x for x in span_to_see if x > mean_before_init+std_before_init/2)
    trial["MovementInit"]=np.where(span_to_see==init)[0][0]



find_movement_initiation(cut,'LeftPaw')
##
def plot_vel_reach( reach_in_trial,reachingpaw,figure, position_in_grid, axis):

    ax = figure.add_subplot(position_in_grid)
    lin = np.linspace(1, len(reach_in_trial[reachingpaw][axis]), len(reach_in_trial[reachingpaw][axis]))

    ax.plot(lin, reach_in_trial[reachingpaw][axis], '-o', markersize=3, label=reachingpaw)

    ax.legend()
    ax.set_title(" velocity of "+reachingpaw+ ' '+axis)
    ax.set_xlabel("time")
    ax.set_ylabel(axis)

def plot_position_reach(trial,reaching_paw,figure,position_in_grid,axis):
    ax = figure.add_subplot(position_in_grid)
    lin = np.linspace(1, len(trial["RightPaw"][axis]), len(trial["RightPaw"][axis]))

    maxi_pos = np.nanargmin(trial[reaching_paw]['x-interpolated'])
    maxi_vel = np.nanargmin(trial[reaching_paw]['x-vel'])
    mvmt_initiation = trial['MovementInit']

    ax.vlines(x=maxi_pos, ymin=90, ymax=400,linestyles='dashed',color='black',linewidth=2,label='MaxPosX '+reaching_paw)
    ax.vlines(x=maxi_vel, ymin=90, ymax=400,linestyles='dashed',color='black',linewidth=2,label='MaxVelX '+reaching_paw)
    ax.vlines(x=mvmt_initiation, ymin=90, ymax=400,linestyles='dashed',color='black',linewidth=2,label='MovInitX '+reaching_paw)

    ax.plot(lin, trial["RightPaw"][axis], '-o',markersize = 3,label="RightPaw")
    ax.plot(lin, trial["LeftPaw"][axis], '-o',markersize = 3,label="LeftPaw")


    ax.legend()
    ax.set_title(axis+" position of paws")
    ax.set_xlabel("time")
    ax.set_ylabel(axis)

def plot_reach_in_trial(reach_in_trial,reaching_paw,bodyparts=['Nose', 'Tongue', 'Droplet','LeftPaw', 'RightPaw']):
    markers = ['<','<','v','o','o']
    colors = ['Greens','Wistia','Greys','Reds','Blues']
    colors_ = ['green', 'orange', 'gray','red', 'blue']

    fig = plt.figure(tight_layout=True)
    grid = plt.GridSpec(ncols=3, nrows=2, width_ratios=[3, 3,3])

    ax1 = fig.add_subplot(grid[0:3,0])

    for k in range(len(bodyparts)):
        x = reach_in_trial[bodyparts[k]]['x'][0]
        y = reach_in_trial[bodyparts[k]]['y'][0]
        ax1.scatter(x, y, marker=markers[k], label=bodyparts[k], c=colors_[k])

    for i in range(len(bodyparts)):
        x = reach_in_trial[bodyparts[i]]['x']
        y = reach_in_trial[bodyparts[i]]['y']
        lin = np.linspace(1, len(x), len(x))
        ax1.scatter(x,y,marker = markers[i],c = lin,cmap=colors[i])

    ax1.set_ylim(500, 250)
    ax1.set_xlim(50,350)
    ax1.legend()
    ax1.set_title("Trajectory of each object")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Plot traces of paws x
    plot_position_reach(reach_in_trial,reaching_paw,fig,grid[0,1],'x-interpolated')

    # Plot traces of paws y
    plot_position_reach(reach_in_trial, reaching_paw,fig, grid[1,1], 'y-interpolated')

    # Plot velocity of reaching paw
    plot_vel_reach(reach_in_trial,reaching_paw, fig, grid[0,2], 'x-vel')
    plot_vel_reach(reach_in_trial,reaching_paw, fig, grid[1,2], 'y-vel')



    fig.suptitle(reach_in_trial["Name"]+reach_in_trial["T"]+"_Reach")
    plt.show()

plot_reach_in_trial(cut,'LeftPaw')
##
for i in range(2,12):
    print(i)
    cut = find_reaches(data[i],"LeftPaw")
    plot_current_trial(data[i])
    correct_n_interpolate(cut,["LeftPaw","RightPaw"])
    return_vel_during_reach(cut,["LeftPaw","RightPaw"])
    find_movement_initiation(cut,'LeftPaw')
    plot_reach_in_trial(cut,'LeftPaw')

##
fig, ax= plt.subplots(1,2)

x = cut["LeftPaw"]['x']
xneew = cut["LeftPaw"]['x-interpolated']
l = len(x)
time = np.arange(1, l+ 1, 1)
time_new = np.arange(1, l+ 1, 1)

ax[0].plot(time_new,xneew,'-')

ax[0].plot(time,x,'x')

ax[1].plot(time_new[:-1],np.diff(xneew))
from scipy.ndimage import gaussian_filter1d

ax[1].plot(time_new[:-1],gaussian_filter1d(np.diff(xneew),1),c='r')
plt.show()

##
der = ac.derivative(1)

fig, ax= plt.subplots(1,2)

x = cut["LeftPaw"]['x']
xneew = cut["LeftPaw"]['x-interpolated']
l = len(x)
time = np.arange(1, l+ 1, 1)
time_new = np.arange(1, l+ 1, 1)

ax[0].plot(time_new,xneew,'-')

ax[0].plot(time,x,'x')

ax[1].plot(time_new[:],der(time_new))

ax[1].plot(time_new,gaussian_filter1d(der(time_new),1),c='r')
plt.show()
##
fig, ax= plt.subplots(1,2)
ax[0].plot(time_new,xneew,'-')

ax[0].plot(time,x,'x')
ax[1].plot(time_new[:-1],gaussian_filter1d(np.diff(xneew),1),c='r')
ax[1].plot(time_new[:],savgol_filter(xneew,11,3,1))
ax[1].plot(time_new[:],savgol_filter(der(time_new),10,3,1),c='g')




##
print(max((v, i) for i, v in enumerate(gaussian_filter1d(der(time_new),1)))[1])
print(max((v, i) for i, v in enumerate(gaussian_filter1d(np.diff(xneew),1)))[1])
print(max((v, i) for i, v in enumerate(savgol_filter(xneew,11,3,1)))[1])


##
from scipy import interpolate
x = np.arange(1,100,1)

y = np.sin(x)
plt.plot(x,y)
plt.show()
b = interpolate.PchipInterpolator(x, y, axis=0, extrapolate=None)
xnew = np.arange(1, 100, 0.1)
ynew = b(xnew)
plt.plot(x, y, 'o', xnew, ynew, '-')
plt.show()
##
y[20] = np.nan
y_wo_nan = [elem for elem in y if not np.isnan(elem)]
x_wo_nan = [i for i,j in zip(x,y) if not np.isnan(j)]

b = interpolate.PchipInterpolator(x_wo_nan, y_wo_nan, axis=0, extrapolate=True)
xnew = np.arange(1, 100, 0.1)
ynew = b(xnew)
plt.plot(x, y, 'o',x_wo_nan,y_wo_nan,'x', xnew, ynew, '-')
plt.show()

##

