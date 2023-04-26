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
                   'y':trial[bodypart]['y'][interval[0]:interval[1]],
                   'z':trial[bodypart]['z'][interval[0]:interval[1]]}
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
tresh =40
cut = find_reaches(data[184],"LeftPaw")

reach = cut["LeftPaw"]['x']
plt.plot(reach,'o-')
plt.plot(np.diff(reach),'o-')
no_nans_loc=[] # input without nans
i= 0
for elem in reach:
    if not np.isnan(elem):
        no_nans_loc.append(i)
    i+=1
med = np.nanmedian(reach[1:30])
next_nearmed = [item for item in reach if abs(item-med)<tresh][0]
if abs(med-reach[0])>tresh:
    indx =np.where(abs(np.array(reach)-next_nearmed)<0.00001)[0][0]
    reach[0:indx]=[np.nan]*(indx+1)
diff = np.diff(reach)
indx_jump = np.where(abs(diff)>tresh)[0]
nb_jump = len(indx_jump)
flag_unknown_jump = False
#check if there are jumps that didn't come back
if nb_jump !=0:
    noBack_jumps = np.where(np.diff(np.sign(diff[indx_jump]))==0)[0]
    if len(noBack_jumps)!=0:
        if indx_jump[noBack_jumps[-1]] >= no_nans_loc[-4]:
            indx_jump.append(no_nans_loc[-1])
            noBack_jumps = np.where(np.diff(np.sign(diff[indx_jump])) == 0)[0] #reapeat after excluding end of data

        # ADD COME BACK JUMP, IF IT DIDN'T RETURN DUE TO NANS

        added_stops = []

        for i in range(len(noBack_jumps)):
            idx = noBack_jumps[i]
            find_nans = np.where(np.isnan(reach[indx_jump[idx]:indx_jump[idx+1]]))[0]
            find_nans
            if len(find_nans)!=0:
                find_first_jump_nan = np.where(np.diff(find_nans)>1)[0]
                if len(find_first_jump_nan)==0:
                    next_nonNan = indx_jump[idx]+ np.where(np.diff(find_nans)>=1)[0][0] + find_nans[0] +1## check why -1
                else:
                    next_nonNan = indx_jump[idx] +find_first_jump_nan[0] +find_nans[0] +1

                added_stops.append(next_nonNan)
            else:
                print('Warning! jump without return, uknown why... remove it')
                to_remove = np.where(indx_jump==idx)[0][0]
                indx_jump = np.delete(indx_jump,to_remove)
                flag_unknown_jump=True
    ind_jmp_tmp = np.concatenate((indx_jump,added_stops),dtype=int)
    indx_jump = np.sort(ind_jmp_tmp)


# some_list[start:stop:step]
ind_start = indx_jump[0::2]
ind_stop = indx_jump[1::2]
flag_large_jump = False
if len(ind_stop)!= 0:
    # Ignore jumps too large
    remove_start = []
    remove_stop = []
    for i in range(len(ind_start)):
        if len(ind_stop)-1<i:
            remove_start.append(i)
            print("Uneven number of jump, jump without return here",ind_start[i])
        else:
            if ind_stop[i]-ind_start[i]>20:
                remove_start.append(i)
                remove_stop.append(i)
                print("JumpTooLarge! Interval",ind_start[i],'-',ind_stop[i])
                flag_large_jump = True
    if len(remove_start)!=0:
        ind_start = [int(elem) for elem in ind_start if elem not in remove_start]
        # for elem in remove_start:
        #     ind_start.remove(elem)

    if len(remove_stop)!=0:
        ind_stop = [int(elem) for elem in ind_stop if elem not in remove_stop]

        # for elem in remove_stop:
        #     ind_stop.remove(elem)
    print("prout")

    #replace jump by nan
    for i in range(len(ind_stop)):
        reach[ind_start[i]:ind_stop[i]+1]=[np.nan]*(ind_stop[i]-ind_start[i]+1)





print(next_nearmed)
print(nb_jump)
print(noBack_jumps)
print(ind_start)
print(ind_stop)
plt.plot(reach)
##
#todo add here future exclusion of jump
def remove_DLC_jumps(trial_reach,thresh,paws,dim):

    for paw in paws:
        reach = trial_reach[paw][dim]
        #Correct for start / end / all points = nan
        if np.isnan(reach[0]):
            reach[0]=next((item for item in reach if not np.isnan(item)),'All elem of reach are nan!'+trial_reach["Name"])
        if np.isnan(reach[-1]):
            reach[-1] = next((item for item in reach[::-1] if not np.isnan(item)),'All elem of reach are nan!' + trial_reach["Name"])

        no_nans_loc = []  # input without nans
        i = 0
        for elem in reach:
            if not np.isnan(elem):
                no_nans_loc.append(i)
            i += 1
        med = np.nanmedian(reach[1:30])
        next_nearmed = [item for item in reach if abs(item - med) < tresh][0]
        if abs(med - reach[0]) > tresh:
            indx = np.where(abs(np.array(reach) - next_nearmed) < 0.00001)[0][0]
            reach[0:indx] = [np.nan] * (indx)
        diff = np.diff(reach)
        indx_jump = np.where(abs(diff) > tresh)[0]
        nb_jump = len(indx_jump)
        flag_unknown_jump = False
        # check if there are jumps that didn't come back
        if nb_jump != 0:
            noBack_jumps = np.where(np.diff(np.sign(diff[indx_jump])) == 0)[0]
            if len(noBack_jumps) != 0:
                if indx_jump[noBack_jumps[-1]] >= no_nans_loc[-4]:
                    indx_jump.append(no_nans_loc[-1])
                    noBack_jumps = np.where(np.diff(np.sign(diff[indx_jump])) == 0)[0]  # reapeat after excluding end of data

                # ADD COME BACK JUMP, IF IT DIDN'T RETURN DUE TO NANS

                added_stops = []

                for i in range(len(noBack_jumps)):
                    idx = noBack_jumps[i]
                    find_nans = np.where(np.isnan(reach[indx_jump[idx]:indx_jump[idx + 1]]))[0]
                    find_nans
                    if len(find_nans) != 0:
                        find_first_jump_nan = np.where(np.diff(find_nans) > 1)[0]
                        if len(find_first_jump_nan) == 0:
                            next_nonNan = indx_jump[idx] + np.where(np.diff(find_nans) >= 1)[0][0] + find_nans[0] + 1  ## check why -1
                        else:
                            next_nonNan = indx_jump[idx] + find_first_jump_nan[0] + find_nans[0] + 1
              #          if abs(reach[indx_jump[idx]] - reach[next_nonNan]) > 25:
                         added_stops.append(next_nonNan)
                    else:
                        print('Warning! jump without return, uknown why... remove it')
                        to_remove = np.where(indx_jump == idx)[0][0]
                        indx_jump = np.delete(indx_jump, to_remove)
                        flag_unknown_jump = True
                ind_jmp_tmp = np.concatenate((indx_jump, added_stops))
                indx_jump = np.sort(ind_jmp_tmp)

        # some_list[start:stop:step]
        ind_start = indx_jump[0::2]
        ind_stop = indx_jump[1::2]
        flag_large_jump = False
        if len(ind_stop) != 0:
            # Ignore jumps too large
            remove_start = []
            remove_stop = []
            for i in range(len(ind_start)):
                if len(ind_stop) - 1 < i:
                    remove_start.append(i)
                    print("Uneven number of jump, jump without return here", ind_start[i])
                else:
                    if ind_stop[i] - ind_start[i] > 20:
                        remove_start.append(i)
                        remove_stop.append(i)
                        print("JumpTooLarge! Interval", ind_start[i], '-', ind_stop[i])
                        flag_large_jump = True
            if len(remove_start) != 0:
                ind_start = [elem for elem in ind_start if elem not in remove_start]
                # for elem in remove_start:
                #     ind_start.remove(elem)

            if len(remove_stop) != 0:
                ind_stop = [elem for elem in ind_stop if elem not in remove_stop]

                # for elem in remove_stop:
                #     ind_stop.remove(elem)

            # replace jump by nan
            for i in range(len(ind_stop)):
                reach[ind_start[i]:ind_stop[i] + 1] = [np.nan] * (ind_stop[i] - ind_start[i] + 1)

        trial_reach[paw][dim+'-NoJumps']=reach

cut = find_reaches(data[133],"LeftPaw")
remove_DLC_jumps(cut,60,["LeftPaw"],'z')
##
plt.plot(cut["LeftPaw"]['x'])
##
def correct_n_interpolate(trial,parts_to_interpolate,dim,factor_of_point=1):
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
##
thresh =60
cut = find_reaches(data[4],"LeftPaw")
reach = cut["LeftPaw"]['x']

def remove_DLCjump_interpolate(trial_cutted,thresh,paws):
    dim = ['x','y','z']
    for d in dim:
        remove_DLC_jumps(trial_cutted,thresh, paws, d)

    dim_to_int = ['x-NoJumps','y-NoJumps','z-NoJumps']


    correct_n_interpolate(trial_cutted, paws, dim_to_int)


##
cut = find_reaches(data[2],"LeftPaw")
remove_DLCjump_interpolate(cut,60,["LeftPaw","RightPaw"])
plt.plot(cut["LeftPaw"]['x'],'black')
plt.plot(cut["LeftPaw"]['x-NoJumps'],'-o')
plt.plot(cut["LeftPaw"]['x-NoJumps-interpolated'],'-x')

##

def return_vel_during_reach(trial,parts_to_vel):
    dim = ['x','y','z']
    for d in dim:
        for part in parts_to_vel:
            trial[part][d+'-vel']= savgol_filter(trial[part][d+'-NoJumps-interpolated'],11,3,1)

return_vel_during_reach(cut,["LeftPaw","RightPaw"])

##
def find_movement_initiation(trial,reaching_paw):
    maxi_vel = np.nanargmin(trial[reaching_paw]['x-vel'][30:])
    span_to_see = trial[reaching_paw]['x-NoJumps-interpolated'][:maxi_vel]
    mean_before_init = np.mean(span_to_see)
    std_before_init = np.std(span_to_see)

    init = next((x for x in span_to_see if x > mean_before_init+std_before_init/4), -100)
    if init != -100:
        trial["MovementInit"]=np.where(span_to_see==init)[0][0]
    else:
        trial["MovementInit"]= maxi_vel


cut = find_reaches(data[169],"LeftPaw")
remove_DLCjump_interpolate(cut, 40, ["LeftPaw", "RightPaw"])

return_vel_during_reach(cut, ["LeftPaw", "RightPaw"])

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

    maxi_pos = np.nanargmin(trial[reaching_paw]['x-NoJumps-interpolated'])
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
    plot_position_reach(reach_in_trial,reaching_paw,fig,grid[0,1],'x-NoJumps-interpolated')

    # Plot traces of paws y
    plot_position_reach(reach_in_trial, reaching_paw,fig, grid[1,1], 'y-NoJumps-interpolated')

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
    remove_DLCjump_interpolate(cut,60, ["LeftPaw", "RightPaw"])
    return_vel_during_reach(cut,["LeftPaw","RightPaw"])
    find_movement_initiation(cut,'LeftPaw')
    plot_reach_in_trial(cut,'LeftPaw')

##
ax = plt.figure().add_subplot(projection='3d')
j=0
for elem in data[:]:


    cut = find_reaches(elem,"LeftPaw")
    good_trace = True
    for entry in cut["LeftPaw"]:
        if len([point for point in cut["LeftPaw"][entry] if not np.isnan(point)]) < len(cut["LeftPaw"][entry])*9/10:
            good_trace = False
        if len(cut["LeftPaw"][entry])==0:
            good_trace = False
    for entry in cut["RightPaw"]:
        if len([point for point in cut["RightPaw"][entry] if not np.isnan(point)]) < len(cut["RightPaw"][entry])*9/10:
            good_trace = False
        if len(cut["RightPaw"][entry])==0:
            good_trace = False

    if good_trace:
        print("prout")
        print(j)

        remove_DLCjump_interpolate(cut, 50, ["LeftPaw"])
        maxi_pos = np.nanargmin(cut["LeftPaw"]['x-NoJumps-interpolated'])
        return_vel_during_reach(cut, ["LeftPaw"])

        find_movement_initiation(cut, "LeftPaw")
        mvmt_initiation = cut['MovementInit']
        x = cut["LeftPaw"]['x-NoJumps-interpolated'][mvmt_initiation:maxi_pos]
        y = cut["LeftPaw"]['y-NoJumps-interpolated'][mvmt_initiation:maxi_pos]
        z = cut["LeftPaw"]['z-NoJumps-interpolated'][mvmt_initiation:maxi_pos]
        ax.plot(x,z,y)
    j +=1
##
cut = find_reaches(data[132],"LeftPaw")
remove_DLCjump_interpolate(cut, 60, ["LeftPaw", "RightPaw"])

#remove_DLCjump_interpolate(cut, 60, ["LeftPaw", "RightPaw"])
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

