import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import numpy as np
from entropy_train_gui import EntropyTrainGUI
from feature_selection_gui import FeatureSelectionGUI
from proc_entropy import ProcEEGentropy
import mne
import sys
from bciloop_utilities.SpatialFilters import CommonSpatialPatterns, car_filter
from bciloop_utilities.Integrators import ExponentialIntegrator
from bciloop_utilities.proc_fisher2 import proc_fisher2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.io import savemat, loadmat
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
import logging
import math

#proc_pos2win(pos, wshift, direction, wlength)
def proc_pos2win(pos, wshift, direction, *args):
    backward = False

    if(direction == 'forward'):
        wlength = []
    elif (direction == 'backward'):
        backward = True
        if(len(args) == 0 or len(args)>1):
            raise ValueError("Backward direction option requires to provide a wlength argument.")
        wlength = args[0]
    else:
        raise ValueError("Direction not recognized: only \'forward\' and \'backward\' are allowed.")

    wpos = np.floor(pos/wshift) + 1

    if(backward == True):
        wpos = wpos - (np.floor(wlength/wshift))
    return wpos.astype(int)


def isMainEvent(code, code_list):
    if (code == 1 or code == 771 or code == 773 or code == 781 or code == 786 or code in code_list):
        return True
    else:
        return False

def getKey(dict, value):
    for key in dict:
        val = dict[key]
        if val == value:
            return int(key)
    return -1

def loadGDF(filename, code_list):
    #get raw gdf
    raw = mne.io.read_raw_gdf(filename)
    #exctract events
    curr_s = np.transpose(raw.get_data())
    curr_h = np.empty((0,3), dtype=int)
    raw_events, event_dic = mne.events_from_annotations(raw)

    for i in range(len(raw_events)):

        currentID = getKey(event_dic, raw_events[i, 2])

        if(isMainEvent(currentID, codes)):
            #initialize new row vector
            newEventRow = np.zeros((1,3), dtype=int)
            #take position
            newEventRow[0,0] = int(raw_events[i,0])
            #take code
            newEventRow[0,2] = int(currentID)
            curr_h = np.vstack((curr_h, newEventRow))

        else:
            if(i==0):
                print('Mistakes were made. Can\'t find original event.')
                return

            #use the off event to compute the duration of the previous one
            prevID = getKey(event_dic, raw_events[i-1, 2])
            #insert duration
            if(currentID == (32768+prevID)):
                curr_h[len(curr_h)-1,1] = int(raw_events[i,0]-raw_events[i-1,0])

    return curr_s, curr_h

def apply_mask(n_bands, mask, csp_train_data, *args):
    if len(args) == 0:
        train_data = np.empty((np.shape(csp_train_data)[0],0))
        for i in range(n_bands):
            train_data = np.hstack((train_data, csp_train_data[:, mask[:,i] == 1, i]))
        return train_data
    elif len(args) == 1:
        csp_test_data = args[0]
        train_data = np.empty((np.shape(csp_train_data)[0],0))
        test_data = np.empty((np.shape(csp_test_data)[0],0))
        for i in range(n_bands):
            train_data = np.hstack((train_data, csp_train_data[:, mask[:,i] == 1, i]))
            test_data = np.hstack((test_data, csp_test_data[:, mask[:,i] == 1, i]))
        return train_data, test_data

    else:
        raise ValueError('Too many parameters in apply_mask function.')



#OPEN GUI
my_gui = EntropyTrainGUI()
my_gui.root.mainloop()
# Settings idea:per i file da usare li faccio mettere in un array ad ogni selezione

#variabili ancora da implementare nella my_gui
files = my_gui.files #contiene array di path di file gdf da caricare, oppure devo farne caricare solo uno? implemento solo uno per ora
first_class_codes = my_gui.first_class_codes #codici classi
second_class_codes = my_gui.second_class_codes
codes = first_class_codes + second_class_codes

window_length = float(my_gui.window_length.get())
window_shift = float(my_gui.window_shift.get())
srate = int(my_gui.srate.get())
filter_order_list = my_gui.filter_order_list
filter_lowf_list = my_gui.filter_lowf_list
filter_highf_list = my_gui.filter_highf_list
nbins = int(my_gui.nbins.get())
cspdimm = int(my_gui.cspdimm.get())
alpha = float(my_gui.alpha.get())
threshold = float(my_gui.threshold.get())
rejection = float(my_gui.rejection.get())
begin = float(my_gui.begin.get())
load_data = bool(my_gui.load_data.get())
save_classifier = bool(my_gui.save_classifier.get())
classifier_name = str(my_gui.className.get())
car_filter_flag = bool(my_gui.car_filter.get())
laplacian = bool(my_gui.laplacian.get())
lap_path = str(my_gui.lap_path.get())
N_CH = 16
n_bands = len(filter_order_list)
nfiles = len(files)
log_flag = bool(my_gui.logVar.get())

if len(first_class_codes) == 0 or len(second_class_codes) == 0:
    raise ValueError('One of the classes is empty.')

if log_flag is True:
    #logging configuration
    #logger = logging.getLogger('entropy_training_logger')
    logging.basicConfig(filename='entropy_train.log', encoding='utf-8', level=logging.INFO,format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.info('Start of the offline training.')
    logging.info(f'First class codes: {first_class_codes}')
    logging.info(f'Second class codes: {second_class_codes}')
    logging.info(f'Sample rate: {srate}')
    logging.info(f'Window length: {window_length}')
    logging.info(f'Window shift: {window_shift}')

    for i in range(n_bands):
        logging.info(f'Filter {i} order: {filter_order_list[i]}')
        logging.info(f'Filter {i} lowf: {filter_lowf_list[i]}')
        logging.info(f'Filter {i} highf: {filter_highf_list[i]}')

    logging.info(f'Nbins: {nbins}')
    logging.info(f'Csp dim: {cspdimm}')
    logging.info(f'Alpha: {alpha}')
    logging.info(f'Threshold: {threshold}')
    logging.info(f'Rejection: {rejection}')
    logging.info(f'Begin: {begin}')

    if car_filter_flag:
        logging.info('Car filter selected')
    if laplacian:
        logging.info('Laplacian filter selected')

    logging.shutdown()


print('[TrainEntropyClassifier] Initialize processing')

proc_entropy = ProcEEGentropy(window_length,
                              window_shift,
                              nbins,
                              filter_lowf_list, filter_highf_list,
                              filter_order_list,
                              srate)
csp = [CommonSpatialPatterns(cspdimm)]*n_bands
clf = LinearDiscriminantAnalysis(solver='eigen', priors=[0.5, 0.5], shrinkage='auto')
integrator = ExponentialIntegrator(alpha, threshold, rejection, begin)

# COMPUTE OR LOAD ENTROPY
if load_data is True:
    print('Loading data..')
    entropy_walking_dic = loadmat('entropy_walking.mat')
    entropy_standing_dic = loadmat('entropy_standing.mat')

    full_entropy_walking = entropy_walking_dic['data']
    full_entropy_standing = entropy_standing_dic['data']
    entropy_standing_start = entropy_standing_dic['filestart'][0]
    entropy_walking_start = entropy_walking_dic['filestart'][0]
    entropy_standing_end = entropy_standing_dic['fileend'][0]
    entropy_walking_end = entropy_walking_dic['fileend'][0]
    nfiles = len(entropy_standing_start)


else:
    # Load subject's dataset
    print('[TrainEntropyClassifier] Loading subject dataset')

    #take the data from the gdf
    s = np.empty((0,N_CH))
    ctyp = np.empty((0,1))
    cpos = np.empty((0,1))
    cdur = np.empty((0,1))
    file_start = np.empty((0,1))
    file_end = np.empty((0,1))

    print('Loading gdf file.')

    for file in files:

        #exctract events
        curr_s, curr_h = loadGDF(file, codes)

        curr_s = curr_s[:, :-1]

        ctyp = np.vstack((ctyp, np.reshape(curr_h[:,2], (len(curr_h),1))))
        cpos = np.vstack((cpos, np.reshape(curr_h[:,0]+len(s), (len(curr_h),1))))
        cdur = np.vstack((cdur, np.reshape(curr_h[:,1], (len(curr_h),1))))

        file_start = np.vstack((file_start, np.shape(s)[0]))
        s = np.vstack((s, curr_s))
        file_end = np.vstack((file_end, np.shape(s)[0]-1))

    if(laplacian):
        entropy_s = proc_entropy.apply_offline(s, lap_path)
    else:
        entropy_s = proc_entropy.apply_offline(s)

    #np.save('entropy_s.npy', entropy_s)
    #update event vectors with new frequency 1/window_shift
    wpos = proc_pos2win(cpos, window_shift*srate, 'backward', window_length*srate)
    wdur = (np.floor(cdur/(window_shift*srate)) + 1).astype(int)
    wtyp = ctyp.astype(int)
    n_samples = len(entropy_s)
    wfile_start = np.reshape(proc_pos2win(file_start, window_shift*srate, 'backward', window_length*srate), (nfiles,))
    wfile_end = np.reshape(proc_pos2win(file_end, window_shift*srate, 'backward', window_length*srate), (nfiles,))

    #extract trials
    trialStart = wpos[np.any([wtyp == id for id in codes], axis=0)]
    trialStop = trialStart + wdur[np.any([wtyp == id for id in codes], axis=0)] + wdur[ctyp == 781] -1
    trialStop = trialStop.astype(int)
    trialCode = wtyp[np.any([wtyp == id for id in codes], axis=0)]
    ntrials = len(trialStart)

    print("Creating trial labels vector.")

    sample_labels = np.zeros(n_samples)
    cumulated_dur = 0
    end_of_trials = False
    #print(trialStop[29])
    #print(wfile_end[0])


    for id in range(ntrials):
        cstart = trialStart[id]
        cstop = trialStop[id]

        if (id < ntrials-1):
            cnextstop = trialStop[id+1]
        else:
            end_of_trials = True
        clab = trialCode[id]
        sample_labels[cstart:cstop+1] = clab
        cumulated_dur = cumulated_dur + (cstop - cstart + 1)

        for i in range(nfiles):
            if (cstop <= wfile_end[i]):

                if (end_of_trials):
                    wfile_end[i] = cumulated_dur -1
                elif (cnextstop > wfile_end[i]):
                    wfile_end[i] = cumulated_dur -1
                    wfile_start[i+1] = cumulated_dur

    #SAVE DATA
    full_entropy_data = entropy_s[np.any([sample_labels == id for id in codes], axis=0), :, :]
    full_entropy_labels = sample_labels[np.any([sample_labels != 0], axis=0)]
    #full_entropy_standing = entropy_s[np.any([sample_labels == id for id in first_class_codes], axis=0), :, :]
    #full_entropy_walking = entropy_s[np.any([sample_labels == id for id in second_class_codes], axis=0), :, :]
    #np.save('entropy_walking.npy', full_entropy_walking)
    #np.save('entropy_standing.npy', full_entropy_standing)
    #np.save('wfile_start.npy', wfile_start)
    #np.save('wfile_end.npy', wfile_end)
    #savemat('entropy.mat', {'standing': full_entropy_standing,'walking': full_entropy_walking})
    savemat('entropy.mat', {'data': full_entropy_data,
                            'filestart': wfile_start,
                            'fileend': wfile_end })

    entropy_standing_start = []
    entropy_walking_start = []
    entropy_standing_end = []
    entropy_walking_end = []
    full_entropy_standing = np.empty((0,N_CH, n_bands))
    full_entropy_walking = np.empty((0,N_CH, n_bands))

    for i in range(nfiles):
        file_data = full_entropy_data[wfile_start[i]:wfile_end[i],:,:]
        standing_file_data = full_entropy_data[np.any([full_entropy_labels == id for id in first_class_codes], axis=0), :, :]
        walking_file_data = full_entropy_data[np.any([full_entropy_labels == id for id in second_class_codes], axis=0), :, :]

        entropy_standing_start.append(np.shape(full_entropy_standing)[0])
        entropy_walking_start.append(np.shape(full_entropy_walking)[0])

        full_entropy_standing = np.vstack((full_entropy_standing, standing_file_data))
        full_entropy_walking = np.vstack((full_entropy_walking, walking_file_data))

        entropy_standing_end.append(np.shape(full_entropy_standing)[0]-1)
        entropy_walking_end.append(np.shape(full_entropy_walking)[0]-1)


    savemat('entropy_standing.mat', {'data': full_entropy_standing,
                            'filestart': entropy_standing_start,
                            'fileend': entropy_standing_end })
    savemat('entropy_walking.mat', {'data': full_entropy_walking,
                            'filestart': entropy_walking_start,
                            'fileend': entropy_walking_end })

#GET TRAINING DATA
full_train_data = np.empty((0, N_CH, n_bands))
full_test_data = np.empty((0, N_CH, n_bands))
train_labels = np.empty((0, 1))
test_labels = np.empty((0, 1))
file_end = []
file_start = []


check_standing = full_entropy_standing
check_walking = full_entropy_walking
standing_count = 0
walking_count = 0

print(np.shape(check_standing))
print(np.shape(check_walking))

print('Standing chek:')
for i in range(np.shape(full_entropy_standing)[0]):
    if math.isnan(check_standing[i,0,0]):
        print(f'Row {i} has a NaN value.')
        full_entropy_standing = np.delete(full_entropy_standing, i-standing_count, axis=0)
        print(np.shape(full_entropy_standing))
        standing_count = standing_count+1
        for file in range(nfiles):
            if entropy_standing_start[file] > i:
                entropy_standing_start[file] = entropy_standing_start[file]-1
            if entropy_standing_end[file] >= i:
                entropy_standing_end[file] = entropy_standing_end[file]-1

print('Walking chek:')
for i in range(np.shape(full_entropy_walking)[0]):
    if math.isnan(check_walking[i,0,0]):
        print(f'Row {i} has a NaN value.')
        full_entropy_walking = np.delete(full_entropy_walking, i-walking_count, axis=0)
        walking_count = walking_count+1
        for file in range(nfiles):
            if entropy_walking_start[file] > i:
                entropy_walking_start[file] = entropy_walking_start[file]-1
            if entropy_walking_end[file] >= i:
                entropy_walking_end[file] = entropy_walking_end[file]-1

#create training and test datasets
for i in range(nfiles):
    entropy_standing = full_entropy_standing[entropy_standing_start[i]:entropy_standing_end[i]+1, :, :]
    entropy_walking = full_entropy_walking[entropy_walking_start[i]:entropy_walking_end[i]+1, :, :]
    np.random.shuffle(entropy_standing)
    np.random.shuffle(entropy_walking)

    #balance classes
    ndiff = np.shape(entropy_walking)[0] - np.shape(entropy_standing)[0]
    halfdataset_dim = 0
    if ndiff > 0:
        halfdataset_dim = np.shape(entropy_standing)[0]
    else:
        halfdataset_dim = np.shape(entropy_walking)[0]

    training_set_dim = np.floor(halfdataset_dim*0.8).astype(int)
    test_set_dim = halfdataset_dim - training_set_dim
    file_start.append(np.shape(full_train_data)[0])
    full_train_data = np.vstack((full_train_data, entropy_standing[0:training_set_dim, :, :]))
    train_labels = np.vstack((train_labels, np.zeros((training_set_dim,1))))
    full_train_data = np.vstack((full_train_data, entropy_walking[0:training_set_dim, :, :]))
    train_labels = np.vstack((train_labels, np.ones((training_set_dim,1))))
    file_end.append(np.shape(full_train_data)[0]-1)

    full_test_data = np.vstack((full_test_data, entropy_standing[training_set_dim:training_set_dim+test_set_dim, :, :]))
    test_labels = np.vstack((test_labels, np.zeros((test_set_dim,1))))
    full_test_data = np.vstack((full_test_data, entropy_walking[training_set_dim:training_set_dim+test_set_dim, :, :]))
    test_labels = np.vstack((test_labels, np.ones((test_set_dim,1))))

train_labels = np.reshape(train_labels, (np.shape(train_labels)[0],))
test_labels = np.reshape(test_labels, (np.shape(test_labels)[0],))

# TRAINING THE CLASSIFIER
csp_train_data = np.empty((np.shape(full_train_data)[0], cspdimm, n_bands))
csp_test_data = np.empty((np.shape(full_test_data)[0], cspdimm, n_bands))

print('[TrainEntropyClassifier] Training the classifier')
for i in range(n_bands):
    #extract training and testing data
    curr_train_data = full_train_data[:, :, i]
    curr_test_data = full_test_data[:, :, i]
    # CSP filter
    csp[i].compute_filters(curr_train_data, train_labels)
    csp_train_data[:, :, i] = csp[i].apply(curr_train_data)
    csp_test_data[:, :, i] = csp[i].apply(curr_test_data)
    # LDA training
    #clf[i].fit(csp_train_data, train_labels)

single_file_fisherscore = np.empty((cspdimm, n_bands, nfiles))
mean_fisherscore = proc_fisher2(csp_train_data, train_labels)

for i in range(nfiles):
    single_file_train_data = csp_train_data[file_start[i]:file_end[i]+1, :, :]
    single_file_train_label = train_labels[file_start[i]:file_end[i]+1]
    single_file_fisherscore[:, :, i] = proc_fisher2(single_file_train_data, single_file_train_label)


selection_gui = FeatureSelectionGUI(single_file_fisherscore, mean_fisherscore)
selection_gui.root.mainloop()

#get mask and select features
mask = selection_gui.mask
train_data, test_data = apply_mask(n_bands, mask, csp_train_data, csp_test_data)
'''
train_data = np.empty((np.shape(csp_train_data)[0],0))
test_data = np.empty((np.shape(csp_test_data)[0],0))

for i in range(n_bands):
    train_data = np.hstack((train_data, csp_train_data[:, mask[:,i] == 1, i]))
    test_data = np.hstack((test_data, csp_test_data[:, mask[:,i] == 1, i]))
'''
#LDA training
clf.fit(train_data, train_labels)

dproba = clf.predict_proba(test_data)
pred, ipp = integrator.apply_array(dproba)
acc = accuracy_score(test_labels, pred)
print('     Accuracy: ' + str(acc))



# TRAIN CLASSIFIER WITH ALL THE DATASET
if save_classifier is True:
    print('[TrainEntropyClassifier] Training and saving classifier')
    full_data = np.vstack((full_train_data, full_test_data))
    labels = np.concatenate([train_labels, test_labels])
    csp_data = np.empty((np.shape(full_data)[0], cspdimm, n_bands))
    for i in range(n_bands):
        #extract training and testing data
        curr_data = full_data[:, :, i]
        # CSP filter
        csp[i].compute_filters(curr_data, labels)
        csp_data[:, :, i] = csp[i].apply(curr_data)

    data =  apply_mask(n_bands, mask, csp_data)
    # LDA training
    clf.fit(data, labels)
    # save
    for i in range(n_bands):
        np.save('csp_coeff_{}.npy'.format(i+1), csp[i].coeff)
    pickle.dump(clf, open(classifier_name, 'wb'))
print('[TrainEntropyClassifier] Successfully exiting the program!')
