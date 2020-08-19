"""Generate images for CNNs from ATTPC events.

Author: Ryan Strauss
"""
import math

import click
import h5py
import matplotlib
import numpy as np
import os
import pandas as pd
import re
#import pytpc
from sklearn.utils import shuffle

#from utils import data_discretization as dd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Real data runs to use
RUNS = ['0130', '0210']

DATA_FILE = "Mg22_alphaalpha_digiSim.h5"

X_COL, Y_COL, Z_COL, CHARGE_COL = 0, 1, 2, 4


# Currently we're setting the image pixel values as the logarithm of the charge!
def _l(a):
    return 0 if a == 0 else math.log10(a)

def get_event_by_index(hf, i):
    return hf["Event_[{}]".format(i)][:]

def get_event_from_key(key):
    re_m = re.match("Event_\[(\d*)\]", key)
    if re_m:
        return int(re_m.groups()[0])
    else: return None

def get_label(key):
    """ Return label for an event number: 
            even: beam = 0
            odd: reaction = 1
    """
    re_m = re.match("Event_\[(\d*)\]", key)
    if re_m:
        event = int(re_m.groups()[0])
        return event % 2
    else:
        print("WARNING: could not determine label for key:", key)
        return None
    
def read_and_label_data(data_dir):
    """Read data into numpy arrays and label it. Return list with data and labels."""
    print('Processing data...')
    data = {}
    
    filename = os.path.join(data_dir, DATA_FILE)
    h5_file = h5py.File(filename, "r")

    for key in h5_file.keys():
        xyzs = np.asarray(pd.DataFrame(h5_file[key][:]))
        if xyzs.shape[0] > 0:
            #data.append([xyzs, get_label(key)])
            data[get_event_from_key(key)] = ([xyzs, get_label(key)])
        else:
            print("WARNING,", key, "has no pads firing. Removing event ...")
    h5_file.close()
    return data

def transform_normalize_data(data):
    """ Transform, shuffle and normalize for image data"""
    
    print("Transform, shuffle and normalize data ...")
    
    #transform
    log = np.vectorize(_l)
    for event in data:
        event[0][:, CHARGE_COL] = log(event[0][:, CHARGE_COL])
        
    # Normalize
    max_charge = np.array(list(map(lambda x: x[0][:, CHARGE_COL].max(), data))).max() #wrt to max in data set

    for e in data:
        for point in e[0]:
            point[CHARGE_COL] = point[CHARGE_COL] / max_charge

    # Shuffle data
    data = shuffle(data)
    
    return data, max_charge

def make_train_test_data(data, fraction_train=0.8):
    """Make train test data split"""
    
    print("Split into train and test sets ...")
    partition = int(len(data) * fraction_train)
    train = data[:partition]
    test = data[partition:]

    return train, test

def make_image_features_targets(data, projection):
    """Generate image features and targets in numpy arrays to be used in training and evaluation"""
    
    print("Make image features and targets ...")
    
    # Make numpy sets
    features = np.empty((len(data), 128, 128, 3), dtype=np.uint8)
    targets = np.empty((len(data),), dtype=np.uint8)

    for i, event in enumerate(data):
        e = event[0]
        if e is None:
            print("Event, ", i, "is None:", e)
        if projection == 'zy':
            x = e[:, Z_COL].flatten()
            z = e[:, Y_COL].flatten()
            c = e[:, CHARGE_COL].flatten()
        elif projection == 'xy':
            x = e[:, X_COL].flatten()
            z = e[:, Y_COL].flatten()
            c = e[:, CHARGE_COL].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        image = np.delete(image, 3, axis=2)
        features[i] = image
        targets[i] = event[1]
        plt.close()
    return features, targets

    
def generate_image_data_set(projection, data_dir, save_path, prefix):
    
    data = list(read_and_label_data(data_dir).values()) #from dict to list
    print("Shape:\n\tdata:", len(data))
    data, max_charge = transform_normalize_data(data)
    train, test = make_train_test_data(data, fraction_train=0.8)
    
    print("Shape:\n\ttrain:", len(train), "\n\ttest:", len(test))
    
    train_features, train_targets = make_image_features_targets(train, 'xy')
    test_features, test_targets = make_image_features_targets(test, 'xy')
    
    print('Saving to HDF5 file...')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.join(save_path, prefix + 'images.h5')

    # Save to HDF5
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('train_features', data=train_features)
    h5.create_dataset('train_targets', data=train_targets)
    h5.create_dataset('test_features', data=test_features)
    h5.create_dataset('test_targets', data=test_targets)
    h5.create_dataset('max_charge', data=np.array([max_charge]))
    h5.close()



def real_labeled(projection, data_dir, save_path, prefix):
    print('Processing data...')
    data = []
    
    filename = os.path.join(data_dir, DATA_FILE)
    h5_file = h5py.File(filename, "r")

    for key in h5_file.keys():
        #event = events[str(evt_id)]
        #xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
        #                  baseline_correction=False,
        #                  cg_times=False)
        xyzs = np.asarray(pd.DataFrame(h5_file[key][:]))
        if xyzs.shape[0] > 0:
            data.append([xyzs, get_label(key)])
        else:
            print("WARNING,", key, "has no pads firing. Removing event ...")

    log = np.vectorize(_l)
    
    print()

    for event in data:
        event[0][:, CHARGE_COL] = log(event[0][:, CHARGE_COL])

    # Shuffle data
    data = shuffle(data)

    # Split into train and test sets
    partition = int(len(data) * 0.8)
    train = data[:partition]
    test = data[partition:]

    # Normalize
    # NOTE, only wrt to max in training data set
    max_charge = np.array(list(map(lambda x: x[0][:, CHARGE_COL].max(), train))).max()

    for e in train:
        for point in e[0]:
            point[CHARGE_COL] = point[CHARGE_COL] / max_charge

    for e in test:
        for point in e[0]:
            point[CHARGE_COL] = point[CHARGE_COL] / max_charge

    print('Making images...')

    # Make train numpy sets
    train_features = np.empty((len(train), 128, 128, 3), dtype=np.uint8)
    train_targets = np.empty((len(train),), dtype=np.uint8)

    for i, event in enumerate(train):
        e = event[0]
        if projection == 'zy':
            x = e[:, Z_COL].flatten()
            z = e[:, Y_COL].flatten()
            c = e[:, CHARGE_COL].flatten()
        elif projection == 'xy':
            x = e[:, X_COL].flatten()
            z = e[:, Y_COL].flatten()
            c = e[:, CHARGE_COL].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        train_features[i] = data
        train_targets[i] = event[1]
        plt.close()

    # Make test numpy sets
    test_features = np.empty((len(test), 128, 128, 3), dtype=np.uint8)
    test_targets = np.empty((len(test),), dtype=np.uint8)

    for i, event in enumerate(test):
        e = event[0]
        if projection == 'zy':
            x = e[:, 2].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        elif projection == 'xy':
            x = e[:, 0].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        test_features[i] = data
        test_targets[i] = event[1]
        plt.close()

    print('Saving file...')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.join(save_path, prefix + 'images.h5')

    # Save to HDF5
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('train_features', data=train_features)
    h5.create_dataset('train_targets', data=train_targets)
    h5.create_dataset('test_features', data=test_features)
    h5.create_dataset('test_targets', data=test_targets)
    h5.create_dataset('max_charge', data=np.array([max_charge]))
    h5.close()


def real_unlabeled(projection, data_dir, save_path, prefix):
    print('Processing data...')
    data = []
    for run in RUNS:
        events_file = os.path.join(data_dir, 'run_{}.h5'.format(run))
        events = pytpc.HDFDataFile(events_file, 'r')

        for event in events:
            xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                              baseline_correction=False,
                              cg_times=False)

            data.append([xyzs, -1])

    # Take the log of charge data
    log = np.vectorize(_l)

    for event in data:
        event[0][:, 3] = log(event[0][:, 3])

    # Shuffle data
    data = shuffle(data)

    # Normalize
    max_charge = np.array(list(map(lambda x: x[0][:, 3].max(), data))).max()

    for e in data:
        for point in e[0]:
            point[3] = point[3] / max_charge

    print('Making images...')

    # Make numpy sets
    images = np.empty((len(data), 128, 128, 3), dtype=np.uint8)

    for i, event in enumerate(data):
        e = event[0]
        if projection == 'zy':
            x = e[:, 2].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        elif projection == 'xy':
            x = e[:, 0].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        images[i] = data
        plt.close()

    print('Saving file...')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.join(save_path, prefix + 'images.h5')

    # Save to HDF5
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('images', data=images)
    h5.create_dataset('max_charge', data=np.array([max_charge]))
    h5.close()


def simulated_labeled(projection, noise, data_dir, save_path, prefix, include_junk):
    print('Processing data...')

    proton_events = pytpc.HDFDataFile(os.path.join(data_dir, prefix + 'proton.h5'), 'r')
    carbon_events = pytpc.HDFDataFile(os.path.join(data_dir, prefix + 'carbon.h5'), 'r')

    # Create empty arrays to hold data
    data = []

    # Add proton events to data array
    for i, event in enumerate(proton_events):
        xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                          baseline_correction=False, cg_times=False)

        if noise:
            # Add artificial noise
            xyzs = dd.add_noise(xyzs).astype('float32')

        data.append([xyzs, 0])

        if i % 50 == 0:
            print('Proton event ' + str(i) + ' added.')

    # Add carbon events to data array
    for i, event in enumerate(carbon_events):
        xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                          baseline_correction=False, cg_times=False)

        if noise:
            # Add artificial noise
            xyzs = dd.add_noise(xyzs).astype('float32')

        data.append([xyzs, 1])

        if i % 50 == 0:
            print('Carbon event ' + str(i) + ' added.')

    # Create junk events
    if include_junk:
        for i in range(len(proton_events)):
            xyzs = np.empty([1, 4])
            if noise:
                xyzs = dd.add_noise(xyzs).astype('float32')
            data.append([xyzs, 2])

            if i % 50 == 0:
                print('Junk event ' + str(i) + ' added.')

    # Take the log of charge data
    log = np.vectorize(_l)

    for event in data:
        event[0][:, 3] = log(event[0][:, 3])

    # Split into train and test sets
    data = shuffle(data)
    partition = int(len(data) * 0.8)
    train = data[:partition]
    test = data[partition:]

    # Normalize
    max_charge = np.array(list(map(lambda x: x[0][:, 3].max(), train))).max()

    for e in train:
        for point in e[0]:
            point[3] = point[3] / max_charge

    for e in test:
        for point in e[0]:
            point[3] = point[3] / max_charge

    # Make train numpy sets
    train_features = np.empty((len(train), 128, 128, 3), dtype=np.uint8)
    train_targets = np.empty((len(train),), dtype=np.uint8)

    for i, event in enumerate(train):
        e = event[0]
        if projection == 'zy':
            x = e[:, 2].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        elif projection == 'xy':
            x = e[:, 0].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        train_features[i] = data
        train_targets[i] = event[1]
        plt.close()

    # Make test numpy sets
    test_features = np.empty((len(test), 128, 128, 3), dtype=np.uint8)
    test_targets = np.empty((len(test),), dtype=np.uint8)

    for i, event in enumerate(test):
        e = event[0]
        if projection == 'zy':
            x = e[:, 2].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        elif projection == 'xy':
            x = e[:, 0].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        test_features[i] = data
        test_targets[i] = event[1]
        plt.close()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = os.path.join(save_path, prefix + 'images.h5')

    # Save to HDF5
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('train_features', data=train_features)
    h5.create_dataset('train_targets', data=train_targets)
    h5.create_dataset('test_features', data=test_features)
    h5.create_dataset('test_targets', data=test_targets)
    h5.create_dataset('max_charge', data=np.array([max_charge]))
    h5.close()


def simulated_unlabeled(projection, noise, data_dir, save_path, prefix, include_junk):
    print('Processing data...')

    proton_events = pytpc.HDFDataFile(os.path.join(data_dir, prefix + 'proton.h5'), 'r')
    carbon_events = pytpc.HDFDataFile(os.path.join(data_dir, prefix + 'carbon.h5'), 'r')

    # Create empty arrays to hold data
    data = []

    # Add proton events to data array
    for i, event in enumerate(proton_events):
        xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                          baseline_correction=False, cg_times=False)

        if noise:
            # Add artificial noise
            xyzs = dd.add_noise(xyzs).astype('float32')

        data.append([xyzs, 0])

        if i % 50 == 0:
            print('Proton event ' + str(i) + ' added.')

    # Add carbon events to data array
    for i, event in enumerate(carbon_events):
        xyzs = event.xyzs(peaks_only=True, drift_vel=5.2, clock=12.5, return_pads=False,
                          baseline_correction=False, cg_times=False)

        if noise:
            # Add artificial noise
            xyzs = dd.add_noise(xyzs).astype('float32')

        data.append([xyzs, 1])

        if i % 50 == 0:
            print('Carbon event ' + str(i) + ' added.')

    # Create junk events
    if include_junk:
        for i in range(len(proton_events)):
            xyzs = np.empty([1, 4])
            if noise:
                xyzs = dd.add_noise(xyzs).astype('float32')
            data.append([xyzs, 2])

            if i % 50 == 0:
                print('Junk event ' + str(i) + ' added.')

    # Take the log of charge data
    log = np.vectorize(_l)

    for event in data:
        event[0][:, 3] = log(event[0][:, 3])

    data = shuffle(data)

    # Normalize
    max_charge = np.array(list(map(lambda x: x[0][:, 3].max(), data))).max()

    for e in data:
        for point in e[0]:
            point[3] = point[3] / max_charge

    print('Making images...')

    # Make numpy sets
    images = np.empty((len(data), 128, 128, 3), dtype=np.uint8)

    for i, event in enumerate(data):
        e = event[0]
        if projection == 'zy':
            x = e[:, 2].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        elif projection == 'xy':
            x = e[:, 0].flatten()
            z = e[:, 1].flatten()
            c = e[:, 3].flatten()
        else:
            raise ValueError('Invalid projection value.')
        fig = plt.figure(figsize=(1, 1), dpi=128)
        if projection == 'zy':
            plt.xlim(0.0, 1250.0)
        elif projection == 'xy':
            plt.xlim(-275.0, 275.0)
        plt.ylim((-275.0, 275.0))
        plt.axis('off')
        plt.scatter(x, z, s=0.6, c=c, cmap='Greys')
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer._renderer, dtype=np.uint8)
        data = np.delete(data, 3, axis=2)
        images[i] = data
        plt.close()

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Saving file...')

    filename = os.path.join(save_path, prefix + 'images.h5')

    # Save to HDF5
    h5 = h5py.File(filename, 'w')
    h5.create_dataset('images', data=images)
    h5.create_dataset('max_charge', data=np.array([max_charge]))
    h5.close()


@click.command()
@click.argument('type', type=click.Choice(['real', 'sim']), nargs=1)
@click.argument('projection', type=click.Choice(['xy', 'zy']), nargs=1)
@click.argument('data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), nargs=1)
@click.option('--save_dir', type=click.Path(exists=False, file_okay=False, dir_okay=True), default='',
              help='Where to save the generated data.')
@click.option('--prefix', type=click.STRING, default='',
              help='Prefix for the saved file names and/or files to read in. By default, there is no prefix.')
@click.option('--labeled', type=click.BOOL, default=True,
              help='If true, only the labeled data will be processed.')
@click.option('--noise', type=click.BOOL, default=True,
              help='Whether or not to add artificial noise to simulated data.')
@click.option('--include_junk', type=click.BOOL, default=True,
              help='Whether or not to include junk events.')
def main(type, projection, data_dir, save_dir, prefix, labeled, noise, include_junk):
    """This script will generate and save images from ATTPC event data to be used for CNN training.

    When using real data, this script will look for runs 0130 and 0210, as these are the runs that have
    been partially hand-labeled.
    """
    if type == 'real':
        if labeled:
            real_labeled(projection, data_dir, save_dir, prefix)
        else:
            real_unlabeled(projection, data_dir, save_dir, prefix)
    else:
        if labeled:
            simulated_labeled(projection, noise, data_dir, save_dir, prefix, include_junk)
        else:
            simulated_unlabeled(projection, noise, data_dir, save_dir, prefix, include_junk)


if __name__ == '__main__':
    main()
