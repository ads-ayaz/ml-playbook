#!/usr/bin/env python

"""
Aluance ML spot-instance-friendly training harness.
Automatically resumes training from where it left off.
"""

import datetime
import getopt
import glob
import logging
import numpy as np
import os
import pandas as pd
import requests
import shutil
import sys
import time
import yaml

import tensorflow as tf
from tensorflow.keras.callbacks import Callback, CSVLogger, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, Activation, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam

import horovod.tensorflow.keras as hvd

# Defaults and keys
AWS_METADATA_SPOT_INSTANCE_ACTION = 'http://169.254.169.254/latest/meta-data/spot/instance-action'

DEFAULT_CONFIG_BATCH_SIZE = 32
DEFAULT_CONFIG_CHECKPOINT_NAMES = 'ads_model.{epoch:03d}.h5'
DEFAULT_CONFIG_CHECKPOINT_PATH = '/ads-ml/training_run/'
DEFAULT_CONFIG_HISTORY_PATH = '/ads-ml/training_history/'
DEFAULT_CONFIG_DATA_PATH = '/ads-ml/data/'
DEFAULT_CONFIG_EPOCHS = 20
DEFAULT_CONFIG_FILE = './config/ads-spot-training-config.yaml'
DEFAULT_CONFIG_FILEBATCH_SIZE = 1
DEFAULT_CONFIG_LEARNING_RATE = 0.03
DEFAULT_CONFIG_LOG_PATH = '/ads-ml/logs/'
DEFAULT_CONFIG_MOUNT_PATH = '/ads-ml/'
DEFAULT_CONFIG_SHUFFLE = True
DEFAULT_CONFIG_SPOT_TERMINATION_SLEEP = 150
DEFAULT_CONFIG_VERBOSITY = 2

DEFAULT_HVD_SIZE = 1

DEFAULT_MODEL_LOSS = 'categorical_crossentropy'
DEFAULT_MODEL_METRICS = ['accuracy']
DEFAULT_MODEL_OPTIMIZER = 'adam'

KEY_ARGS_CONFIG_FILE = 'config-file'
KEY_ARGS_CONFIG_FILE_SHORT = 'c:'

KEY_CONFIG_BATCH_SIZE = 'batch_size'
KEY_CONFIG_CHECKPOINT_NAMES = 'checkpoint_name_format'
KEY_CONFIG_CHECKPOINT_PATH = 'checkpoint_path'
KEY_CONFIG_DATA_PATH = 'data_path'
KEY_CONFIG_EPOCHS = 'epochs'
KEY_CONFIG_FILEBATCH_SIZE = 'filebatch_size'
KEY_CONFIG_LEARNING_RATE = 'learning_rate'
KEY_CONFIG_LOG_PATH = 'tensorboard_logdir'
KEY_CONFIG_MOUNT_PATH = 'mount_dir'
KEY_CONFIG_SHUFFLE = 'shuffle'
KEY_CONFIG_SPOT_TERMINATION_SLEEP = 'spot_termination_sleep_time'
KEY_CONFIG_VERBOSITY = 'verbose'


### HOROVOD | Initialization and pin processes to GPUs

# Initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

### 


def parse_commandline() :
    """Parse any command line arguments that were passed to this process."""

    # Define the command line args we expect
    short_options = KEY_ARGS_CONFIG_FILE_SHORT
    long_options = [KEY_ARGS_CONFIG_FILE]
    
    # Parse command line
    args_list = sys.argv[1:]
    
    try :
        arguments, _ = getopt.getopt(args_list, short_options, long_options)
    except getopt.error as e :
        print(str(e))
        sys.exit(2)


    # Populate the return dictionary with defaults
    ret_dict = {}
    ret_dict[KEY_ARGS_CONFIG_FILE] = DEFAULT_CONFIG_FILE
        
    # Add / overwrite defaults with any passed argument values
    for arg, val in arguments:
        if arg in (KEY_ARGS_CONFIG_FILE_SHORT, KEY_ARGS_CONFIG_FILE) :
            if (os.path.isfile(val)) :
                ret_dict[KEY_ARGS_CONFIG_FILE] = val
            else :
                logging.error('Configuration file %s was not found.' % str(val))
                sys.exit(2)

    return ret_dict
    

def load_configuration(config_file=DEFAULT_CONFIG_FILE) :
    """Load parameters for this training from the YAML configuration file.
    
    Keyword arguments:
    config_file -- filename of the config file.
    """
    
    # Populate return dictionary with default values
    ret_dict = {}
    ret_dict[KEY_CONFIG_BATCH_SIZE] = DEFAULT_CONFIG_BATCH_SIZE
    ret_dict[KEY_CONFIG_EPOCHS] = DEFAULT_CONFIG_EPOCHS
    ret_dict[KEY_CONFIG_CHECKPOINT_NAMES] = DEFAULT_CONFIG_CHECKPOINT_NAMES
    ret_dict[KEY_CONFIG_LOG_PATH] = DEFAULT_CONFIG_LOG_PATH
    ret_dict[KEY_CONFIG_MOUNT_PATH] = DEFAULT_CONFIG_MOUNT_PATH
    ret_dict[KEY_CONFIG_SHUFFLE] = DEFAULT_CONFIG_SHUFFLE
    ret_dict[KEY_CONFIG_SPOT_TERMINATION_SLEEP] = DEFAULT_CONFIG_SPOT_TERMINATION_SLEEP
    ret_dict[KEY_CONFIG_VERBOSITY] = DEFAULT_CONFIG_VERBOSITY
    
    # If there is no config file, return the default values
    if len(config_file) == 0 :
        return ret_dict
    
    # Load yaml configuration file.
    try :
        stream = open(config_file, 'r')
        ret_dict = yaml.safe_load(stream)

    except Exception as e :
        logging.error('Unable to load configuration from %s . Is YAML valid?\n%s'  % (config_file, str(e)))
        sys.exit(2)
        
    finally:
        stream.close()

        
    return ret_dict

    
def load_dataset(data_path=DEFAULT_CONFIG_DATA_PATH, shuffle=DEFAULT_CONFIG_SHUFFLE, batch_size=DEFAULT_CONFIG_BATCH_SIZE) :
    """Returns the X and Y training, validation and test data.

    Keyword arguments:
    data_path -- path to the data folder.
    shuffle -- set to True to shuffle the data.
    batch_size -- size of each batch of data (if generator is returned).
    """
    
    X_train = np.zeros((1, 1))
    Y_train = np.zeros((1, 1))
    X_val = np.zeros((1, 1))
    Y_val = np.zeros((1, 1))
    X_test = np.zeros((1, 1))
    Y_test = np.zeros((1, 1))
    
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    

def create_model(model_params) :
    """Returns a new model.
    
    Keyword arguments:
    model_params -- dictionary of parameters required to instantiate the model.
    """
    
    return None
    

def load_model(checkpoint_path, checkpoint_names) :
    """Loads and returns the model to resume and the starting epoch number."""
    
    checkpoint_file_list = glob.glob(os.path.join(checkpoint_path, '*'))
    
    latest_epoch = max([int(file.split('.')[1]) for file in checkpoint_file_list])
    checkpoint_epoch_path = os.path.join(checkpoint_path,
                                         checkpoint_names.format(epoch=latest_epoch))

    ret_model = tf.keras.models.load_model(checkpoint_epoch_path, compile=False)

    return ret_model, latest_epoch


def get_model(model_params={}, checkpoint_path='', checkpoint_names=DEFAULT_CONFIG_CHECKPOINT_NAMES) :
    """Returns the model and starting epoch. Loads it if a previous training run exists or creates a new instance."""
    
    model = None
    epoch_no = 0
    
    if os.path.isdir(checkpoint_path) and any(glob.glob(os.path.join(checkpoint_path, '*'))):
        model, epoch_no = load_model(checkpoint_path, checkpoint_names)
    else:
        model = create_model(model_params)
        epoch_no = 0

    return model, epoch_no
    

def create_callbacks(scaled_lr=DEFAULT_CONFIG_LEARNING_RATE, checkpoint_path='', checkpoint_names=DEFAULT_CONFIG_CHECKPOINT_NAMES, log_path=DEFAULT_CONFIG_LOG_PATH, csv_path=DEFAULT_CONFIG_HISTORY_PATH, spot_termination_sleep=DEFAULT_CONFIG_SPOT_TERMINATION_SLEEP) :
    """Returns the callbacks list."""
    
    ret_list = []
    now_date = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())
    
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    retlist.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    retlist.append(hvd.callbacks.MetricAverageCallback())

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    retlist.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, initial_lr=scaled_lr, verbose=1))
    

    # Checkpoint callback - add only if we are Horovod worker 0
    if hvd.rank() == 0 :
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        filepath = os.path.join(checkpoint_path, checkpoint_names)
        checkpoint_callback = ModelCheckpoint(
            filepath=filepath, 
            save_weights_only=False, 
            monitor='val_loss', 
            verbose=0, 
            save_best_only=False, 
            mode='min')
        ret_list.append(checkpoint_callback)
        
    # Tensorboard callback
    if not os.path.isdir(log_path) :
        os.makedirs(log_path)
    tensorboard_callback = TensorBoard(log_dir=log_path)
    ret_list.append(tensorboard_callback)

    # CSVLogger callback
    if not os.path.isdir(csv_path) :
        os.makedirs(csv_path)
    csv_logfile = os.path.join(csv_path, ('training_log_%s.csv' % (now_date)))
    csv_callback = CSVLogger(csv_logfile, append=True)
    ret_list.append(csv_callback)

    # Spot termination callback
    class SpotTermination(Callback):
        def on_batch_begin(self, batch, logs={}):
            # Check is action is bening taken to shut down this spot instance
            status_code = requests.get(AWS_METADATA_SPOT_INSTANCE_ACTION).status_code
            if status_code != 404:
                # Sleep to ensure that no further work is done (and avoid writing corrupted files)
                time.sleep(spot_termination_sleep)
    spot_termination_callback = SpotTermination()
    ret_list.append(spot_termination_callback)

    return ret_list
    

def create_metrics() :
    """Returns the metrics list."""
    
    ret_list = DEFAULT_MODEL_METRICS
    
    return ret_list
    

def create_optimizer(learning_rate=DEFAULT_CONFIG_LEARNING_RATE, hvd_size=DEFAULT_HVD_SIZE) :
    """Returns the optimizer."""
    
    # Scale the learning rate based on the horovod size (number of GPUs)
    scaled_learning_rate = learning_rate * hvd_size
    
    # TODO : Create optimizer instance based on scaled learning rate
    opt = None

    # Wrap optimizer with Horovod DistributedOptimizer
    ret_val = hvd.DistributedOptimizer(opt)

    return ret_val, scaled_learning_rate
    

def create_loss() :
    """Returns the loss function."""
    
    ret_fn = DEFAULT_MODEL_LOSS
    
    return ret_fn
    

def save_results(history_path=DEFAULT_CONFIG_HISTORY_PATH, history={}, scores={}) :
    """Save the training results to disk.

    Keyword arguments:
    history_path -- path to directory where history files are saved.
    history -- history dict returned by model.fit().
    scores -- scores dict returned by model.evaluate().
    """

    # Convert the dicts to pandas DataFrames
    df_history = pd.DataFrame(history.history)
    df_scores = pd.DataFrame(scores)

    now_date = '{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now())

    # save to json:  
    history_json_file = os.path.join(history_path, ('history_%s.csv' % (now_date)))
    with open(history_json_file, mode='w') as f:
        df_history.to_json(f)    

    scores_json_file = os.path.join(history_path, ('scores_%s.csv' % (now_date)))
    with open(scores_json_file, mode='w') as f:
        df_scores.to_json(f)
        
    # Backup terminal output
    shutil.copy2(
        '/var/log/cloud-init-output.log', 
        os.path.join(history_path, 'cloud-init-output-%s.log' % (now_date)))


def main() :
    """Main program execution."""
    
    # Parse any command line options
    args_dict = parse_commandline()

    # Load configuration parameters
    config_file = args_dict.get(KEY_ARGS_CONFIG_FILE, DEFAULT_CONFIG_FILE)
    config_dict = load_configuration(config_file=config_file)
    logging.debug("Loaded configuration file: %s" % (config_file))
    
    # Load the dataset
    batch_size = config_dict[KEY_CONFIG_BATCH_SIZE]
    data_path = config_dict[KEY_CONFIG_DATA_PATH]
    shuffle = config_dict[KEY_CONFIG_SHUFFLE]
    logging.debug("Loading dataset with:") 
    logging.debug("\tbatch_size: %i" % (batch_size))
    logging.debug("\tdata_path: %s" % (data_path))
    logging.debug("\tshuffle: %s" % str(shuffle))
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = \
        load_dataset(data_path=data_path, shuffle=shuffle, batch_size=batch_size)
    
    # Get the model and starting epoch
    checkpoint_path = config_dict[KEY_CONFIG_CHECKPOINT_PATH]
    checkpoint_filename_format = config_dict[KEY_CONFIG_CHECKPOINT_NAMES]
    logging.debug("Getting the model with:") 
    logging.debug("\tcheckpoint_path: %s" % (checkpoint_path))
    logging.debug("\tcheckpoint_names: %s" % (checkpoint_filename_format))
    model, epoch_start = get_model(checkpoint_path=checkpoint_path, checkpoint_names=checkpoint_filename_format)
    
    # Set up the callbacks, optimizer, loss function and metrics
    log_path = config_dict[KEY_CONFIG_LOG_PATH]
    optimizer, scaled_lr = create_optimizer()
    callbacks_list = create_callbacks(scaled_lr=scaled_lr, checkpoint_path=checkpoint_path, checkpoint_names=checkpoint_filename_format, log_path=log_path)
    loss_fn = create_loss()
    metrics_list = create_metrics()
    
    # Compile the model and run fit
    # Specify `experimental_run_tf_function=False` to ensure TensorFlow uses hvd.DistributedOptimizer() to compute gradients.
    # -- NOTE that experimental_run_tf_function may have been depricated in TF / Keras 2.2.
    model.compile(
        optimizer = optimizer,
        loss = loss_fn,
        metrics = metrics_list,
        experimental_run_tf_function=False)
    logging.debug("Compiled the model") 
        
    
    # Silence verbosity if this is not Horovod worker 0
    verbosity = config_dict[KEY_CONFIG_VERBOSITY] if hvd.rank() == 0 else 0
    epochs = config_dict[KEY_CONFIG_EPOCHS]
    logging.debug("Training %i epochs starting at %i with verbosity %i." % (epochs, epoch_start, verbosity)) 

    # Check if Y_train is None, indicating that the X and Y training data comes from X_train
    if Y_train is None :
        history = model.fit(
            x = X_train,
            epochs = epochs,
            initial_epoch = epoch_start,
            callbacks = callbacks_list,
            validation_data = X_val,
            verbose = verbosity)

        # Calculate scores for this model using test data
        scores = model.evaluate(
            x = X_test,
            # callbacks = callbacks_list,
            return_dict = True,
            verbose = 0)

    else :
        history = model.fit(
            x = X_train,
            y = Y_train,
            batch_size = batch_size,
            epochs = epochs,
            initial_epoch = epoch_start,
            callbacks = callbacks_list,
            validation_data = (X_val, Y_val),
            shuffle = shuffle,
            verbose = verbosity)
        
        # Calculate scores for this model using test data
        scores = model.evaluate(
            x = X_test,
            y = Y_test,
            batch_size = batch_size,
            # callbacks = callbacks_list,
            return_dict = True,
            verbose = 0)

    # Preserve the history and scores
    save_results(history, scores)
    logging.debug("Saving history and scores.") 
    

## CALL MAIN() WHEN EXEUTED.
if __name__ == "__main__":
    main()