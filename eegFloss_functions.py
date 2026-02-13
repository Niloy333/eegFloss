#%% Author information
"""
eegFloss_v1.0 (GitHub.com/Niloy333/eegFloss)
Created by Niloy Sikder (scholar.google.com/citations?user=0ALk5j4AAAAJ&hl=en)
Affiliations: PhD Candidate, Radboud University Medical Center, Donders Institute for Brain, Cognition and Behaviour, Nijmegen, The Netherlands &
Scientific Assistant, Faculty of Technology and Bionics, Rhine-Waal University of Applied Sciences, Kleve, Germany.
Contact: niloy.sikder@donders.ru.nl, niloy.sikder@hochschule-rhein-waal.de.
Project Supervision: Matthias Krauledat, Paul Zerr, and Martin Dresler.
Copyright (c) 2025 Niloy Sikder
"""
#%% What does this script do?
"""
This script contains all the import and functions required by the scripts of the eegFloss package. Modification to this script is necessary only if specific functions require customization to work differently.
"""
#%% Imports:
import io
import os
import gc
import csv
import sys
import copy
import time
import glob
import json
import pickle
import requests
import pyedflib
import matplotlib
import numpy as np
import pandas as pd
from scipy import signal
from datetime import datetime
import scipy # version: 1.8.1
import tsfel # version: 0.1.4
import matplotlib.pyplot as plt
from colorama import Fore, Style
from joblib import Parallel, delayed
from scipy.signal import spectrogram
from lspopt import spectrogram_lspopt
from scipy.interpolate import interp1d
import lightgbm as lgb # version: 3.3.2
from matplotlib.collections import LineCollection
from scipy.signal import butter, filtfilt, iirnotch
from matplotlib.ticker import FuncFormatter, MultipleLocator
from importlib.metadata import version, PackageNotFoundError
matplotlib.use('Agg')

#%% Global variables:
def declare_globals():
    global Raw_Data_Dir
    global Output_Dir
    global Script_Dir
    global Device_Name
    global EEG_Channels
    global ACC_Channels
    global All_Signals_in_One_File
    global Sleep_Scores_Flname
    global Sleep_Scores_Epoch_Length
    global Check_Usability
    global eegUsability_Model
    global Usability_Epoch_Length
    global Sleep_Scores_Skiprows
    global Aggregate_Scores
    global Unusable_Label
    global Ignore_M_Shaped_Noise
    global Determine_TIB
    global eegMobility_Model
    global Min_Laying_Time_for_TIB
    global Calculate_Sleep_Stats
    global Save_Features
    global Save_Prediction_Mats
    global Usability_Scores_Flname
    global Usability_Graph_Flname
    global Artifact_Rejected_Scores_Flname
    global Lights_Out_On_Flname
    global Artifact_Rejected_Scores_within_TIB_Flname
    global Usability_Scores_within_TIB_Flname
    global Hypnogram_Flname
    global Sleep_Stats_Flname
    global Error_Nights
    global Plot_DPI
    global Error_Txts
    global Mobility_Epoch_Length
    global Add_Index_in_Outputs
    global Feats_flname
    global Pred_Mat_Flname
    global Target_Samp_Rate
    Raw_Data_Dir = os.path.normpath(Raw_Data_Dir)
    Output_Dir = os.path.normpath(Output_Dir)
        
#%% Extra calibration variables:
Feats_flname = 'eegFloss_stat_features.npz'
Pred_Mat_Flname = 'eegFloss_prediction_mats.pkl'    
Error_Nights = pd.DataFrame(columns = ['Night_dir', 'Reason'])
usa_lite_versions = ['lite', 'v0.7', 'lite-weighted-M', 'v0.7.2', 'lite-binary', 'v0.7.3']
Plot_DPI = 150
Figure_Width = 15 #(inches)
Usability_Model_Version = None
Mobility_Epoch_Length = 10
Target_Samp_Rate = 256
n_cores = -2
eegFloss_Link = 'GitHub.com/Niloy333/eegFloss'
line_divider = ['<<<e>>><<<n>>><<<d>>><<<o>>><<<f>>><<<h>>><<<e>>><<<a>>><<<d>>><<<e>>><<<r>>>']
TIB_Info = ["Scores were cropped based on the identified 'Lights Out' and 'Lights On' time points."]
Error_Txts = {
    1: 'One or more EDF files were not found or could not be read. Check the data files.',
    2: 'Recording duration is less than 1 minute.',
    3: "Multiple EDF files were found in the night's directory, which is not allowed when All_Signals_in_One_File = True since this risks the results being overwritten.",
    4: "Sleep scores were not found in the night's directory, could not be read, contain non-numeric values, or were not in the first column of the file. Aggregation is not possible.",
    5: "Sleep scores' length is inconsistent with data. Condition: Sleep_Score_Length = floor(Signal_Length/Sampling_Rate/Sleep_Scores_Epoch_Length).",
    6: 'Sleep scores contain unexpected values. Conditions: 0 <= sleep_stage =< 5 and REM = 4 or 5',
    7: 'The aggregation of sleep and usability scores was unsuccessful. Condition: Usability_Score_Length = Sleep_Score_Length*(Sleep_Scores_Epoch_Length/Usability_Epoch_Length).',
    8: 'Sleep statistics could not be calculated (likely due to insufficient scorable epochs).',
    9: 'TIB could not be determined or TIB = 0 seconds.',
    10: 'Sleep scores were unavailable; sleep statistics could not be calculated.',
    11: 'Sleep statistics could not be calculated due to TIB <= 1 mins.',
    12: 'The provided ACC data may be incompatible. The Zmax ACC training data ranges from -2g to +2g. The normalized ACC data ranges from 0g to 3.46g, centered around 1 g. Please ensure your input data adheres to the expected range and unit (g).',
    13: 'The given channels were not found in the file/folder. Check EEG_Channels and ACC_Channels.'
    }

#%%
# def autocopy_args(fn):
#     def wrapper(*args, **kwargs):
#         args = [copy.deepcopy(arg) if isinstance(arg, (list, dict, set, np.ndarray)) else arg for arg in args]
#         kwargs = {k: copy.deepcopy(v) if isinstance(v, (list, dict, set, np.ndarray)) else v for k, v in kwargs.items()}
#         return fn(*args, **kwargs)
#     return wrapper

#%%
# def warn_on_mutation(fn):
#     def wrapper(*args, **kwargs):
#         args_copy = [copy.deepcopy(a) if isinstance(a, (list, dict, set, np.ndarray)) else a for a in args]
#         kwargs_copy = {k: copy.deepcopy(v) if isinstance(v, (list, dict, set, np.ndarray)) else v for k, v in kwargs.items()}
#         result = fn(*args, **kwargs)
#         for i, (orig, after) in enumerate(zip(args_copy, args)):
#             if isinstance(orig, (list, dict, set, np.ndarray)) and not np.array_equal(orig, after):
#                 warnings.warn(f"Argument #{i} was mutated in-place.")
#         for k in kwargs_copy:
#             if isinstance(kwargs_copy[k], (list, dict, set, np.ndarray)) and not np.array_equal(kwargs_copy[k], kwargs[k]):
#                 warnings.warn(f"Keyword argument '{k}' was mutated in-place.")
#         return result
#     return wrapper

#%%    
def tic():
    return time.time()

#%%
def toc(start_time, m, n, N):
    e_secs = time.time() - start_time
    sec_per_night = e_secs / m
    mins_per_night = int(sec_per_night // 60)
    sec_per_night_2 = int(sec_per_night % 60)
    total_remaining_secs = sec_per_night * (N - n)
    r_days = int(total_remaining_secs // (24 * 3600))
    total_remaining_secs %= (24 * 3600)
    r_hrs = int(total_remaining_secs // 3600)
    total_remaining_secs %= 3600
    r_mins = total_remaining_secs / 60
    print(f"{Fore.CYAN}{Style.BRIGHT}\nMean processing time per night: {mins_per_night} min {sec_per_night_2} sec.{Style.RESET_ALL}")
    
    if r_days > 0:
        print(f"{Fore.CYAN}{Style.BRIGHT}Estimated remaining time: {r_days} days {r_hrs} hrs {r_mins:.0f} mins.{Style.RESET_ALL}")
    elif r_hrs > 0:
        print(f"{Fore.CYAN}{Style.BRIGHT}Estimated remaining time: {r_hrs} hrs {r_mins:.0f} mins.{Style.RESET_ALL}")
    else:
        print(f"{Fore.CYAN}{Style.BRIGHT}Estimated remaining time: {r_mins:.1f} mins.{Style.RESET_ALL}")
        
#%%
def toc2(tic1):
    print(f"Elapsed time: {time.time() - tic1}")
    
#%%
def initial_checks():
    if Aggregate_Scores is True and (Sleep_Scores_Epoch_Length % Usability_Epoch_Length) != 0:
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}Sleep_Scores_Epoch_Length must be divisible by Usability_Epoch_Length when Aggregate_Scores = True.{Style.RESET_ALL}")
    
    if lgb.__version__ != '3.3.2':
        raise ImportError(f"{Fore.RED}{Style.BRIGHT}lightgbm version must be 3.3.2, but found {lgb.__version__}{Style.RESET_ALL}")
    
    try:
        if version("tsfel") != "0.1.4":
            raise ImportError("{Fore.RED}{Style.BRIGHT}TSFEL version must be 0.1.4.{Style.RESET_ALL}")
    except PackageNotFoundError:
        raise ImportError("{Fore.RED}{Style.BRIGHT}TSFEL is not installed.{Style.RESET_ALL}")
    
    if scipy.__version__ != "1.8.1":
        raise ImportError(f"{Fore.RED}{Style.BRIGHT}scipy version must be 1.8.1, but found {scipy.__version__}{Style.RESET_ALL}")
    
    if ACC_Channels is not None:
        if len(ACC_Channels) != 3:
            raise ValueError(f"{Fore.RED}{Style.BRIGHT}ACC_Channels must contain three channels' names representing tri-axial ACC data or be None.{Style.RESET_ALL}")
        if '.edf' in ACC_Channels or '.EDF' in ACC_Channels:
            raise ValueError(f"{Fore.RED}{Style.BRIGHT}Please provide EEG and ACC channels' names without file format (.edf/.EDF).{Style.RESET_ALL}")            
    
    if '.edf' in EEG_Channels or '.EDF' in EEG_Channels:
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}Please provide EEG and ACC channels' names without file format (.edf/.EDF).{Style.RESET_ALL}")
    
    if Sleep_Scores_Flname is not None:
        if not (Sleep_Scores_Flname.endswith('.txt') or Sleep_Scores_Flname.endswith('.csv')) or Sleep_Scores_Flname == 'pseudo_scores':
            raise ValueError(f"{Fore.RED}{Style.BRIGHT}Sleep_Scores_Flname must be in .txt or .csv format.{Style.RESET_ALL}")
    
    if Aggregate_Scores is True:
        if (Unusable_Label > -1 and Unusable_Label < 6) or not isinstance(Unusable_Label, int):
            raise ValueError(f"{Fore.RED}{Style.BRIGHT}Unusable_Label must be an integer and -1⩾Unusable_Label⩾6 to avoid conflicts with sleep scores.{Style.RESET_ALL}")
    
    if Device_Name == 'Zmax' and Ignore_M_Shaped_Noise is True:
        print(f"{Fore.YELLOW}{Style.BRIGHT}Warning: For Zmax data, Ignore_M_Shaped_Noise should be False.{Style.RESET_ALL}")
    
    if Device_Name == 'Zmax' and All_Signals_in_One_File is True:
        print(f"{Fore.YELLOW}{Style.BRIGHT}Warning: For Zmax data, All_Signals_in_One_File should be False.{Style.RESET_ALL}")
    
    if Device_Name != 'Zmax' and Determine_TIB is True:
        print(f"{Fore.YELLOW}{Style.BRIGHT}Warning: The eegMobility model was trained on Zmax ACC data and may give erroneous outputs while checking data from other devices.{Style.RESET_ALL}")
    
    if Device_Name != 'Zmax' and Check_Usability is True:
        print(f"{Fore.YELLOW}{Style.BRIGHT}Please note that the eegUsability model was trained on Zmax (EEG and ACC) data. Please manually check outputs (especially usability graphs) to see if adjustments (such as data normalization) are required. {Style.RESET_ALL}")
    
    if len(EEG_Channels) < 1:
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}No EEG_Channels were given (or incorrect format) Expected: ['channel1_name', 'channel2_name', ...].{Style.RESET_ALL}")
    
    if Output_Dir == Raw_Data_Dir:
        print(f"{Fore.YELLOW}{Style.BRIGHT}eegFloss outputs will be saved alongside the EDF files containing EEG data.{Style.RESET_ALL}")
    
    if not os.path.isabs(Raw_Data_Dir):
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}Please provide absolute paths in Raw_Data_Dir.{Style.RESET_ALL}")
    
    if not os.path.isabs(Output_Dir):
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}Please provide absolute paths in Output_Dir.{Style.RESET_ALL}")
    
    if not os.path.isabs(Script_Dir):
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}Please provide absolute paths in Script_Dir.{Style.RESET_ALL}")
    
    if Determine_TIB is False and Calculate_Sleep_Stats is True:
        print(f"{Fore.YELLOW}{Style.BRIGHT}Since Determine_TIB = False, sleep statistics will not be calculated within the TIB but on the entire night (unless existing {Artifact_Rejected_Scores_within_TIB_Flname} is found).{Style.RESET_ALL}")

#%%
def load_usability_model():
    print(f"{Fore.GREEN}{Style.BRIGHT}Loading eegFloss models...{Style.RESET_ALL}")
    usability_model = None
    
    if Check_Usability is True:
        usability_model_name = determine_usability_model_name()
        try:
            usability_model = load_offline_model(usability_model_name)
        except:
            usability_model = download_pkl(usability_model_name)
        
        if usability_model is None:
            raise ImportError(f"{Fore.RED}{Style.BRIGHT}eegUsability models could not be loaded. Check internet connection.{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}{Style.BRIGHT}eegUsability model was successfully loaded!{Style.RESET_ALL}")
    return usability_model
            
#%%
def load_mobility_model(key):
    mobility_model = None
    
    if Determine_TIB is True and key == 'Zmax':
        mobility_model_name = determine_mobility_model_name()
        try:
            mobility_model = load_offline_model(mobility_model_name)
        except:
            mobility_model = download_pkl(mobility_model_name)
        if mobility_model is None:
            raise ImportError(f"{Fore.RED}{Style.BRIGHT}eegMobility models could not be loaded. Check internet connection.{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}{Style.BRIGHT}eegMobility model was successfully loaded!{Style.RESET_ALL}")
    return mobility_model

#%%
def determine_usability_model_name():
    global Usability_Model_Version
    
    if eegUsability_Model in ['default', 'v1.0']:
        Usability_Model_Version = r'default (v1.0)'
        usability_model_name = 'eegUsability_model_v1.0.pkl'
    elif eegUsability_Model in ['weighted-M', 'v0.8']:
        Usability_Model_Version = r'weighted-M (v0.8)'
        usability_model_name = "eegUsability_model_v0.8_weighted-M.pkl"
    elif eegUsability_Model in ['binary', 'v0.6']:
        Usability_Model_Version = r'binary (v0.6)'
        usability_model_name = "eegUsability_model_v0.6_binary.pkl"
    elif eegUsability_Model in ['lite', 'v0.7']:
        Usability_Model_Version = r'lite, (v0.7)'
        usability_model_name = "eegUsability_model_v0.7_lite.pkl"
    elif eegUsability_Model in ['lite-weighted-M', 'v0.7.2']:
        Usability_Model_Version = r'lite-weighted-M (v0.7.2)'
        usability_model_name = "eegUsability_model_v0.7.2_lite_weighted-M.pkl"
    elif eegUsability_Model in ['lite-binary', 'v0.7.3']:
        Usability_Model_Version = r'lite-binary (v0.7.3)'
        usability_model_name = "eegUsability_model_v0.7.3_lite_binary.pkl"
    elif eegUsability_Model in ['full', 'v0.9']:
        Usability_Model_Version = r'full (v0.9)'
        usability_model_name = "eegUsability_model_v0.9_full.pkl"
    else:
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}Invalid eegUsability_Model (ID/version) was provided!{Style.RESET_ALL}")
    return usability_model_name

#%%
def determine_mobility_model_name():
    if eegMobility_Model == 'default':
        mobility_model_name = "eegMobility_model.pkl"
    elif eegMobility_Model == 'lite':
        mobility_model_name = "eegMobility_lite_model.pkl"
    else:
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}Invalid eegMobility_Model (ID/version) was provided!{Style.RESET_ALL}")
    return mobility_model_name

#%%
def load_offline_model(model_name):
    model_dir = os.path.normpath(os.path.join(Script_Dir, 'eegFloss_models', model_name))
    with open(model_dir, 'rb') as f:
        data = pickle.load(f)
        model1 = data['model']
    return model1

#%%
def download_pkl(model_name):
    models_dict = download_json(r'https://drive.usercontent.google.com/download?id=1f55ko0vH8BUYO9HqAQAVH4iGTu3Xdyf8&export=download&authuser=0')
    model_url = models_dict[model_name]
    try:
        response = requests.get(model_url)
        response.raise_for_status()
        data = pickle.load(io.BytesIO(response.content))
        model1 = data['model']
        return model1
    except:
        return None
    
#%%
def download_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        raise ImportError(f"{Fore.RED}{Style.BRIGHT}eegFloss models could not be loaded. Check internet connection.{Style.RESET_ALL}")
        
#%%
def find_nights(raw_dir):
    all_nights = []
    target_flname = EEG_Channels[0] + '.edf'
    
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        for filename in filenames:
            fname_lower = filename.lower()
            if All_Signals_in_One_File is False:
                if fname_lower == target_flname.lower():
                    all_nights.append(os.path.join(dirpath, filename))
            else:
                if fname_lower.endswith('.edf') or fname_lower.endswith('bdf'):
                    all_nights.append(os.path.join(dirpath, filename))               
    
    all_nights = [os.path.normpath(path) for path in all_nights]
    all_nights = [os.path.dirname(path) for path in all_nights]
    print(f'{Fore.GREEN}{Style.BRIGHT}\nFound {len(all_nights)} nights in\n{Raw_Data_Dir}.{Style.RESET_ALL}')
    return all_nights

#%%
def determine_dest_dir(src_night, num_night, total_nights):
    print(f"{Fore.GREEN}{Style.BRIGHT}\nProcessing night {num_night} of {total_nights}:{Style.RESET_ALL}\nFrom: {src_night}")
    dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
    os.makedirs(dest_night, exist_ok=True)
    return dest_night
#%%
def check_exists(src_night):
    dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
    
    files = {
        'Sleep_Scores': Sleep_Scores_Flname,
        'usability_scores': Usability_Scores_Flname,
        'artifact_rejected_scores': Artifact_Rejected_Scores_Flname,
        'lights_out_on': Lights_Out_On_Flname,
        'artifact_rejected_scores_within_tib': Artifact_Rejected_Scores_within_TIB_Flname,
        'usability_scores_within_tib': Usability_Scores_within_TIB_Flname,
        'features': Feats_flname}
    file_exists = {key: False for key in files}
    
    for key, fname in files.items():
        if fname is not None:
            path_src = os.path.normpath(os.path.join(src_night, fname))
            path_dst = os.path.normpath(os.path.join(dest_night, fname))
            if os.path.exists(path_src):
                file_exists[key] = path_src
            elif os.path.exists(path_dst):
                file_exists[key] = path_dst
            else:
                file_exists[key] = False
            if file_exists[key]:
                print(f"{Fore.GREEN}{Style.BRIGHT}\tExisting {key} were found.{Style.RESET_ALL}")
    return file_exists

#%%
def check_exists_2(src_night, files):
    dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
    file_exists = {key: False for key in files}
    
    for key, fname in files.items():
        if fname is not None:
            path_src = os.path.normpath(os.path.join(src_night, fname))
            path_dst = os.path.normpath(os.path.join(dest_night, fname))
            if os.path.exists(path_src):
                file_exists[key] = path_src
            elif os.path.exists(path_dst):
                file_exists[key] = path_dst
            else:
                file_exists[key] = False
            if file_exists[key]:
                print(f"{Fore.GREEN}{Style.BRIGHT}\tExisting {key} were found.{Style.RESET_ALL}")
    return file_exists

#%%
def current_night(num_of_nights, night, raw_data_dir, output_dir, n):
    print(f"\nProcessing night {n} of {num_of_nights}:\nFrom: {night}")
    night = os.path.dirname(night)
    night_dir = night.replace(raw_data_dir, output_dir)
    return night, night_dir
#%%
def read_data(src_night, sig_type):
    print(f"\tReading {sig_type} data...")
    signals = samp_rates = None
    channels = EEG_Channels if sig_type == 'EEG' else ACC_Channels
    
    if All_Signals_in_One_File is False:
        signals, samp_rates = process_single_edfs(src_night, channels)
    elif All_Signals_in_One_File is True:
        signals, samp_rates = read_combined_edf(src_night, channels)
    else:
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}ValueError: All_Signals_in_One_File must be True or False.{Style.RESET_ALL}")
    
    if samp_rates is not None:
        if samp_rates[channels[0]] != 0:
            dur = len(signals[channels[0]])/samp_rates[channels[0]]
            print(f"\tRecording duration: {round(dur/60, 2)} mins.")
    
    if signals is None:
        log_error(src_night, 1)        
        return None, None, None
    elif signals == -1:
        log_error (src_night, 3)
        return None, None, None
    elif dur < 60:
        log_error(src_night, 2)
        return None, None, None
    else:
        return signals, samp_rates, dur

#%%
def process_single_edfs(night_dir, fls_list):
    def read_channel_file(file_name):
        for ext in [".edf", ".EDF", ".bdf", ".BDF"]:
            file_path = os.path.normpath(os.path.join(night_dir, f"{file_name}{ext}"))
            if os.path.exists(file_path):
                signal, fs = read_single_edf(file_path)
                if signal is not None:
                    return file_name, signal, fs
        return file_name, None, None
    
    results = Parallel(n_jobs=n_cores)(delayed(read_channel_file)(ch) for ch in fls_list)
    all_channels = {}
    samp_rates = {}
    
    for name, signal1, fs in results:
        if signal1 is None:
            return None, None
        all_channels[name] = signal1
        samp_rates[name] = fs
    return all_channels, samp_rates

#%%
def read_single_edf(file_path):
    edf_reader = None
    try:
        edf_reader = pyedflib.EdfReader(file_path)
        signals = [edf_reader.readSignal(i) for i in range(edf_reader.signals_in_file)]
        fs = int(edf_reader.getSampleFrequencies()[0])
        signals = np.concatenate(signals).reshape(-1).astype(np.float32)
        return signals, fs
    except:
        return None, None
    finally:
        if edf_reader is not None:
            edf_reader.close()
            del edf_reader
            gc.collect()

#%%
def read_combined_edf(src_night, channels):
    all_channels = {}
    samp_rates = {}
    file_path1 = glob.glob(os.path.normpath(os.path.join(src_night, "*.edf"))) + \
            glob.glob(os.path.normpath(os.path.join(src_night, "*.EDF"))) + \
            glob.glob(os.path.normpath(os.path.join(src_night, "*.bdf"))) + \
            glob.glob(os.path.normpath(os.path.join(src_night, "*.BDF")))
    file_path1 = sorted(set(file_path1))
    
    if len(file_path1) == 1:
        try:
            psg_edf = pyedflib.EdfReader(file_path1[0])
            label = psg_edf.getSignalLabels()
            if all(item in label for item in channels):
                for i in range(psg_edf.signals_in_file):
                    signal_data = None
                    label = psg_edf.getSignalLabels()[i]
                    if label in channels:
                        signal_data = psg_edf.readSignal(i)
                        signal_data = np.array(signal_data, dtype='float32')
                        fs = int(psg_edf.getSampleFrequencies()[i])
                        all_channels[label] = signal_data
                        samp_rates[label] = fs
                psg_edf.close()
                return all_channels, samp_rates
            else:
                psg_edf.close()
                log_error(src_night, 13)
                return None, None
        except:
            return None, None
        finally:
            gc.collect()
    elif len(file_path1) < 1:
        return None, None
    else:
        return -1, None

#%%
# def adjust_samp_rate(signals, samp_rates, target_rate = None):
#     if target_rate is None:
#         max_rate = max(samp_rates.values())
#     else:
#         max_rate = target_rate
#     adjusted_signals = {}
#     for ch, signal1 in signals.items():
#         current_rate = samp_rates[ch]
#         if current_rate == max_rate:
#             adjusted_signals[ch] = signal1
#         else:
#             print(f"{Fore.YELLOW}{Style.BRIGHT}\tChannels have different sampling rates. Adjusting...{Style.RESET_ALL}")
#             original_time = np.linspace(0, len(signal1) / current_rate, num=len(signal1), endpoint=False)
#             new_length = int(len(signal1) * (max_rate / current_rate))
#             target_time = np.linspace(0, len(signal1) / current_rate, num=new_length, endpoint=False)
#             interpolator = interp1d(original_time, signal1, kind='linear')
#             upsampled_signal = interpolator(target_time).astype(np.float32)
#             adjusted_signals[ch] = upsampled_signal
#     return adjusted_signals, max_rate

#%%
def check_acc_signals(src_night, acc_signals):
    if acc_signals.min() < -2 or acc_signals.min() > 2:
        log_error(src_night, 12)
        return False
    else:
        return True       

#%%
# def adjust_samp_rate(signals, samp_rates, target_rate=None):
#     def _adjust_single_channel(ch, signal1, current_rate, max_rate):
#         if current_rate == max_rate:
#             return ch, signal1
#         else:
#             print(f"{Fore.YELLOW}{Style.BRIGHT}\tChannels have different sampling rates. Adjusting channel '{ch}'...{Style.RESET_ALL}")
#             original_time = np.linspace(0, len(signal1) / current_rate, num=len(signal1), endpoint=False)
#             new_length = int(len(signal1) * (max_rate / current_rate))
#             target_time = np.linspace(0, len(signal1) / current_rate, num=new_length, endpoint=False)
#             interpolator = interp1d(original_time, signal1, kind='linear')
#             upsampled_signal = interpolator(target_time).astype(np.float32)
#             return ch, upsampled_signal
#     max_rate = target_rate if target_rate is not None else max(samp_rates.values())
#     results = Parallel(n_jobs=n_cores)(
#         delayed(_adjust_single_channel)(ch, signal1, samp_rates[ch], max_rate)
#         for ch, signal1 in signals.items())
#     adjusted_signals = dict(results)
#     return adjusted_signals, max_rate

#%%
def adjust_samp_rate(signals, samp_rates, final_target_rate=Target_Samp_Rate):
    def _resample_to_target(ch, signal, current_rate, target_rate):
        if current_rate == target_rate:
            return ch, signal
        print(f"{Fore.YELLOW}{Style.BRIGHT}\tResampling channel '{ch}' from {current_rate} Hz to {target_rate} Hz...{Style.RESET_ALL}")
        original_time = np.linspace(0, len(signal) / current_rate, num=len(signal), endpoint=False)
        new_length = int(len(signal) * (target_rate / current_rate))
        target_time = np.linspace(0, len(signal) / current_rate, num=new_length, endpoint=False)
        interpolator = interp1d(original_time, signal, kind='cubic')
        resampled_signal = interpolator(target_time).astype(np.float32)
        return ch, resampled_signal
    
    final_target_rate = Target_Samp_Rate
    results = Parallel(n_jobs=n_cores)(
        delayed(_resample_to_target)(ch, signal, samp_rates[ch], final_target_rate)
        for ch, signal in signals.items())
    adjusted_signals = dict(results)
    return adjusted_signals, final_target_rate

#%%
def crop_n_make_nparray(eeg_signals, samp_rate):
    channels = list(eeg_signals.keys())
    all_channels = [eeg_signals[key] for key in channels]        
    all_channels = np.vstack(all_channels)
    num_epochs = np.array((all_channels.shape[1] / Usability_Epoch_Length / samp_rate), dtype='int16')
    all_channels = all_channels[:, : num_epochs * Usability_Epoch_Length * samp_rate]    
    return all_channels.astype('float32')

#%%
def euclidean_norm(all_channels):
    eu_norm = np.sqrt(np.sum(all_channels ** 2, axis=0))
    return eu_norm

#%%
def make_samples(signals, samp_rate, key):
    if signals.ndim == 1:
        signals = np.expand_dims(signals, axis=0)
    if key == 'Usability':
        ep_len = Usability_Epoch_Length
    elif key == 'Mobility':
        ep_len = Mobility_Epoch_Length
    
    samples = signals.reshape(signals.shape[0], -1, ep_len * samp_rate).transpose(1, 0, 2)
    return samples

#%%
# def extract_spectrogram_features(all_channels, samp_rate):
#     print
#     num_epochs, num_channels, _ = all_channels.shape
#     spec_feats = np.zeros((num_epochs, num_channels, 11, 129), dtype = 'float32')
#     for epoch in range(num_epochs):
#         for channel in range(num_channels):
#             _, _, Sxx = spectrogram(all_channels[epoch, channel], samp_rate)
#             spec_feats[epoch, channel, :, :] = Sxx.T
#     return spec_feats.astype(np.float32)

#%%
def extract_spectrogram_features(all_channels, samp_rate, n_cores=-1):
    def _resample_channel(e, c, signal, P, target_points, kind):
        x_old = np.arange(P)
        x_new = np.linspace(0, P - 1, target_points)
        f = interp1d(x_old, signal, kind=kind)
        return e, c, f(x_new)
    
    def _resample(channels, target_points, kind='cubic'):
        E, C, P = channels.shape
        if P == target_points:
            return channels, 1.0
        results = Parallel(n_jobs=n_cores)(
            delayed(_resample_channel)(e, c, channels[e, c], P, target_points, kind)
            for e in range(E)
            for c in range(C))
        out = np.empty((E, C, target_points), dtype=channels.dtype)
        for e, c, res in results:
            out[e, c] = res
        return out, (target_points / P)
    
    def _compute_spectrogram(epoch_idx, channel_idx, signal1, fs):
        _, _, Sxx = spectrogram(signal1, fs)
        return epoch_idx, channel_idx, Sxx.T
    
    num_epochs, num_channels, num_points = all_channels.shape
    target_points = 10 * 256  # 2560
    all_channels, rate_factor = _resample(all_channels, target_points, kind='cubic')
    samp_rate *= rate_factor
    _, _, Sxx_sample = spectrogram(all_channels[0, 0], samp_rate)
    time_bins, freq_bins = Sxx_sample.T.shape
    spec_feats = np.zeros((num_epochs, num_channels, time_bins, freq_bins),
                          dtype='float32')
    results = Parallel(n_jobs=n_cores)(
        delayed(_compute_spectrogram)(e, c, all_channels[e, c], samp_rate)
        for e in range(num_epochs)
        for c in range(num_channels))
    
    for e, c, Sxx_T in results:
        spec_feats[e, c] = Sxx_T
    return spec_feats

#%%
def get_stat_feats(file_exists, samples_data, stat_feats_key, samp_rate, key):
    if np.isnan(stat_feats_key).all() or not check_feats_ok(stat_feats_key.shape, samples_data.shape, key):
        if file_exists['features']:
            # Reading existing features (if available):
            stat_feats = np.load(file_exists['features'], allow_pickle=True)
            stat_feats_key = stat_feats[key]       
            if np.isnan(stat_feats_key).all() or not check_feats_ok(stat_feats_key.shape, samples_data.shape, key):
                print(f"{Fore.YELLOW}{Style.BRIGHT}\tExisting features set is incompatible with data.{Style.RESET_ALL}")
                stat_feats_key = extract_tsfel_features(samples_data, samp_rate, key)
        else:
            stat_feats_key = extract_tsfel_features(samples_data, samp_rate, key)
    return stat_feats_key    

#%%
def tsfel_per_channel(epoch_data, samp_rate):
    cgf_statistical = tsfel.get_features_by_domain("statistical")
    cgf_temporal = tsfel.get_features_by_domain("temporal")
    cgf_spectral = tsfel.get_features_by_domain("spectral")    
    temp_feat_1 = tsfel.time_series_features_extractor(cgf_statistical, epoch_data, fs=samp_rate, verbose=0)
    temp_feat_2 = tsfel.time_series_features_extractor(cgf_temporal, epoch_data, fs=samp_rate, verbose=0)
    temp_feat_3 = tsfel.time_series_features_extractor(cgf_spectral, epoch_data, fs=samp_rate, verbose=0)        
    temp_feat_1 = np.array(temp_feat_1)
    temp_feat_2 = np.array(temp_feat_2)
    temp_feat_3 = np.array(temp_feat_3)        
    temp_feat = np.concatenate((temp_feat_1, temp_feat_2, temp_feat_3), axis=1)
    return temp_feat.astype(np.float32)

#%%
def extract_tsfel_features(all_channels, samp_rate, key):
    print(f"\tExtracting features from {key} data...")
    num_epochs, num_channels, _ = all_channels.shape
    num_features = 36 + 18 + 336
    stat_feats = np.zeros((num_epochs, num_channels, num_features), dtype='float32')
    # results = Parallel(n_jobs=-1)(delayed(extract_features_for_channel)(all_channels[epoch, channel, :]) for epoch in range(num_epochs) for channel in range(num_channels))
    tasks = []
    
    for epoch in range(num_epochs):
        for channel in range(num_channels):
            epoch_channel_data = all_channels[epoch, channel, :]
            task = delayed(tsfel_per_channel)(epoch_channel_data, samp_rate)
            tasks.append(task)
    
    results = Parallel(n_jobs=n_cores)(tasks)    
    index = 0
    
    for epoch in range(num_epochs):
        for channel in range(num_channels):
            if results[index] is not None:
                stat_feats[epoch, channel, :] = results[index]
            index += 1
    return stat_feats.astype(np.float32)

#%%
# def load_stat_feats(path1):
#     dest_night = dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
#     file_path1 = os.path.normpath(os.path.join(src_night, Feats_flname))
#     file_path2 = os.path.normpath(os.path.join(dest_night, Feats_flname))
#     if os.path.exists(file_path1):
#         stat_feats = np.load(file_path1, allow_pickle=True)
#     else:
#         stat_feats = np.load(file_path2, allow_pickle=True)
#     return stat_feats

#%%
def save_extracted_features(dest_night, stat_feats_eeg, stat_feats_acc_agg, stat_feats_acc):
    if stat_feats_eeg is np.nan and stat_feats_acc_agg is np.nan and stat_feats_acc is np.nan:
        return
    else:
        file_path = os.path.normpath(os.path.join(dest_night, Feats_flname))
        if ACC_Channels is None:
            stat_feats_acc_agg = stat_feats_acc = np.nan
        mats = [stat_feats_eeg, stat_feats_acc_agg, stat_feats_acc]
        keys = ['EEG', 'ACCagg', 'ACC']   
        np.savez(file_path, **{keys[i]: mats[i] for i in range(len(keys))})   

#%%
# alternative: np.array_equal, np.allclose
def all_values_equal(mat1, mat2):
    comparison = (mat1 == mat2)
    any_mismatch = np.any(~comparison)
    return ~any_mismatch

#%%    
def create_samples_usability(spec_feats_eeg, stat_feats_eeg, spec_feats_acc_agg, stat_feats_acc_agg):
    print("\tPreparing samples for usability detection...")
    spec_feats_eeg = spec_feats_eeg.reshape(spec_feats_eeg.shape[0], spec_feats_eeg.shape[1], -1)    
    spec_feats_acc_agg = spec_feats_acc_agg.reshape(spec_feats_acc_agg.shape[0], spec_feats_acc_agg.shape[1], -1)
    channels = stat_feats_eeg.shape[1]
    usa_samples = []    
    
    for ch in range(channels):
        combined_features = np.concatenate((spec_feats_eeg[:, ch, :], 
                                            spec_feats_acc_agg[:, 0, :], 
                                            stat_feats_eeg[:, ch, :], 
                                            stat_feats_acc_agg[:, 0, :]), axis=1)
        usa_samples.append(combined_features)
    usa_samples = np.stack(usa_samples, axis=0)
    return usa_samples

#%%
def create_samples_usability_lite(spec_feats_eeg, spec_feats_acc_agg):
    print("\tPreparing samples for usability detection...")
    spec_feats_eeg = spec_feats_eeg.reshape(spec_feats_eeg.shape[0], spec_feats_eeg.shape[1], -1)    
    spec_feats_acc_agg = spec_feats_acc_agg.reshape(spec_feats_acc_agg.shape[0], spec_feats_acc_agg.shape[1], -1)
    channels = spec_feats_eeg.shape[1]
    usa_samples = []
    
    for ch in range(channels):
        combined_features = np.concatenate((spec_feats_eeg[:, ch, :], 
                                            spec_feats_acc_agg[:, 0, :]), axis=1)
        usa_samples.append(combined_features)
    usa_samples = np.stack(usa_samples, axis=0)
    return usa_samples    

#%%
# def create_samples_usability_zmax(spec_feats_eeg, stat_feats_eeg, spec_feats_acc_agg, stat_feats_acc_agg):
    # spec_feats = np.concatenate((spec_feats_eeg, spec_feats_acc_agg), axis=1)
    # stat_feats_usability = np.concatenate((stat_feats_eeg, stat_feats_acc_agg), axis=1)
    # spec_feats_l = spec_feats[:, [0, 2], :, :] # EEG_L and Acc_Agg
    # spec_feats_r = spec_feats[:, [1, 2], :, :] # EEG_R and Acc_Agg
    # spec_feats_l = np.hstack((spec_feats_l[:, 0, :, :], spec_feats_l[:, 1, :, :])) # Stacking EEG_L and Acc_Agg features side-by-side
    # temp = [spec_feats_l[:, i, :] for i in range(spec_feats_l.shape[1])] # Flattening features
    # spec_feats_l = np.hstack(temp) # Stacking flattened features
    # spec_feats_r = np.hstack((spec_feats_r[:, 0, :, :], spec_feats_r[:, 1, :, :]))
    # temp = [spec_feats_r[:, i, :] for i in range(spec_feats_r.shape[1])]
    # spec_feats_r = np.hstack(temp)
    # # Preparing Satistical features:
    # stat_feats_usability_l = stat_feats_usability[:, [0, 2], :] # EEG_L and Acc_Agg
    # stat_feats_usability_r = stat_feats_usability[:, [1, 2], :] # EEG_R and Acc_Agg
    # stat_feats_usability_l = np.hstack((stat_feats_usability_l[:, 0, :], stat_feats_usability_l[:, 1, :])) # Stacking EEG_L and Acc_Agg features side-by-side
    # stat_feats_usability_r = np.hstack((stat_feats_usability_r[:, 0, :], stat_feats_usability_r[:, 1, :]))
    # # Stacking Sectrogram and Satistical features side-by-side
    # x_test_l = np.hstack((spec_feats_l, stat_feats_usability_l))
    # x_test_r = np.hstack((spec_feats_r, stat_feats_usability_r))
    #return x_test_l, x_test_r
    
#%%
def check_model_compatibility(featuremap, model_features, model_type):
    if featuremap[2] != model_features:
        raise ValueError(f"{Fore.RED}{Style.BRIGHT}The featuremap generated from data does not match with the {model_type} model's expected featuremap.\n\tData(samples, features): ({featuremap[1]}, {featuremap[2]})\n\t{model_type} Model expected: ({featuremap[1]}, {model_features})\nExecution can not contunue. Check {model_type} model's source and version. {Style.RESET_ALL}")

#%%
def pseudo_acc_feats(signal_len, samp_rate):
    acc_agg = np.ones(signal_len, dtype='int8')
    acc_agg_samples = acc_agg.reshape(1, -1, Usability_Epoch_Length * samp_rate).transpose(1, 0, 2)
    spec_acc = extract_spectrogram_features(acc_agg_samples, samp_rate)    
    stat_feats_1 = np.load(os.path.normpath(os.path.join(Script_Dir, 'no_acc.npy')))
    stat_acc = np.tile(stat_feats_1, (acc_agg_samples.shape[0], 1, 1))
    return spec_acc, stat_acc, samp_rate

#%%
def check_feats_ok(feats_shape, data_shape, key):
    if key == 'ACCagg':
        compatible = (feats_shape[0] == data_shape[0] and feats_shape[1] == 1)
    else:
        compatible = (feats_shape[0] == data_shape[0] and feats_shape[1] == data_shape[1])
    return compatible

#%%  
def predict_usability(src_night, usa_samples, usability_model, samp_rate, key = None):
    check_model_compatibility(usa_samples.shape, usability_model.num_feature(), 'Usability')
    print("\tChecking data usability...")
    channels = usa_samples.shape[0]
    usa_pred_mat = []
    usa_pred_class = []
    
    for ch in range(channels):
        pred_mat = usability_model.predict(usa_samples[ch, :, :])
        pred_class = np.argmax(pred_mat, axis=1)
        pred_mat = np.array(pred_mat * 100).astype(np.float16)
        pred_class = np.array(pred_class).astype(np.int8)
        usa_pred_mat.append(pred_mat)
        usa_pred_class.append(pred_class)
    usability_scores = np.stack(usa_pred_class, axis=1).astype(np.int8)
    
    if Ignore_M_Shaped_Noise is True:
        usability_scores[usability_scores == 4] = 0    
    dur = usability_scores.shape[0] * Usability_Epoch_Length / 60
    data_source = [f"Data source: '{src_night}'", f"EEG signals' sampling rate: {samp_rate} Hz and (checked) duration: {round(dur, 2)} minutes."]
    usability_info = [f"EEG data usability was assessed by eegFloss v1.0 (GitHub.com/Niloy333/eegFloss)", f"using eegUsability model: '{Usability_Model_Version}' in {Usability_Epoch_Length}-second epochs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.", f"Class labels: 0 = Good Data; 1 = No Data; 2 = High Noise; 3 = Spiky Noise; 4 = M-shaped Noise."]
    pred_mat_data = {
        'usability_info': data_source + usability_info + [f"This file contains the prediction matrices of classifications of {usability_scores.shape[0]} samples and {EEG_Channels} channels"],
        'EEG_Channels': EEG_Channels,
        'usability_pred_mats': dict(zip(EEG_Channels, usa_pred_mat))}
    
    if Usability_Scores_Flname is not None:
        save_usability_scores(src_night, usability_scores, samp_rate, key)
    usable_percent = (usability_scores == 0).sum() / usability_scores.size * 100
    
    if usable_percent < 80:
       print(f"{Fore.YELLOW}{Style.BRIGHT}\tDetected usable data: {round(usable_percent, 2)}%.{Style.RESET_ALL}")
    elif usable_percent < 50:  
      print(f"{Fore.RED}{Style.BRIGHT}\tDetected usable data: {round(usable_percent, 2)}%.{Style.RESET_ALL}")
    else:  
      print(f"{Fore.GREEN}{Style.BRIGHT}\tDetected usable data: {round(usable_percent, 2)}%.{Style.RESET_ALL}")    
    return usability_scores, pred_mat_data

#%%
def save_usability_scores(src_night, usability_scores, samp_rate, key):
    dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
    
    if key is None:
        file_path = os.path.normpath(os.path.join(dest_night, Usability_Scores_Flname))
    elif key == 'post_filter':
        flname, ext = os.path.splitext(Usability_Scores_Flname)
        file_path = os.path.normpath(os.path.join(dest_night, f"{flname}_after_spiky_noise_filtering{ext}"))
    usability_scores = usability_scores.astype(np.int8)
    
    if Add_Index_in_Outputs == 'timestamp':
        start_end = get_epoch_timestamp(usability_scores.shape[0], samp_rate, Usability_Epoch_Length)
    elif Add_Index_in_Outputs == 'data_index':
        start_end = get_epoch_data_indices(usability_scores.shape[0], samp_rate, Usability_Epoch_Length)
    else:
        start_end = None
    
    dur = usability_scores.shape[0] * Usability_Epoch_Length / 60
    data_source = [f"Data source: '{src_night}'.", f"EEG signals' sampling rate: {samp_rate} Hz and (checked) duration: {round(dur, 2)} minutes."]
    usability_info = [f"EEG data usability was assessed by eegFloss v1.0 (GitHub.com/Niloy333/eegFloss)", f"using eegUsability model: '{Usability_Model_Version}' in {Usability_Epoch_Length}-second epochs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.", f"Class labels: 0 = Good Data; 1 = No Data; 2 = High Noise; 3 = Spiky Noise; 4 = M-shaped Noise."]
    channels = f"[{'; '.join(EEG_Channels)}]"
    
    if key is None:
        file_header = data_source + usability_info + [f"This file contains the usability scores of {usability_scores.shape[0]} samples and {channels} channels."] + line_divider
    elif key == 'post_filter':
        file_header = data_source + usability_info + [f"This file contains the usability scores of {usability_scores.shape[0]} samples and {channels} channels after spiky noise filtering."] + line_divider
    
    write_scores_csv(file_path, file_header, usability_scores, start_end, EEG_Channels, 'Usability')

#%%
def write_scores_csv(file_path, file_header, scores, start_end, channels, key):
    with open(file_path, mode='w', newline='') as f:
        for line in file_header:
            f.write(f"# {line}\n")
        writer = csv.writer(f)
        
        if Add_Index_in_Outputs == 'timestamp':
            writer.writerow(channels + ["Start_time_sec", "End_time_sec"])
        elif Add_Index_in_Outputs == 'data_index':
            writer.writerow(channels + ["Start_data_index", "End_data_index"])
        else:
            writer.writerow(EEG_Channels)
        
        for i in range(scores.shape[0]):
            if Add_Index_in_Outputs in ['timestamp', 'data_index']:
                if key == 'Usability':
                    row = list(scores[i]) + list(start_end[i])
                elif key == 'Sleep':
                    row = list([scores[i]]) + list(start_end[i])
            else:
                if key == 'Usability':
                    row = list(scores[i])
                elif key == 'Sleep':
                    row = list([scores[i]])    
            writer.writerow(row)

#%%
def get_epoch_timestamp(num_epochs, samp_rate, epoch_len):
    dur = num_epochs * epoch_len
    points_per_epoch = int(epoch_len * samp_rate)
    timestamp = np.arange(1/samp_rate, dur + 1/samp_rate, 1/samp_rate)
    epoch_timestamp = timestamp.reshape(num_epochs, points_per_epoch)
    epoch_bounds = np.column_stack((epoch_timestamp[:, 0], epoch_timestamp[:, -1]))
    return epoch_bounds.astype(np.float64)

#%%
def get_epoch_data_indices(num_epochs, samp_rate, epoch_len):
    points_per_epoch = int(epoch_len * samp_rate)
    indices = np.arange(num_epochs * points_per_epoch)
    epoch_indices = indices.reshape(num_epochs, points_per_epoch)
    epoch_bounds = np.column_stack((epoch_indices[:, 0], epoch_indices[:, -1]))
    return epoch_bounds.astype(np.int32)    
    
#%%    
def read_scores(file_path):
    df = pd.read_csv(file_path, comment="#", delimiter=',')
    cols = df.columns
    if 'End' in cols[-1]:
        scores = df.iloc[:, :-2]
        scores = scores.astype(np.int8)
        start_end_info = df.iloc[:, -2:]
    else:
        scores = df.astype(np.int8)
        start_end_info = None
    
    comments = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                comments.append(line.strip("# \n"))    
    return scores, start_end_info, comments

#%%
def read_sleep_scores(src_night, file_path, data_dur, usa_scores_len):
    if Sleep_Scores_Flname == 'pseudo_scores':
        print("\tGenerating pseudo sleep scores...")
        scaling_factor = Sleep_Scores_Epoch_Length / Usability_Epoch_Length
        sleep_scores = np.ones(int(usa_scores_len / scaling_factor)) * 9
        return sleep_scores.astype(np.int8)
    else:
        print("\tReading sleep scores...")
        sleep_scores = None
        try:
            sleep_scores = pd.read_csv(file_path, delimiter = '\t', skiprows = Sleep_Scores_Skiprows, header = None)
            if sleep_scores.shape[1] > 1:
                sleep_scores = sleep_scores.iloc[:, 0]     
            sleep_scores = np.squeeze(sleep_scores.values)
            sleep_scores = np.array(sleep_scores).astype(np.int8)
        except Exception as e:
            error_detail = ' Details: ' + str(e)
            log_error(src_night, 4, error_detail)
            return None
        sleep_scores = check_sleep_scores_validity(sleep_scores, src_night, data_dur, usa_scores_len)
        return sleep_scores
    
#%%
def check_sleep_scores_validity(scores, src_night, data_dur, usa_scores_len):
    if data_dur is not None:
        if len(scores) != int(data_dur / Sleep_Scores_Epoch_Length):
            log_error(src_night, 5)
            return None
    scaling_factor = int(Sleep_Scores_Epoch_Length / Usability_Epoch_Length)
    
    if usa_scores_len is not None:
        if len(scores) != int(usa_scores_len / scaling_factor):
            log_error(src_night, 7)
            return None
    
    if not np.all((scores >= 0) & (scores <= 5)):
        log_error(src_night, 6)
        return None
    
    if (np.any(scores == 5) and np.any(scores == 4)):
        print(f"{Fore.YELLOW}{Style.BRIGHT}\tSleep scores contain labels '4' and '5', both of which usually refer to REM. '5' to be re-labeled to '4'.{Style.RESET_ALL}")
        scores[scores == 5] = 4
    return scores

#%%
# @autocopy_args
def aggregate_scores(dest_night, sleep_scores, usa_scores, samp_rate):
    src_night = os.path.normpath(dest_night.replace(Output_Dir, Raw_Data_Dir))
    
    if samp_rate is None:
        dur, samp_rate = fetch_samp_rate(src_night, 'EEG')
    print("\tAggregating sleep and usability scores...")
    scaling_factor = int(Sleep_Scores_Epoch_Length / Usability_Epoch_Length)
    usa_scores = usa_scores.copy()
    usa_scores[usa_scores != 0] = 1
    agg_usa = np.zeros(usa_scores.shape[0], dtype = np.int8)
    agg_usa[np.mean(usa_scores, axis=1) > 0.5] = 1
    agg_usa_2 = np.zeros(int(len(agg_usa) / scaling_factor), dtype=np.int8)
    
    if len(agg_usa_2) != len(sleep_scores):
        log_error(src_night, 7)
        return None
    for i in range(len(agg_usa_2)):
        segment = agg_usa[i * scaling_factor : (i + 1) * scaling_factor]
        if np.mean(segment) > 0.5:
            agg_usa_2[i] = 1
    
    agg_scores = np.where(agg_usa_2 == 1, Unusable_Label, sleep_scores).astype(np.int8)
    
    if Artifact_Rejected_Scores_Flname is not None:
        save_art_rej_sleep_scores(src_night, agg_scores, samp_rate)
    return agg_scores

#%%
def save_art_rej_sleep_scores(src_night, agg_scores, samp_rate):
    dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
    file_path = os.path.normpath(os.path.join(dest_night, Artifact_Rejected_Scores_Flname))
    
    if Add_Index_in_Outputs == 'timestamp':
        start_end = get_epoch_timestamp(agg_scores.shape[0], samp_rate, Sleep_Scores_Epoch_Length)
    elif Add_Index_in_Outputs == 'data_index':
        start_end = get_epoch_data_indices(agg_scores.shape[0], samp_rate, Sleep_Scores_Epoch_Length)
    else:
        start_end = None
    
    dur = agg_scores.shape[0] * Sleep_Scores_Epoch_Length / 60
    data_source = [f"Data source: '{src_night}'.", f"EEG signals' sampling rate: {samp_rate} Hz and (checked) duration: {round(dur, 2)} minutes."]
    arss_info = [f"EEG data usability was assessed by eegFloss v1.0 (GitHub.com/Niloy333/eegFloss)", f"using eegUsability model: '{Usability_Model_Version}' in {Usability_Epoch_Length}-second epochs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.", f"Sleep scores were imported from '{Sleep_Scores_Flname}'; scoring was performed in {Sleep_Scores_Epoch_Length}-second epochs.", f"This file contains aggregated sleep and usability scores based on the majority rule.", f"Class labels: {Unusable_Label} = Unusable Data; 0-4 = W; N1; N2; N3; REM; 9 = placeholder; (or check '{Sleep_Scores_Flname}')."]
    channels = ['Sleep_scores']  
    file_header = data_source + arss_info + line_divider
    write_scores_csv(file_path, file_header, agg_scores, start_end, channels, 'Sleep')

#%%
# def extract_pow_feats(samples, samp_rate):
#     n_samples, n_channels, _ = samples.shape
#     pow_feats = np.zeros((n_samples, n_channels, 129), dtype=np.float32)
#     for i in range(n_samples):
#         for j in range(n_channels):
#             _, psd = signal.welch(samples[i, j, :], fs = samp_rate)
#             pow_feats[i, j, :] = psd
#     pow_feats_flat = np.hstack([pow_feats[:, ch, :] for ch in range(n_channels)])
#     return pow_feats_flat.astype(np.float32)

#%%
def extract_pow_feats(samples, samp_rate):
    def _compute_psd(sample_idx, channel_idx, data, fs):
        _, psd = signal.welch(data, fs=fs)
        return sample_idx, channel_idx, psd
    n_samples, n_channels, _ = samples.shape
    _, psd_sample = signal.welch(samples[0, 0, :], fs=samp_rate)
    n_freqs = len(psd_sample)
    pow_feats = np.zeros((n_samples, n_channels, n_freqs), dtype=np.float32)
    results = Parallel(n_jobs=n_cores)(
        delayed(_compute_psd)(i, j, samples[i, j], samp_rate)
        for i in range(n_samples)
        for j in range(n_channels))
    
    for i, j, psd in results:
        pow_feats[i, j, :] = psd
    
    pow_feats_flat = pow_feats.reshape(n_samples, -1)
    return pow_feats_flat.astype(np.float32)

#%%
def create_samples_mobility(stat_feats_acc):
    print("\tPreparing samples for mobility detection...")
    flat_feats = np.hstack([stat_feats_acc[:, i, :] for i in range(stat_feats_acc.shape[1])])
    return flat_feats

#%%
def predict_mobility(src_night, mob_samples, mobility_model, samp_rate, pred_mat_data):
    print(f"\tIdentifying degree of mobility...")
    mob_samples = mob_samples[np.newaxis, :, :]
    check_model_compatibility(mob_samples.shape, mobility_model.num_feature(), 'Mobility')
    mob_samples = mob_samples[0]
    mob_pred_mat = mobility_model.predict(mob_samples)
    mob_pred_class = np.argmax(mob_pred_mat, axis=1)        
    mob_pred_mat = np.array(mob_pred_mat * 100).astype(np.float16)
    mob_pred_class = np.array(mob_pred_class).astype(np.int8)
    dur = mob_pred_class.shape[0] * Mobility_Epoch_Length / 60
    mobility_info = [f"Degree of movement was assessed by eegFloss v1.0 (GitHub.com/Niloy333/eegFloss)", f"using eegMobility model: '{eegMobility_Model}' in {Mobility_Epoch_Length}-second epochs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.", f"Class labels: 0 = Idle; 1 = Laying; 2 = Stationary; 3 = Mobile."]
    data_source = [f"Data source: '{src_night}'", f"ACC signals' sampling rate: {samp_rate} Hz and (checked) duration: {round(dur, 2)} minutes."]
    
    if pred_mat_data is None:
        pred_mat_data = {}
    pred_mat_data['mobility_info'] = data_source + mobility_info + [f"This file contains the prediction matrices of classifications of {mob_pred_class.shape[0]} samples from combined {ACC_Channels} channels."]
    pred_mat_data['ACC_Channels'] = f"{ACC_Channels} 'combined for mobility detection'"
    pred_mat_data['mob_pred_mat'] = dict(zip(['ACC'], mob_pred_mat))
    return mob_pred_class, pred_mat_data

#%%
def save_pred_mat(dest_night, pred_mat_data):
    file_path = os.path.normpath(os.path.join(dest_night, Pred_Mat_Flname))
    with open(file_path, "wb") as f:
        pickle.dump(pred_mat_data, f)
    
#%%
def determine_lights_out_on(mob_scores, dest_night, samp_rate):
    print("\tDetecting Lights Out, Lights On, and TIB...")
    lay_th = int(Min_Laying_Time_for_TIB * 60 / Mobility_Epoch_Length)
    lights_out_ep = lights_on_ep = None
    lights_out = {}
    lights_on = {}
    laying_label = 1
    count = 0
    
    for i, v in enumerate(mob_scores):
        count = count + 1 if v == laying_label else 0
        if count == lay_th:
            lights_out_ep = i - lay_th + 1
            break
    
    count = 0
    for i in range(len(mob_scores) - 1, -1, -1):
        count = count + 1 if mob_scores[i] == laying_label else 0
        if count == lay_th:
            lights_on_ep = i + lay_th - 1
            break
    
    if lights_out_ep is None or lights_on_ep is None or lights_out_ep == lights_on_ep:
        src_night = os.path.normpath(dest_night.replace(Output_Dir, Raw_Data_Dir))
        log_error(src_night, 9)
        return None, None   
    
    lights_on_ep = lights_on_ep + 1
    lights_out['ep'] = int(lights_out_ep)
    lights_out['sec'] = int(lights_out['ep'] * Mobility_Epoch_Length)
    lights_out['idx'] = int(lights_out['sec'] * samp_rate)
    lights_on['ep'] = int(lights_on_ep)
    lights_on['sec'] = int(lights_on['ep'] * Mobility_Epoch_Length)
    lights_on['idx'] = int(lights_on['sec'] * samp_rate)    
    
    if Lights_Out_On_Flname is not None:
        save_lights_out_on(dest_night, lights_out, lights_on, samp_rate)
    TIB_min = (lights_on['sec'] - lights_out['sec']) / 60
    print(f"{Fore.GREEN}{Style.BRIGHT}\twhich are ~{int(lights_out['sec']/60)}, ~{int(lights_on['sec']/60)}, and {round(TIB_min, 2)} mins.{Style.RESET_ALL}")
    return int(lights_out_ep), int(lights_on_ep)
        
#%%        
def save_lights_out_on(dest_night, lights_out, lights_on, samp_rate):
    src_night = os.path.normpath(dest_night.replace(Output_Dir, Raw_Data_Dir))
    
    if samp_rate is None:
        dur, samp_rate = fetch_samp_rate(src_night, 'ACC')
    file_path = os.path.normpath(os.path.join(dest_night, Lights_Out_On_Flname))
    data_source = [f"Data source: '{src_night}'.", f"ACC signals' sampling rate: {samp_rate} Hz"]
    mobility_info = [f"Degree of movement was assessed by eegFloss v1.0 (GitHub.com/Niloy333/eegFloss)", f"using eegMobility model: '{eegMobility_Model}' in {Mobility_Epoch_Length}-second epochs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."]
    file_header = data_source + mobility_info + [f"Lights Out and Lights On moments were determined from mobility scores. Min_Laying_Time_for_TIB = {Min_Laying_Time_for_TIB} minutes.", "The provided indexes are suitable for Python-style indexing.","E.g.: TIB_data = data[lights_out_index : lights_on_index]."] + line_divider
    
    with open(file_path, mode='w', newline='') as f:
        for line in file_header:
            f.write(f"# {line}\n")
        writer = csv.writer(f)
        writer.writerow(['Lights_out_epoch', 'Lights_on_epoch', 'Lights_out_second', 'Lights_on_second', 'Lights_out_data_index', 'Lights_on_data_index'])
        row = [lights_out['ep'], lights_on['ep'], lights_out['sec'], lights_on['sec'], lights_out['idx'], lights_on['idx']]   
        writer.writerow(row)

#%%
def save_scores_within_TIB(src_night, file_exists, usability_scores, agg_scores, lights_out_ep, lights_on_ep, samp_rate, dur):
    dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
    
    if samp_rate is None or dur is None:
        dur, samp_rate = fetch_samp_rate(src_night, 'ACC')
    else:
        dur = dur/60
    
    agg_scores_tib = usability_scores_tib = None
    data_source = [f"Data source: '{src_night}'", f"EEG signals' sampling rate: {samp_rate} Hz and (checked) duration: {round(dur, 2)} minutes."]
    arss_info = [f"EEG data usability was assessed by eegFloss v1.0 (GitHub.com/Niloy333/eegFloss)", f"using eegUsability model: '{Usability_Model_Version}' in {Usability_Epoch_Length}-second epochs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.", f"Sleep scores were imported from '{Sleep_Scores_Flname}'; scoring was performed in {Sleep_Scores_Epoch_Length}-second epochs."]
    usability_info = [f"EEG data usability was assessed by eegFloss v1.0 (GitHub.com/Niloy333/eegFloss)", f"using eegUsability model: '{Usability_Model_Version}' in {Usability_Epoch_Length}-second epochs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."]
    mobility_info = [f"Degree of movement was assessed using eegMobility model: '{eegMobility_Model}' in {Mobility_Epoch_Length}-second epochs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."]
    
    if Artifact_Rejected_Scores_within_TIB_Flname is not None:
        if agg_scores is None and file_exists['artifact_rejected_scores']:
            agg_scores_df, _, _ = read_scores(file_exists['artifact_rejected_scores'])
            agg_scores = agg_scores_df.values.astype(np.int8)
        if agg_scores is None:
            print(f"{Fore.RED}{Style.BRIGHT}\tAggregated scores could not be fetched/loaded.{Style.RESET_ALL}")
        else:
            if Add_Index_in_Outputs == 'timestamp':
                start_end = get_epoch_timestamp(agg_scores.shape[0], samp_rate, Sleep_Scores_Epoch_Length)
            elif Add_Index_in_Outputs == 'data_index':
                start_end = get_epoch_data_indices(agg_scores.shape[0], samp_rate, Sleep_Scores_Epoch_Length)
            else:
                start_end = None            
            if (Sleep_Scores_Epoch_Length % Mobility_Epoch_Length) != 0:
                print(f"{Fore.RED}{Style.BRIGHT}\tSleep_Scores_Epoch_Length is incompatible with Mobility_Epoch_Length. Sleep scores can not be cropped.{Style.RESET_ALL}")
            else:            
                scaling_factor = int(Sleep_Scores_Epoch_Length / Mobility_Epoch_Length)
                score_start = int(lights_out_ep / scaling_factor)
                score_end = int((lights_on_ep - 1) / scaling_factor) + 1
                agg_scores_tib = agg_scores[score_start : score_end]
                start_end = start_end[score_start : score_end]         
                file_header = data_source + arss_info + mobility_info + [f"Lights Out and Lights On moments were determined from mobility scores. Min_Laying_Time_for_TIB = {Min_Laying_Time_for_TIB} minutes.", f"This file contains the aggregated (based on the majority rule) sleep and usability scores trimmed at the Lights Out and Lights On moments.", "The provided indexes are suitable for Python-style indexing; e.g.: TIB_data = data[lights_out_index : lights_on_index].", f"Class labels: {Unusable_Label} = Unusable Data; 0-4 = W; N1; N2; N3; REM; 9 = placeholder; (or check '{Sleep_Scores_Flname}')."] + line_divider
                file_path = os.path.normpath(os.path.join(dest_night, Artifact_Rejected_Scores_within_TIB_Flname))
                channels = ['Sleep_scores']
                write_scores_csv(file_path, file_header, agg_scores_tib, start_end, channels, 'Sleep')
    
    if Usability_Scores_within_TIB_Flname is not None:
        if usability_scores is None and file_exists['usability_scores']:
            usability_scores_df, _, _ = read_scores(file_exists['usability_scores'])
            usability_scores = usability_scores_df.values.astype(np.int8)
        if usability_scores is None:
            print(f"{Fore.RED}{Style.BRIGHT}\tUsability scores could not be fetched/loaded.{Style.RESET_ALL}")
        else:
            if Add_Index_in_Outputs == 'timestamp':
                start_end = get_epoch_timestamp(usability_scores.shape[0], samp_rate, Usability_Epoch_Length)
            elif Add_Index_in_Outputs == 'data_index':
                start_end = get_epoch_data_indices(usability_scores.shape[0], samp_rate, Usability_Epoch_Length)
            else:
                start_end = None
            if (Usability_Epoch_Length % Mobility_Epoch_Length) != 0:
                print(f"{Fore.RED}{Style.BRIGHT}\tUsability_Epoch_Length is incompatible with Mobility_Epoch_Length. Sleep scores can not be cropped.{Style.RESET_ALL}")
            else:
                scaling_factor = int(Usability_Epoch_Length / Mobility_Epoch_Length)
                score_start = int(lights_out_ep / scaling_factor)
                score_end = int((lights_on_ep - 1) / scaling_factor) + 1
                usability_scores_tib = usability_scores[score_start : score_end]
                start_end = start_end[score_start : score_end]
                file_header = data_source + usability_info + mobility_info + [f"Lights Out and Lights On moments were determined from mobility scores. Min_Laying_Time_for_TIB = {Min_Laying_Time_for_TIB} minutes.", f"This file contains usability scores cropped at the Lights Out and Lights On moments.", "The provided indexes are suitable for Python-style indexing; e.g.: TIB_data = data[lights_out_index : lights_on_index].", f"Class labels: 0 = Good Data; 1 = No Data; 2 = High Noise; 3 = Spiky Noise; 4 = M-shaped Noise."] + line_divider
                file_path = os.path.normpath(os.path.join(dest_night, Usability_Scores_within_TIB_Flname))                 
                write_scores_csv(file_path, file_header, usability_scores_tib, start_end, EEG_Channels, 'Usability')

#%%
def fetch_samp_rate(src_night, key):
    ch = ACC_Channels[0] if key == 'ACC' else EEG_Channels[0]
    
    if All_Signals_in_One_File is False:
        signal, samp_rate = process_single_edfs(src_night, [ch])
    elif All_Signals_in_One_File is True:
        signal, samp_rate = read_combined_edf(src_night, [ch])
    
    signal = signal[ch]
    samp_rate = samp_rate[ch]
    dur = len(signal)/samp_rate/60
    return dur, samp_rate

#%%
def calculate_n_save_stats(src_night, file_exists, sleep_scores, agg_scores, lights_out_ep, lights_on_ep):
    if Sleep_Scores_Flname != 'pseudo_scores':
        print("\tCalculating sleep stats...")
        dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
        file_path = os.path.normpath(os.path.join(dest_night, Sleep_Stats_Flname))
        best_sleep_scores = None
        header_df = pd.DataFrame()
        header_df['Data_source'] = [src_night]
        
        if agg_scores is not None:
            best_sleep_scores = agg_scores
            header_df['Sleep_score_source'] = [Artifact_Rejected_Scores_Flname]           
        else:
            try:
                agg_scores_df, start_end_info_df, agg_scores_header = read_scores(dest_night, Artifact_Rejected_Scores_Flname)
                best_sleep_scores = agg_scores_df.values.astype(np.int8)
                header_df['Sleep_score_source'] = [Artifact_Rejected_Scores_Flname]   
            except:
                if sleep_scores is not None:
                    best_sleep_scores = sleep_scores
                    header_df['Sleep_score_source'] = [Sleep_Scores_Flname]   
                else:
                    best_sleep_scores = read_sleep_scores(src_night, file_exists['Sleep_Scores'], None, None)
                    header_df['Sleep_score_source'] = [Sleep_Scores_Flname]   
        
        if best_sleep_scores is None:
            log_error(src_night, 10)
            return False
        else:
            stats_df = get_sleep_stats(src_night, best_sleep_scores, lights_out_ep, lights_on_ep)
            if stats_df is not None:
                stats_df['Reference'] = [eegFloss_Link]        
                stats_df = pd.concat([header_df, stats_df], axis=1)        
                stats_df = stats_df.T
                stats_df.to_csv(file_path, header=False)
                return True
        
#%%
# def get_sleep_stats(src_night, sleep_scores, lights_out_ep, lights_on_ep):
#     # sleep_scores = best_sleep_scores
#     epoch_to_min = Sleep_Scores_Epoch_Length / 60
#     stats_df = pd.DataFrame()
#     stats_df['Scores duration minute'] = [len(sleep_scores) * epoch_to_min]
#     if lights_out_ep is None or lights_on_ep is None or lights_out_ep == lights_on_ep:
#         print(f"{Fore.YELLOW}{Style.BRIGHT}\t\twithout TIB...{Style.RESET_ALL}")
#         stats_df['Lights out sec'] = ['Undetected']
#         stats_df['Lights on sec'] = ['Undetected']
#         stats_df['TIB min'] = stats_df['Scores duration minute']
#     else:
#         scaling_factor = int(Sleep_Scores_Epoch_Length / Mobility_Epoch_Length)
#         score_start = int(lights_out_ep / scaling_factor)
#         score_end = int((lights_on_ep - 1) / scaling_factor) + 1
#         sleep_scores = sleep_scores[score_start : score_end]
#         stats_df['Lights out sec'] = [int(lights_out_ep * Mobility_Epoch_Length)]
#         stats_df['Lights on sec'] = [int(lights_on_ep * Mobility_Epoch_Length)]
#         stats_df['TIB min'] = [len(sleep_scores) * epoch_to_min]
        
#     stats_df['Scorable %'] = [round(np.count_nonzero(sleep_scores != -1) / len(sleep_scores) * 100, 2)]
    
#     if stats_df['TIB min'].item() <= 1:
#         log_error(src_night, 11)
#         return None
    
#     sleep_eps = np.sum(sleep_scores <= 0)
#     if sleep_eps == 0:
#         print("\t\t{Fore.YELLOW}{Style.BRIGHT}\t\tno sleep was detected...{Style.RESET_ALL}")
#         stats_df['SPT min'] = [0]
#         stats_df['TST min'] = [0]
#         stats_df['N1 min'] = [0]
#         stats_df['N1 %'] = ['Unidentified']
#         stats_df['N2 min'] = [0]
#         stats_df['N2 %'] = ['Unidentified']
#         stats_df['N3 min'] = [0]
#         stats_df['N3 %'] = ['Unidentified']
#         stats_df['REM min'] = [0]
#         stats_df['REM %'] = ['Unidentified']
#         stats_df['NREM min'] = [0]
#         stats_df['NREM %'] = ['Unidentified']
#         stats_df['WASO min'] = ['Unidentified']
#         stats_df['SOL min'] = ['Unidentified']
#         stats_df['N1 latency min'] = ['Unidentified']
#         stats_df['N2 latency min'] = ['Unidentified']
#         stats_df['N3 latency min'] = ['Unidentified']
#         stats_df['REM latency min'] = ['Unidentified']
#         stats_df['PSW min'] = ['Unidentified']
#         stats_df['SE %'] = [0]
#         stats_df['SME %'] = ['Unidentified']
#         return stats_df
#     else:      
#         first_sleep_ep = np.argmax(sleep_scores > 0)
#         last_sleep_ep = len(sleep_scores) - np.argmax(sleep_scores[::-1] > 0) - 1        
#         stats_df['SPT min'] = [(last_sleep_ep - first_sleep_ep + 1) * epoch_to_min]
#         stats_df['TST min'] = [np.sum(sleep_scores > 0) * epoch_to_min]
#         stats_df['N1 min'] = [np.sum(sleep_scores == 1) * epoch_to_min]
#         stats_df['N1 %'] = [round(stats_df['N1 min'].item() / stats_df['TST min'].item() * 100, 2)]
#         stats_df['N2 min'] = [np.sum(sleep_scores == 2) * epoch_to_min]
#         stats_df['N2 %'] = [round(stats_df['N2 min'].item() / stats_df['TST min'].item() * 100, 2)]
#         stats_df['N3 min'] = [np.sum(sleep_scores == 3) * epoch_to_min]
#         stats_df['N3 %'] = [round(stats_df['N3 min'].item() / stats_df['TST min'].item() * 100, 2)]
#         stats_df['REM min'] = [np.sum(sleep_scores == 4) * epoch_to_min]
#         stats_df['REM %'] = [round(stats_df['REM min'].item() / stats_df['TST min'].item() * 100, 2)]
#         stats_df['NREM min'] = stats_df['N1 min'].item() + stats_df['N2 min'].item() + stats_df['N3 min'].item()
#         stats_df['NREM %'] = [round(stats_df['NREM min'].item() / stats_df['TST min'].item() * 100, 2)]
#         after_sleep_scores = sleep_scores[first_sleep_ep : last_sleep_ep + 1]
#         stats_df['WASO min'] = [np.sum(after_sleep_scores == 0) * epoch_to_min]
#         stats_df['SOL min'] = [first_sleep_ep * epoch_to_min]        
#         stage_masks = {stage: (sleep_scores == stage) for stage in [1, 2, 3, 4]}
#         stats_df['N1 latency min'] = [np.argmax(stage_masks[1]) * epoch_to_min if stage_masks[1].any() else 'Unidentified']
#         stats_df['N2 latency min'] = [np.argmax(stage_masks[2]) * epoch_to_min if stage_masks[2].any() else 'Unidentified']
#         stats_df['N3 latency min'] = [np.argmax(stage_masks[3]) * epoch_to_min if stage_masks[3].any() else 'Unidentified']
#         stats_df['REM latency min'] = [np.argmax(stage_masks[4]) * epoch_to_min if stage_masks[4].any() else 'Unidentified']        
#         stats_df['PSW min'] = [np.argmax(sleep_scores[::-1] > 0) * epoch_to_min]
#         stats_df['SE %'] = [round(stats_df['TST min'].item() / stats_df['TIB min'].item() * 100, 2)]
#         stats_df['SME %'] = [round(stats_df['TST min'].item() / stats_df['SPT min'].item() * 100, 2)]
#         return stats_df
        
#%%        
def get_sleep_stats(src_night, sleep_scores, lights_out_ep, lights_on_ep):
    sleep_scores = sleep_scores.copy()
    
    def stage_latency(stage_val):
        mask = (sleep_scores == stage_val)
        duration = np.argmax(mask) * epoch_to_min if mask.any() else 'Unidentified'
        return duration
    
    def pct(n, d):
        return round(n / d * 100, 2) if d > 0 else 'Unidentified'
    
    epoch_to_min = Sleep_Scores_Epoch_Length / 60
    stats_df = pd.DataFrame()
    stats_df['Scores_duration_min'] = [len(sleep_scores) * epoch_to_min]
    
    if lights_out_ep is None or lights_on_ep is None or lights_out_ep == lights_on_ep:
        print(f"{Fore.YELLOW}{Style.BRIGHT}\t\twithout TIB...{Style.RESET_ALL}")
        stats_df['Lights_out_sec'] = ['Undetected']
        stats_df['Lights_on_sec'] = ['Undetected']
        stats_df['TIB_min'] = stats_df['Scores_duration_min']
    else:
        scaling_factor = int(Sleep_Scores_Epoch_Length / Mobility_Epoch_Length)
        score_start = int(lights_out_ep / scaling_factor)
        score_end = int((lights_on_ep - 1) / scaling_factor) + 1
        sleep_scores = sleep_scores[score_start:score_end]
        stats_df['Lights_out_sec'] = [int(lights_out_ep * Mobility_Epoch_Length)]
        stats_df['Lights_on_sec'] = [int(lights_on_ep * Mobility_Epoch_Length)]
        stats_df['TIB_min'] = [len(sleep_scores) * epoch_to_min]
    stats_df['Scorable_%'] = [pct(np.count_nonzero(sleep_scores != Unusable_Label), len(sleep_scores))]
    
    if stats_df['TIB_min'].item() <= 1:
        log_error(src_night, 11)
        return None
    sleep_eps = np.sum(sleep_scores <= 0)
    
    if sleep_eps == 0:
        print("\t\t{Fore.YELLOW}{Style.BRIGHT}\t\tno sleep was detected...{Style.RESET_ALL}")
        for stage in ['SPT min', 'TST min', 'N1 min', 'N2 min', 'N3 min', 'REM min', 'NREM min']:
            stats_df[stage] = [0]
        for stage in ['N1_%', 'N2_%', 'N3_%', 'REM_%', 'NREM_%', 'WASO_min', 'SOL_min', 'N1_latency_min', 'N2_latency_min', 'N3_latency_min', 'REM_latency_min', 'PSW_min', 'SME_%']:
            stats_df[stage] = ['Unidentified']
        stats_df['SE_%'] = [0]
        return stats_df
    else:
        first_sleep_ep = np.argmax(sleep_scores > 0)
        last_sleep_ep = len(sleep_scores) - np.argmax(sleep_scores[::-1] > 0) - 1
        stats_df['SPT_min'] = [(last_sleep_ep - first_sleep_ep + 1) * epoch_to_min]
        stats_df['TST_min'] = [np.sum(sleep_scores > 0) * epoch_to_min]
        for stage_label, stage_name in zip([1, 2, 3, 4], ['N1', 'N2', 'N3', 'REM']):
            stats_df[f'{stage_name}_min'] = [np.sum(sleep_scores == stage_label) * epoch_to_min]
            stats_df[f'{stage_name}_%'] = [pct(stats_df[f'{stage_name}_min'].item(), stats_df['TST_min'].item())]
        stats_df['NREM_min'] = stats_df['N1_min'].item() + stats_df['N2_min'].item() + stats_df['N3_min'].item()
        stats_df['NREM_%'] = [pct(stats_df['NREM_min'].item(), stats_df['TST_min'].item())]
        after_sleep_scores = sleep_scores[first_sleep_ep:last_sleep_ep + 1]
        stats_df['WASO_min'] = [np.sum(after_sleep_scores == 0) * epoch_to_min]
        stats_df['SOL_min'] = [first_sleep_ep * epoch_to_min]
        for stage_val, stage_name in zip([1, 2, 3, 4], ['N1', 'N2', 'N3', 'REM']):
            stats_df[f'{stage_name}_latency_min'] = [stage_latency(stage_val)]        
        stats_df['PSW_min'] = [np.argmax(sleep_scores[::-1] > 0) * epoch_to_min]
        stats_df['SE_%'] = [pct(stats_df['TST_min'].item(), stats_df['TIB_min'].item())]
        stats_df['SME_%'] = [pct(stats_df['TST_min'].item(), stats_df['SPT_min'].item())]
        return stats_df

#%%
def format_time_axis_signal(ax, time_vector):
    duration_secs = time_vector[-1]    
    
    if duration_secs < 3600:
        ax.xaxis.set_major_locator(MultipleLocator(600))  # every 10 min
        def mm_ss_formatter(x, pos):
            total_seconds = int(x)
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}:{seconds:02d}"
        ax.xaxis.set_major_formatter(FuncFormatter(mm_ss_formatter))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(1800))  # every 30 min
        def hh_mm_formatter(x, pos):
            total_seconds = int(x)
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}:{minutes:02d}"
        ax.xaxis.set_major_formatter(FuncFormatter(hh_mm_formatter))    
    ax.tick_params(axis='x', rotation=0)

#%%
def format_time_from_index(ax, total_epochs, epoch_len):
    total_seconds = total_epochs * epoch_len
    
    if total_seconds < 3600:
        ax.xaxis.set_major_locator(MultipleLocator(600 / epoch_len))  # every 10 min
        def mm_ss_formatter(x, pos):
            total = int(x * epoch_len)
            minutes = total // 60
            seconds = total % 60
            return f"{minutes}:{seconds:02d}"
        ax.xaxis.set_major_formatter(FuncFormatter(mm_ss_formatter))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(1800 / epoch_len))  # every 30 min
        def hh_mm_formatter(x, pos):
            total = int(x * epoch_len)
            hours = total // 3600
            minutes = (total % 3600) // 60
            return f"{hours}:{minutes:02d}"
        ax.xaxis.set_major_formatter(FuncFormatter(hh_mm_formatter))
    ax.tick_params(axis='x', rotation=0)

#%%
# def spectrogram_plot_calc(signals, samp_rate):
#     specs_all = []
#     plot_data = {}
#     num_signals = signals.shape[0]
#     for i in range(num_signals):
#         freqs, times, Spec = spectrogram_lspopt(signals[i, :], samp_rate, nperseg = int(Usability_Epoch_Length * samp_rate), noverlap = 0)
#         specs_all.append(Spec[3:303, :])
#     specs_all = np.array(specs_all)
#     specs_all = (specs_all - specs_all.min()) / (specs_all.max() - specs_all.min()) * 10
#     freqs = freqs[3:303]
#     plot_data['specs'] = specs_all.astype(np.float32)
#     plot_data['freqs'] = freqs.astype(np.float32)
#     plot_data['times'] = times.astype(np.float32)
#     return plot_data
    
#%%
def spectrogram_plot_calc(signals, samp_rate):
    def _compute_spectrogram(signal1):
        freqs, times, Spec = spectrogram_lspopt(
            signal1, samp_rate,
            nperseg=int(10 * samp_rate), noverlap = 0)
        return Spec[3:303, :], freqs[3:303], times
    results = Parallel(n_jobs=n_cores)(
        delayed(_compute_spectrogram)(signals[i, :])
        for i in range(signals.shape[0]))
    specs_list, freqs, times = zip(*results)
    specs_all = np.array(specs_list)
    specs_all = (specs_all - specs_all.min()) / (specs_all.max() - specs_all.min()) * 10
    plot_data = {
        'specs': specs_all.astype(np.float32),
        'freqs': freqs[0].astype(np.float32),
        'times': times[0].astype(np.float32)}
    return plot_data

#%%
def plot_usability_graph(src_night, eeg_signals, eeg_samp_rate, acc_agg, acc_samp_rate, usa_scores, file_exists, key = None):
    usa_scores = usa_scores.copy()
    # if eeg_signals is None:
    #     eeg_signals, eeg_samp_rates, org_dur = read_data(src_night, 'EEG')
    #     eeg_signals, eeg_samp_rate = adjust_samp_rate(eeg_signals, eeg_samp_rates)
    #     eeg_signals = crop_n_make_nparray(eeg_signals, eeg_samp_rate)
    
    # if acc_agg is None:
    #     if ACC_Channels is not None:
    #         acc_signals, acc_samp_rates, _ = read_data(src_night, 'ACC')
    #         acc_signals, acc_samp_rate = adjust_samp_rate(acc_signals, acc_samp_rates, eeg_samp_rate)
    #         acc_signals = crop_n_make_nparray(acc_signals, acc_samp_rate)
    #         acc_agg = euclidean_norm(acc_signals)
    
    # if usa_scores is None:
    #     usa_scores_df, _, _= read_scores(file_exists['usability_scores'])
    #     usa_scores = usa_scores_df.values.astype(np.int8)
    
    # if Usability_Graph_Flname is None or eeg_signals is None or usa_scores is None:
    #     return None
    
    if eeg_signals.shape[0] != usa_scores.shape[1]:
        print(f"{Fore.RED}{Style.BRIGHT}\tData shape incompatible with usability scores.\n\tUsability graph will not be plotted.{Style.RESET_ALL}")
        return None
    print("\tPlotting usability graph...")
    dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
    if key is None:
        file_path = os.path.normpath(os.path.join(dest_night, Usability_Graph_Flname))
    elif key == 'post_filter':
        flname, ext = os.path.splitext(Usability_Graph_Flname)
        file_path = os.path.normpath(os.path.join(dest_night, f"{flname}__after_spiky_noise_filtering{ext}"))
    
    num_signals = eeg_signals.shape[0]

    if acc_agg is not None:
        times_acc = np.linspace(1/acc_samp_rate, len(acc_agg)/acc_samp_rate, len(acc_agg))
    
    plot_data = spectrogram_plot_calc(eeg_signals, eeg_samp_rate)    
    
    num_plot = num_signals * 2 + 1
    plt.ioff()
    # plt.rcParams['font.family'] = 'Times New Roman'
    fig, axs = plt.subplots(num_plot, 1, figsize=(Figure_Width, num_plot * 1.75))
    
    title_base = f"Usability graph showing artifacts detected in each {Device_Name} EEG channel by eegUsability ({Usability_Model_Version}) in {Usability_Epoch_Length}-second epochs (at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\nof recording: '{src_night}'"
    
    if key == 'post_filter':
        title_base += " (after spiky noise filtering)"
    
    if Ignore_M_Shaped_Noise is True:
        title_base += " (the M-shaped noise was ignored)"
        usa_scores[usa_scores == 4] = 0
    
    fig.suptitle(title_base, fontsize=10, fontweight='bold', y=.99 if acc_agg is None else 0.98)
    plt.subplots_adjust(top=0.95)

    # Subplot 1: Aggregated Accelerometer
    if acc_agg is None:
        axs[0].text(0.5, 0.5, 'No accelerometer data', horizontalalignment='center',
                    verticalalignment='center', transform=axs[0].transAxes,
                    fontsize=13, fontweight='bold', color='red')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
    else:
        axs[0].plot(times_acc, acc_agg, color='black', linewidth=0.6)
        axs[0].set_xlim(times_acc.min(), times_acc.max())
        axs[0].set_title('Aggregated acceleration', fontsize=9, fontweight='bold')
        axs[0].set_ylabel('Acceleration (g)', fontsize=9)
        axs[0].grid(True, axis='y', color='gainsboro')
        axs[0].yaxis.set_major_locator(MultipleLocator(1.0))  # Full number major ticks
        axs[0].tick_params(axis='y', labelsize=9)
        axs[0].tick_params(axis='x', labelsize=9)
        format_time_axis_signal(axs[0], times_acc)
        label = 'Elapsed time (hh:mm)' if times_acc.max() >= 3600 else 'Elapsed time (mm:ss)'
        axs[0].set_xlabel(label, fontsize=9)

    # Subplots: Spectrogram + Usability per EEG channel
    total_sec = usa_scores.shape[0] * Usability_Epoch_Length
    label = 'Elapsed time (hh:mm)' if total_sec >= 3600 else 'Elapsed time (mm:ss)'
    for i in range(num_signals):
        ch_name = EEG_Channels[i]

        # Spectrogram
        axs[2*i + 1].pcolormesh(plot_data['times'], plot_data['freqs'],
                    10 * np.log10(np.maximum(plot_data['specs'][i], 1e-10)),
                    shading='gouraud', cmap='seismic')
        axs[2*i + 1].set_ylabel('Frequency (Hz)', fontsize=9)
        if i == 0:
            axs[2*i + 1].set_title(f'Spectrogram of channel {ch_name} [colorbar: (high-power) red>>white>>blue (low-power)]', fontsize=9, fontweight='bold')
        else:
            axs[2*i + 1].set_title(f'Spectrogram of channel {ch_name}', fontsize=9, fontweight='bold')
        axs[2*i + 1].tick_params(axis='y', labelsize=9)
        axs[2*i + 1].tick_params(axis='x', labelsize=9)
        format_time_axis_signal(axs[2*i + 1], plot_data['times'])
        axs[2*i + 1].set_xlabel(label, fontsize=9)

        # Usability scores
        y = usa_scores[:, i].copy()
        x = np.arange(len(y))
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = ['limegreen' if y[j] == 0 and y[j+1] == 0 else 'crimson' for j in range(len(y) - 1)]
        linewidths = [1.0 if y[j] == 0 and y[j+1] == 0 else 0.8 for j in range(len(y) - 1)]
        lc = LineCollection(segments, colors=colors, linewidths=linewidths)
        axs[2*i + 2].add_collection(lc)
        axs[2*i + 2].set_xlim(x[0], x[-1])
        axs[2*i + 2].set_ylim(-0.2, 4.1)  # Adjust based on label range (0–4)
        axs[2*i + 2].grid(True, axis='y', color='gainsboro')
        axs[2*i + 2].set_title(f'Usability/artifact labels for channel {ch_name}', fontsize=9, fontweight='bold')
        axs[2*i + 2].tick_params(axis='y', labelsize=9)
        axs[2*i + 2].tick_params(axis='x', labelsize=9)
        format_time_from_index(axs[2*i + 2], len(y), Usability_Epoch_Length)
        axs[2*i + 2].set_xlabel(label, fontsize=9)
        del y, x, points, segments, colors, linewidths
        if Ignore_M_Shaped_Noise is True:
            axs[2*i + 2].set_yticks(np.linspace(0, 3, 4))
            axs[2*i + 2].set_yticklabels(['Good data', 'No data', 'High Noise', 'Spiky'], fontsize=9)
        elif eegUsability_Model in ['binary', 'v0.6', 'lite-binary', 'v0.7.3']:
            axs[2*i + 2].set_yticks(np.linspace(0, 1, 2))
            axs[2*i + 2].set_yticklabels(['Scorable', 'Unscorable'], fontsize=9)
        else:
            axs[2*i + 2].set_yticks(np.linspace(0, 4, 5))
            axs[2*i + 2].set_yticklabels(['Good data', 'No data', 'High Noise', 'Spiky', 'M-shaped'], fontsize=9)

    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, format='png', dpi=Plot_DPI)
    plt.close(fig)
    plt.close('all')
    gc.collect()
    return plot_data
    
#%%
def plot_hypnogram(src_night, eeg_signals, eeg_samp_rate, acc_agg, acc_samp_rate, agg_scores, lights_out_ep, lights_on_ep, mobility_scores, plot_data):
    dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
    file_path = os.path.normpath(os.path.join(dest_night, Hypnogram_Flname))
    if eeg_signals is None:
        eeg_signals, eeg_samp_rates, org_dur = read_data(src_night, 'EEG')
        if eeg_signals is None: return 
        eeg_signals, eeg_samp_rate = adjust_samp_rate(eeg_signals, eeg_samp_rates)
        eeg_signals = crop_n_make_nparray(eeg_signals, eeg_samp_rate)    
    if agg_scores is None:
        try:
            agg_scores_df, start_end_info_df, agg_scores_header = read_scores(dest_night, Artifact_Rejected_Scores_Flname)
            agg_scores = agg_scores_df.values.astype(np.int8)
        except:
            print(f"{Fore.RED}{Style.BRIGHT}\tAggregated sleep scores could not be read.\n\tHypnogram can not be plotted.{Style.RESET_ALL}")
            return False
    if int(eeg_signals.shape[1]/eeg_samp_rate/Sleep_Scores_Epoch_Length) != len(agg_scores):
        print(f"{Fore.RED}{Style.BRIGHT}\tData incompatible with sleep scores.\n\tHypnogram will not be plotted.{Style.RESET_ALL}")
        return False
    print("\tPlotting hypnogram...")
    n_sigs = eeg_signals.shape[0]    
    if ACC_Channels is not None and acc_agg is None:
        acc_signals, acc_samp_rates, _ = read_data(src_night, 'ACC')
        acc_signals, acc_samp_rate = adjust_samp_rate(acc_signals, acc_samp_rates, eeg_samp_rate)
        acc_signals = crop_n_make_nparray(acc_signals, acc_samp_rate)
        acc_agg = euclidean_norm(acc_signals)

    if acc_agg is not None:
        times_acc = np.linspace(1/acc_samp_rate, len(acc_agg)/acc_samp_rate, len(acc_agg))

    if plot_data is None:        
        plot_data = spectrogram_plot_calc(eeg_signals, eeg_samp_rate)

    num_plot = n_sigs + 2
    if mobility_scores is not None:
        num_plot = num_plot + 1
    
    agg_scores_plot = agg_scores.copy()
    if Unusable_Label != -1:
        agg_scores_plot[agg_scores_plot == Unusable_Label] = -1
    
    plt.ioff()
    # plt.rcParams['font.family'] = 'Times New Roman'
    fig, axs = plt.subplots(num_plot, 1, figsize=(Figure_Width, num_plot * 1.75))

    title_base = f"Hypnogram with (aggregated) data usability, device: {Device_Name}, sleep scores: '{Sleep_Scores_Flname}' in {Sleep_Scores_Epoch_Length}-sec epochs, usability model: eegUsability {Usability_Model_Version} in {Usability_Epoch_Length}-sec epochs\nData source: '{src_night}'"
    
    fig.suptitle(title_base, fontsize=10, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.95)
    
    total_sec = agg_scores.shape[0] * Sleep_Scores_Epoch_Length
    label = 'Elapsed time (hh:mm)' if total_sec >= 3600 else 'Elapsed time (mm:ss)'
    for i in range(n_sigs):
        ch_name = EEG_Channels[i]
        # Spectrogram
        axs[i].pcolormesh(plot_data['times'], plot_data['freqs'],
                    10 * np.log10(np.maximum(plot_data['specs'][i], 1e-10)),
                    shading='gouraud', cmap='seismic')
        axs[i].set_ylabel('Frequency (Hz)', fontsize=9)
        if i == 0:
            axs[i].set_title(f'Spectrogram of channel {ch_name} [colorbar: (high-power) red>>white>>blue (low-power)]', fontsize=9, fontweight='bold')
        else:
            axs[i].set_title(f'Spectrogram of channel {ch_name}', fontsize=9, fontweight='bold')
        axs[i].tick_params(axis='y', labelsize=9)
        axs[i].tick_params(axis='x', labelsize=9)
        format_time_axis_signal(axs[i], plot_data['times'])
        axs[i].set_xlabel(label, fontsize=9)

    # Subplot 1: Aggregated Accelerometer
    if acc_agg is None:
        axs[n_sigs].text(0.5, 0.5, 'No accelerometer data', horizontalalignment='center',
                    verticalalignment='center', transform=axs[n_sigs].transAxes,
                    fontsize=13, fontweight='bold', color='red')
        axs[n_sigs].set_xticks([])
        axs[n_sigs].set_yticks([])
    else:
        axs[n_sigs].plot(times_acc, acc_agg, color='black', linewidth=0.6)
        axs[n_sigs].set_xlim(times_acc.min(), times_acc.max())
        axs[n_sigs].set_title('Aggregated acceleration', fontsize=9, fontweight='bold')
        axs[n_sigs].set_ylabel('Acceleration (g)', fontsize=9)
        axs[n_sigs].grid(True, axis='y', color='gainsboro')
        axs[n_sigs].yaxis.set_major_locator(MultipleLocator(1.0))  # Full number major ticks
        axs[n_sigs].tick_params(axis='y', labelsize=9)
        axs[n_sigs].tick_params(axis='x', labelsize=9)
        format_time_axis_signal(axs[n_sigs], times_acc)
        labela = 'Elapsed time (hh:mm)' if times_acc.max() >= 3600 else 'Elapsed time (mm:ss)'
        axs[n_sigs].set_xlabel(labela, fontsize=9)
    
    # Agg_scores
    y = agg_scores.copy()
    x = np.arange(len(y))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = ['crimson' if y[j] == -1 and y[j+1] == -1 else 'mediumblue' for j in range(len(y) - 1)]
    linewidths = [1 if y[j] == -1 and y[j+1] == -1 else 0.75 for j in range(len(y) - 1)]
    lc = LineCollection(segments, colors=colors, linewidths=linewidths)
    axs[n_sigs + 1].add_collection(lc)
    axs[n_sigs + 1].set_xlim(x[0], x[-1])
    axs[n_sigs + 1].set_ylim(-1.2, 4.2)
    axs[n_sigs + 1].grid(True, axis='y', color='gainsboro')
    axs[n_sigs + 1].set_title(f'Hypnogram (with aggregated data usability)', fontsize=9, fontweight='bold')
    axs[n_sigs + 1].tick_params(axis='y', labelsize=9)
    axs[n_sigs + 1].tick_params(axis='x', labelsize=9)
    format_time_from_index(axs[n_sigs + 1], len(agg_scores), Sleep_Scores_Epoch_Length)
    axs[n_sigs + 1].set_xlabel(label, fontsize=9)
    axs[n_sigs + 1].set_yticks(np.linspace(-1, 4, 6))
    axs[n_sigs + 1].set_yticklabels(['Unscorable', 'Wake', 'N1', 'Light/N2', 'SWS/Deep/N3', 'REM'], fontsize=9)
    
    if mobility_scores is not None:
        axs[n_sigs + 2].plot(mobility_scores, linewidth=0.8, color='teal')
        axs[n_sigs + 2].set_xlim(0, len(mobility_scores))
        axs[n_sigs + 2].set_ylim(-0.2, 4)
        axs[n_sigs + 2].grid(True, axis='y', color='gainsboro')
        axs[n_sigs + 2].set_title(f'Degree of Mobility (the purple line indicates the identified time-in-bed)', fontsize=9, fontweight='bold')    
        axs[n_sigs + 2].tick_params(axis='y', labelsize=9)
        axs[n_sigs + 2].tick_params(axis='x', labelsize=9)
        format_time_from_index(axs[n_sigs + 2], len(mobility_scores), Mobility_Epoch_Length)
        axs[n_sigs + 2].set_xlabel(label, fontsize=9)
        axs[n_sigs + 2].set_yticks(np.linspace(0, 3, 4))
        axs[n_sigs + 2].set_yticklabels(['Idle', 'Lying', 'Stationary', 'Mobile'], fontsize=9)
        if lights_out_ep is not None and lights_out_ep != lights_on_ep:
            lights_out_ep, lights_on_ep = lights_out_ep + 2, lights_on_ep - 2
            axs[n_sigs + 2].plot(range(lights_out_ep, lights_on_ep), [3.5] * (lights_on_ep - lights_out_ep), color='purple', linewidth=1.25, linestyle='-')                

    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.savefig(file_path, format='png', dpi=Plot_DPI)
    plt.close(fig)
    plt.close('all')
    gc.collect()
    return True

#%%
def spiky_noise_filter(segments, fs, **filter_kwargs):
    print("\tFiltering epochs with Spiky Noise...")
    def _butter_notch_filter(signal1, fs, low_cutoff=30, filter_order=4, noise_freqs=[8, 16, 24], bandwidth=2):
        nyq = fs / 2
        b, a = butter(filter_order, low_cutoff / nyq, btype='low')
        filtered_signal = filtfilt(b, a, signal1)
        for f in noise_freqs:
            wo = f / nyq
            bw = bandwidth / nyq
            b_notch, a_notch = iirnotch(wo, bw)
            filtered_signal = filtfilt(b_notch, a_notch, filtered_signal)
        return filtered_signal
    filtered_segments = Parallel(n_jobs=n_cores)(
        delayed(_butter_notch_filter)(segment, fs, **filter_kwargs) for segment in segments)
    return np.vstack(filtered_segments)

#%%
# matplotlib.use('QtAgg')
# a = spiky_data[200, :]
# b = filtered_data[200, :]
# %matplotlib qt     
# plt.plot(a)
# plt.show()
# plt.plot(b)
# plt.show()

#%%
def get_spiky_epochs(usa_scores, samples_eeg):
    epochs, channels = usa_scores.shape
    spiky_indices = []
    print("\tFound Spiky Noise in")
    
    for ch in range(channels):
        spike_rows = np.where(usa_scores[:, ch] == 3)[0]
        for epoch in spike_rows:
            spiky_indices.append([ch, epoch])
        print(f"\t\t{len(spike_rows)} epochs of {EEG_Channels[ch]}")
    
    spiky_ep_indx = np.array(spiky_indices, dtype=int)    
    epochs, dtpoints = spiky_ep_indx.shape[0], samples_eeg.shape[2]
    spiky_data = np.zeros((epochs, dtpoints))    
   
    for e in range(epochs):
        ch, ep = spiky_ep_indx[e, :]
        spiky_data[e, :] = samples_eeg[ep, ch, :]
    return spiky_ep_indx, spiky_data 

#%%
def save_filtered_signals(src_night, samples_eeg, spiky_ep_indx, filtered_data):
    samples_eeg_filtered = np.copy(samples_eeg)
    channels = samples_eeg.shape[1]
    
    for i in range(spiky_ep_indx.shape[0]):
        ch, ep = spiky_ep_indx[i]
        samples_eeg_filtered[ep, ch, :] = filtered_data[i, :]
    eeg_signals_filtered = samples_eeg_filtered.transpose(1, 0, 2).reshape(channels, -1)
    
    if All_Signals_in_One_File is True:
        save_single_edf(src_night, eeg_signals_filtered)
    else:
        save_multiple_edfs(src_night, eeg_signals_filtered)
    return eeg_signals_filtered

#%%
def save_single_edf(src_night, eeg_signals):
    edf_file = glob.glob(os.path.join(src_night, '*.edf')) + glob.glob(os.path.join(src_night, '*.EDF'))
    edf_file = list(dict.fromkeys(edf_file))
    filename, ext = os.path.splitext(os.path.basename(edf_file[0]))
    dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
    dest_night = os.path.normpath(os.path.join(dest_night, 'spiky_noise_filtered_signals'))
    os.makedirs(dest_night, exist_ok=True)
    dest_path = os.path.normpath(os.path.join(dest_night, f"{filename}_spiky_noise_filtered{ext}"))
    print(f"\tSaving filtered signals to: {dest_path}")
    edf_reader = pyedflib.EdfReader(edf_file[0])
    channel_labels = edf_reader.getSignalLabels()
    signal_headers = edf_reader.getSignalHeaders()
    file_header = edf_reader.getHeader()
    n_channels = edf_reader.signals_in_file
    updated_signals = []
    
    for i, label in enumerate(channel_labels):
        src_signal = edf_reader.readSignal(i)
        updated_signal = np.array(src_signal)
        if label in EEG_Channels:
            new_signal = np.array(eeg_signals[EEG_Channels.index(label), :]).astype(src_signal.dtype)
            # if len(new_signal) > len(updated_signal):
            #     print(f"Warning: Truncating {label} new signal from {len(new_signal)} to {len(updated_signal)}")
            # min_len = min(len(updated_signal), len(new_signal))
            # updated_signal[0:min_len] = new_signal[0:min_len]
            updated_signal[0:len(new_signal)] = new_signal
        signal_headers[i]['n_samples'] = len(updated_signal)
        updated_signals.append(updated_signal)
        del updated_signal
    
    edf_reader.close()
    writer = pyedflib.EdfWriter(dest_path, n_channels=n_channels, file_type=pyedflib.FILETYPE_EDF)
    writer.setHeader(file_header)
    for i in range(n_channels):
        writer.setSignalHeader(i, signal_headers[i])
    writer.writeSamples(updated_signals)
    writer.close()

#%%
def save_multiple_edfs(src_night, eeg_signals):
    dest_night = os.path.normpath(src_night.replace(Raw_Data_Dir, Output_Dir))
    dest_night = os.path.normpath(os.path.join(dest_night, 'spiky_noise_filtered_signals'))
    os.makedirs(dest_night, exist_ok=True)
    print(f"\tSaving filtered signals to: {dest_night}")
    
    for i, ch in enumerate(EEG_Channels):
        for ext in [".edf", ".EDF"]:
            file_path = os.path.normpath(os.path.join(src_night, f"{ch}{ext}"))
            if os.path.exists(file_path):
                edf_reader = pyedflib.EdfReader(file_path)
                src_signal = edf_reader.readSignal(0)
                signal_headers = edf_reader.getSignalHeaders()
                file_header = edf_reader.getHeader()
                edf_reader.close()
                updated_signal = np.array(src_signal)
                new_signal = np.array(eeg_signals[i, :]).astype(src_signal.dtype)
                updated_signal[0:len(new_signal)] = new_signal
                signal_headers[0]['n_samples'] = len(updated_signal)
                dest_file = os.path.normpath(os.path.join(dest_night, f"{ch}.edf"))
                writer = pyedflib.EdfWriter(dest_file, n_channels=1, file_type=pyedflib.FILETYPE_EDF)
                writer.setHeader(file_header)
                writer.setSignalHeader(0, signal_headers[0])
                writer.writeSamples([updated_signal])  # always write as list of arrays
                writer.close()
                print(f"\t\t{ch}.edf")
                break

#%%
def create_filtered_usa_samples(spiky_ep_indx, filtered_data, spec_feats_eeg, stat_feats_eeg, spec_feats_acc_agg, stat_feats_acc_agg, eeg_samp_rate):
    filtered_data = filtered_data[:, np.newaxis, :]
    spec_feats_f = extract_spectrogram_features(filtered_data, eeg_samp_rate)
    stat_feats_f = extract_tsfel_features(filtered_data, eeg_samp_rate, 'filtered')    
    
    for i in range(spiky_ep_indx.shape[0]):
        ch, ep = spiky_ep_indx[i]
        spec_feats_eeg[ep, ch, :, :] = spec_feats_f[i, :, :, :]
        stat_feats_eeg[ep, ch, :] = stat_feats_f[i, :, :]
    
    usa_samples = create_samples_usability(spec_feats_eeg, stat_feats_eeg, spec_feats_acc_agg, stat_feats_acc_agg)
    return usa_samples

#%%
def count_spiky_to_usable(usa_scores, usa_scores_new):
    mask_3 = (usa_scores == 3)
    mask_3_to_0 = (usa_scores == 3) & (usa_scores_new == 0)
    recovered = (np.sum(mask_3_to_0) / np.sum(mask_3)) * 100    
    print(f"{Fore.GREEN}{Style.BRIGHT}\t{np.sum(mask_3_to_0)} epochs ({round(recovered, 2)}%) with Spiky Noise became usable after filtering.{Style.RESET_ALL}")
    return

#%%
def log_error(night_dir, k, error_details=None):
    if error_details is not None:
        txt = Error_Txts[k] + error_details
    else:
        txt = Error_Txts[k]
    Error_Nights.loc[len(Error_Nights)] = [night_dir, txt]
    print(f"{Fore.RED}{Style.BRIGHT}\tError: {txt}\n\tIgnoring current night...{Style.RESET_ALL}")

#%%
def print_report(out_dir, N):
    print(f"{Fore.CYAN}{Style.BRIGHT}\nExecution Finished!\n{N - len(Error_Nights)} {Device_Name} recordings(s) were successfully processed.")
    
    if Error_Nights.empty is False:
        error_report_path = os.path.normpath(os.path.join(out_dir, "eegFloss_error_nights.csv"))
        Error_Nights.to_csv(error_report_path, index=False)
        print(f"{Fore.RED}{Style.BRIGHT}Checking {len(Error_Nights)} night(s) failed. Please see this file for details:{Style.RESET_ALL}\n{error_report_path}")

#%%
if __name__ == "__main__":
    print(f"{Fore.GREEN}{Style.BRIGHT}\neegFloss_functions.py was run directly.\nAll necessary packages and functions were successfully loaded!{Style.RESET_ALL}")



