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
1. It evaluates the usability of sleep EEG data by detecting various artifacts using the eegUsability model. The model is trained on Zmax data but applies to sleep EEG data from other devices as well.
2. It then uses a special filter to filter out the spiky noise and save the filtered signals in a separate file/files.
3. It also re-checks the filtered signals to see if the parts containing spiky noise have become usable after filtering.
"""
#%% Input-output options:

Raw_Data_Dir = r'C:\Sciebo_files\eegFloss_v1.0\0.eegFloss_v1.0_beta\sample_data\Zmax'
# Directory containing the EDF files with EEG data. Each night's data must be stored in a separate subdirectory within Raw_Data_Dir. Always provide absolute paths.

Output_Dir = r'C:\Sciebo_files\eegFloss_v1.0\0.eegFloss_v1.0_beta\sample_data\Zmax\output'
# Directory to store the generated scores and other output files.
# = Raw_Data_Dir (will save outputs alongside the data)

Device_Name = 'Zmax'
# Name of the recording device.

EEG_Channels = ['EEG L', 'EEG R']
# List of EEG channels to be processed.

ACC_Channels = ['dX', 'dY', 'dZ']
# Names of the tri-axial accelerometer channels, order: X, Y, Z.
# None: accelerometer data unavailable.

All_Signals_in_One_File = False
# False: Each signal is stored in a separate EDF file (e.g., for Zmax).
# True: All signals are stored in a single EDF file as separate channels.

Check_Usability = True
# False: Data usability will not be checked.

eegUsability_Model = 'default'
# The specific usability model to use.
# Options: 'default'/'v1.0', 'weighted-M'/'v0.8', 'binary'/'v0.6', 'lite'/'v0.7', 'lite-weighted-M'/'v0.7.2', 'lite-binary'/'v0.7.3', 'full'/'v0.9'.

Usability_Epoch_Length = 10
# Window size (in sec) for segmenting data during usability checking. Must be a divisor of Sleep_Scores_Epoch_length.

Ignore_M_Shaped_Noise = False

Usability_Scores_Flname = 'usability_scores.csv'
# File to store the usability scores with the detected artifact labels.

Usability_Graph_Flname = 'usability_graph.png'
# File to save the spectrogram and artifact detection visualizations.

Add_Index_in_Outputs = 'timestamp'
# 'timestamp': Include start and end timestamps of each epoch with their usability/sleep scores.
# 'data_index': Include start and end data indices of each epoch.
# None: Do not include any indices.

#%% Importing packages, functions and ML models, and preforming the initial checkes:

import os, inspect
Script_Dir = os.path.normpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
os.chdir(Script_Dir)                # Change to script directory
with open(r'eegFloss_functions.py', 'r') as file:
    eegFloss_functions = file.read()
exec(eegFloss_functions)            # Execute loaded functions into memory
declare_globals()                   # Set global variables and constants
initial_checks()                    # Verify all inputs, folders, dependencies
usability_model = load_usability_model()

#%% Listing nights for processing:

all_nights = find_nights()
total_nights = len(all_nights)
num_done_nights = 0

#%% Main():

start_time = tic()                  # Start stopwatch

for num_night, src_night in enumerate(all_nights, start=1):
    ## num_night, src_night = 1, all_nights[0]
    
    # Initializing important variables:
    dest_night = initialize_night(src_night, num_night, total_nights)
    eeg_signals = eeg_samp_rates = eeg_samp_rate = org_dur = spec_feats_eeg = acc_signals = acc_samp_rates = acc_samp_rate = acc_agg = samples_acc_agg = spec_feats_acc_agg = usa_samples = usability_scores = pred_mat_data = usability_scores_df = spiky_ep_indx = spiky_data = filtered_data = eeg_signals_filtered = filtered_usa_samples = usability_scores_new = None
    stat_feats_eeg = stat_feats_acc_agg = np.nan

    # Checking previous progress:
    files_list = {'usability_scores': Usability_Scores_Flname,
                  'features': Feats_flname,
                  'usability_scores_spiky': 'usability_scores_after_spiky_noise_filtering.csv'}
    file_exists = check_exists_2(src_night, files_list)

#%% Checking channel-wise EEG data usability:
    
    if Check_Usability is True and not file_exists['usability_scores_spiky']:
        # Reading EEG data:
        eeg_signals, eeg_samp_rates, org_dur = read_data(src_night, 'EEG')
        
        # If EEG data could not be read, the night will be skipped:
        if eeg_signals is None: continue
        
        # All EEG channels' samp_rate needs to be the same. If not, adjusting:
        eeg_signals, eeg_samp_rate = adjust_samp_rate(
            eeg_signals, eeg_samp_rates)

        # Converting to np array:
        eeg_signals = crop_n_make_nparray(eeg_signals, eeg_samp_rate)
        
        # Reshaping to make samples with eeg_samp_rate*Usability_Epoch_Length datapoints each:
        samples_eeg = make_samples(eeg_signals, eeg_samp_rate, 'Usability')
        
        # Extracting spectrogram features from EEG data:
        spec_feats_eeg = extract_spectrogram_features(
            samples_eeg, eeg_samp_rate)

        if ACC_Channels is not None:
            # Reading ACC data:
            acc_signals, acc_samp_rates, _ = read_data(src_night, 'ACC')
            
            # If ACC data could not be read, the night will be skipped:
            if acc_signals is None: continue
            
            # All ACC channels' samp_rate needs to be the same as eeg_samp_rate. If not, adjusting:
            acc_signals, acc_samp_rate = adjust_samp_rate(
                acc_signals, acc_samp_rates, eeg_samp_rate)
            
            # Converting to np array:
            acc_signals = crop_n_make_nparray(acc_signals, acc_samp_rate)
            
            # Calculating aggregated ACC signal:
            acc_agg = euclidean_norm(acc_signals)
            
            # Reshaping to make samples with acc_samp_rate*Usability_Epoch_Length datapoints each:
            samples_acc_agg = make_samples(acc_agg, acc_samp_rate, 'Usability')
            
            # Extracting spectrogram features from ACC data:
            spec_feats_acc_agg = extract_spectrogram_features(
                samples_acc_agg, acc_samp_rate)

        else:
            # If real ACC data is unavailable, features from pseudo ACC data (no movements) is still needed to match the expected featuremap. This step will not cause delay.
            spec_feats_acc_agg, stat_feats_acc_agg, acc_samp_rate = pseudo_acc_feats(
                eeg_signals.shape[1], eeg_samp_rate)

        if eegUsability_Model in usa_lite_versions:
            # For eegUsability lite versions only spectrogram features are necessary.
            # So, samples for classification can already be created.
            usa_samples = create_samples_usability_lite(
                spec_feats_eeg, spec_feats_acc_agg)
        else:
            # For regular versions, statistical features are also necessary. Extracting from EEG:
            stat_feats_eeg = get_stat_feats(
                file_exists, samples_eeg, stat_feats_eeg, eeg_samp_rate, 'EEG')

            # Extracting from ACC:
            if acc_signals is not None:
                stat_feats_acc_agg = get_stat_feats(
                    file_exists, samples_acc_agg, stat_feats_acc_agg, acc_samp_rate, 'ACCagg')
            
            # Combining spectrogram and statistical features to prepare samples for classification:
            usa_samples = create_samples_usability(
                spec_feats_eeg, stat_feats_eeg, spec_feats_acc_agg, stat_feats_acc_agg)

        # Checking data usability:
        usability_scores, _ = predict_usability(
            src_night, usa_samples, usability_model, eeg_samp_rate)

        # Plotting usability graph:
        _ = plot_usability_graph(src_night, eeg_signals, eeg_samp_rate,
                                 acc_agg, acc_samp_rate, usability_scores, file_exists)

        # Finding epochs cntaining Spiky Noise:
        spiky_ep_indx, spiky_data = get_spiky_epochs(
            usability_scores, samples_eeg)

        # Filtering the epochs containing Spiky Noise:
        filtered_data = spiky_noise_filter(spiky_data, eeg_samp_rate)

        # Placing the filtered epochs back to the original EEG signals and saving them:
        eeg_signals_filtered = save_filtered_signals(
            src_night, samples_eeg, spiky_ep_indx, filtered_data)

        # Creating new samples for usability detection:
        filtered_usa_samples = create_filtered_usa_samples(
            spiky_ep_indx, filtered_data, spec_feats_eeg, stat_feats_eeg, spec_feats_acc_agg, stat_feats_acc_agg, eeg_samp_rate)

        # Checking data usability:
        usability_scores_new, _ = predict_usability(
            src_night, filtered_usa_samples, usability_model, eeg_samp_rate, 'post_filter')

        # Calculating how many epochs with Spiky Noise have passed as usable data after filtering:
        count_spiky_to_usable(usability_scores, usability_scores_new)

        # Plotting the usability graph after filtering:
        _ = plot_usability_graph(src_night, eeg_signals_filtered, eeg_samp_rate,
                                 acc_agg, acc_samp_rate, usability_scores_new, file_exists, 'post_filter')

        print(f"{Fore.GREEN}{Style.BRIGHT}\tThe night was processed successfully.\nOutputs were saved to:\n{dest_night}'{Style.RESET_ALL}")
        gc.collect()
        
        num_done_nights = num_done_nights + 1
        # Showing estimated remaining time:
        if num_night < len(all_nights):
            toc(start_time, num_done_nights, num_night, total_nights)

#%% Report

print_report(Output_Dir, total_nights)
