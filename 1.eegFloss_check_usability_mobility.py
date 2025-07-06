#%% Author information:
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
2. It can combine usability scores with sleep scores (based on the majority rule) to generate artifact-rejected sleep scores.
3. It automatically determines the Lights Out and Lights On moments and calculates Time-in-Bed (TIB) from Zmax accelerometer data using the eegMobility model.
4. It computes sleep statistics from the artifact-rejected sleep scores, factoring in the TIB duration.
"""
#%% Read before executing the script:
"""
1. eegFloss_v1.0 can process only EDF files.
2. The script does not autoscore sleep data. You must provide sleep scores along with the data (in the same directory) for the aggregation process. However, it can generate a set of artifact-rejected pseudo-sleep scores (where '9' denotes the placeholder for a sleep stage), which can later be aggreagted with the actual sleep scores.
3. Sleep stages are expected to be labeled as 0: Wake, 1: N1, 2: N2, 3: N3, and 4/5: REM. Using different labels will result in incorrect graphs and sleep statistics.
4. Automatic TIB detection using eegMobility is applicable only to Zmax data. Its applicability to other devices' data has not been checked extensively.
5. Analyzing EEG signals without accelerometer outputs may result in the removal of (some) arousals.
6. For non-Zmax data, carefully review the sampling rates. If initial outputs are unsatisfactory, consider implementing normalization techniques.
7. If sleep statistics are calculated without TIB detection, the entire duration of the night will be considered as TIB.
8. Consider saving the additional outputs, even if they are not immediately needed, as they can help the script skip certain steps and save time when reprocessing the same nights.
9. Adjust all the input variables accrodingly to avoid errors.
10. If you decide to change the ML model's version, please make sure the input & output directories do not contain previous results.
"""
#%% Primary inputs:

Raw_Data_Dir = r'C:\0.eegFloss_v1.0\sample_data\Zmax'
# Directory containing the EDF files with EEG data. Each night's data must be stored in a separate subdirectory within Raw_Data_Dir. Always provide absolute paths.

Output_Dir = r'C:\eegFloss_v1.0\sample_data\Zmax\output1'
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

Sleep_Scores_Flname = 'DreamentoScorer.txt'  #'Autoscores.txt'
# Name of the (TXT/CSV) file containing the sleep scores (numerical values between 0 and 5).
# Needs to be the same for all nights and must be present in the same subdirectory within Raw_Data_Dir or Output_Dir for aggregation. Other options:
# None: The sleep data has not been scored. Do not aggregate sleep and usability scores.
# 'pseudo_scores': Generate a set of pseudo-scores where '9' denotes a sample sleep stage.

Sleep_Scores_Epoch_Length = 30
# Window size (in sec) in which sleep scoring was performed.

#%% Calibration:

Check_Usability = True
# False: Data usability will not be checked.

eegUsability_Model = 'default'
# The specific usability model to use.
# Options: 'default'/'v1.0', 'weighted-M'/'v0.8', 'binary'/'v0.6', 'lite'/'v0.7', 'lite-weighted-M'/'v0.7.2', 'lite-binary'/'v0.7.3', 'full'/'v0.9'.

Usability_Epoch_Length = 10
# Window size (in sec) for segmenting data during usability checking. Must be a divisor of Sleep_Scores_Epoch_length.

Sleep_Scores_Skiprows = 5
# Number of header/info rows to skip in the sleep score file to reach the first numerical value.
# Set to 0 if the file contains only numbers, or to 1 if it includes column names.
# eegFloss assumes sleep scores are located in the first column.

Aggregate_Scores = True
# False: Sleep scores and usability scores will not be aggregated.

Unusable_Label = -1
# Numeric label to represent "Unusable" in the aggregated scores.
# Must be distinct from any valid sleep stage label (usually 0–5).

Ignore_M_Shaped_Noise = False
# True: The M-shaped Noise will be ignored.
# The M-shaped Noise is likely native to Zmax. Ignoring it might be beneficial for non-Zmax data.

Determine_TIB = True
# False: TIB will not be determined from Zmax accelerometer data.

eegMobility_Model = 'default'
# Mobility detection model. Options: 'default' or 'lite' (a lightweight version).

Min_Laying_Time_for_TIB = 2
# Minutes of uninterrupted lying down required to indicate the participant has gone to bed.

Calculate_Sleep_Stats = True
# False: Sleep statistics will not be calculated.

Save_Features = True
# False: TSFEL features will not be saved.
# Saving TSFEL features is recommended to avoid time-consuming recalculations.

Save_Prediction_Mats = True
# False: Detailed prediction outputs will not be saved.
# Saving prediction matrices is recommended for future in-depth analysis.

#%% Output file names:

# To avoid saving a particular output, use None instead of a filename string. Do NOT remove the variable.
# Save raw outputs as CSV files, which can be read by TXT/CSV readers.
# Saving plots as vector graphics may increase file size and processing time.

Usability_Scores_Flname = 'usability_scores.csv'
# File to store the usability scores with the detected artifact labels.

Usability_Graph_Flname = 'usability_graph.png'
# File to save the spectrogram and artifact detection visualizations.

Artifact_Rejected_Scores_Flname = 'atrifact_rejected_sleep_scores.csv'
# File to store the aggregated sleep scores.

Lights_Out_On_Flname = 'lights_out_lights_on_moments.csv'
# File to store the detected Lights Out and Lights On moments. Also a part of the sleep stats.

Artifact_Rejected_Scores_within_TIB_Flname = 'atrifact_rejected_sleep_scores_TIB.csv'
# File to store the aggregated sleep scores within the TIB duration.

Usability_Scores_within_TIB_Flname = 'usability_scores_TIB.csv'
# File to store the  scores within the TIB duration.

Hypnogram_Flname = 'hypnogram_with_usability.png'
# File to save a figure showing a hypnogram, spectrograms, and the detected TIB.

Sleep_Stats_Flname = 'sleep_statistics.csv'
# File to store the calculated sleep statistics.

Add_Index_in_Outputs = 'timestamp'
# 'timestamp': Include start and end timestamps of each epoch with their usability/sleep scores.
# 'data_index': Include start and end data indices of each epoch.
# None: Do not include any indices.

''' The necessary input has been provided, please run the script.'''

#%% Importing packages, functions and ML models, and preforming the initial checkes:

# from eegFloss_functions import *
import os, inspect
Script_Dir = os.path.normpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
os.chdir(Script_Dir)                # Change to script directory
with open(r'eegFloss_functions.py', 'r') as file:
    eegFloss_functions = file.read()
exec(eegFloss_functions)            # Execute loaded functions into memory
declare_globals()                   # Set global variables and constants
initial_checks()                    # Verify all inputs, folders, dependencies

# Load ML models for artifact detection and mobility prediction:
usability_model, mobility_model = load_usability_model(), load_mobility_model('Zmax')

#%% Listing nights for processing:

# Scan directories under Raw_Data_Dir to find nights
all_nights = find_nights(Raw_Data_Dir)
total_nights = len(all_nights)      # Total nights to process
num_done_nights = 0                 # Counter for processed nights

#%% Main():

start_time = tic()                  # Start stopwatch

for num_night, src_night in enumerate(all_nights, start=1):
    ## num_night, src_night = 1, all_nights[0]
    # Determining the destination directory:
    dest_night = determine_dest_dir(src_night, num_night, total_nights)

    # Initializing important variables:
    eeg_signals = eeg_samp_rates = org_dur = eeg_samp_rate = samples_eeg = spec_feats_eeg = acc_signals = acc_samp_rates = acc_samp_rate = acc_signals_ok = acc_agg = samples_acc_agg = spec_feats_acc_agg = usa_samples = usability_scores = pred_mat_data = plot_data = usability_scores_df = sleep_scores = agg_scores = mob_samples = mobility_scores = lights_out_ep = lights_on_ep = None
    stat_feats_eeg = stat_feats_acc_agg = stat_feats_acc = np.nan

    # Checking previous progress to avoid recomputation:
    file_exists = check_exists(src_night)

#%% Step-1: Checking channel-wise EEG data usability:

    if Check_Usability is True and not file_exists['usability_scores']:
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
            if acc_signals is None:  continue

            # All ACC channels' samp_rate needs to be the same as eeg_samp_rate. If not, adjusting:
            acc_signals, acc_samp_rate = adjust_samp_rate(
                acc_signals, acc_samp_rates, eeg_samp_rate)

            # Converting to np array:
            acc_signals = crop_n_make_nparray(acc_signals, acc_samp_rate)

            # Checking ACC signals' compatibility (range: ±2, unit: g):
            acc_signals_ok = check_acc_signals(src_night, acc_signals)

            # If ACC signals do not follow the training data format, skipping the night:
            if acc_signals_ok is False: continue

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
        usability_scores, pred_mat_data = predict_usability(
            src_night, usa_samples, usability_model, eeg_samp_rate)

        # Saving features for future use:
        if Save_Features is True:
            save_extracted_features(
                dest_night, stat_feats_eeg, stat_feats_acc_agg, stat_feats_acc)
        
        # Saving prediction metrics for future use:
        if Save_Prediction_Mats is True:
            save_pred_mat(dest_night, pred_mat_data)
        
        # Deleting large, unnecessary variables:
        del samples_eeg, spec_feats_eeg, samples_acc_agg, spec_feats_acc_agg, usa_samples

#%% Step-2: Plotting usability graph:
    
    if Usability_Graph_Flname is not None and not file_exists['usability_scores']:

        if eeg_signals is None:
            # If EEG data is not in the memory, reading it again:
            eeg_signals, eeg_samp_rates, org_dur = read_data(src_night, 'EEG')
            if eeg_signals is None: continue
            eeg_signals, eeg_samp_rate = adjust_samp_rate(
                eeg_signals, eeg_samp_rates)
            eeg_signals = crop_n_make_nparray(eeg_signals, eeg_samp_rate)

        if acc_agg is None:
            if ACC_Channels is not None:
                # If ACC data is not in the memory, reading it again:
                acc_signals, acc_samp_rates, _ = read_data(src_night, 'ACC')
                if acc_signals is None:  continue
                acc_signals, acc_samp_rate = adjust_samp_rate(
                    acc_signals, acc_samp_rates, eeg_samp_rate)
                acc_signals = crop_n_make_nparray(acc_signals, acc_samp_rate)
                acc_signals_ok = check_acc_signals(src_night, acc_signals)
                acc_agg = euclidean_norm(acc_signals)

        if usability_scores is None:
            # Reading existing usability scores if not in the current memory:
            # Use this function to read eegFloss CSV outputs in your work:
            usability_scores_df, start_end_info_df, usability_scores_header = read_scores(
                file_exists['usability_scores'])
            usability_scores = usability_scores_df.values.astype(np.int8)

        # Plotting the usability graphs:
        plot_data = plot_usability_graph(
            src_night, eeg_signals, eeg_samp_rate, acc_agg, acc_samp_rate, usability_scores, file_exists)

#%% Step-3: Aggregating sleep and usability scores:
    
    if Aggregate_Scores is True and Sleep_Scores_Flname is not None and (not file_exists['artifact_rejected_scores'] or not file_exists['artifact_rejected_scores_within_tib']):
        if usability_scores is None:
            usability_scores_df, _, _ = read_scores(
                file_exists['usability_scores'])
            usability_scores = usability_scores_df.values.astype(np.int8)

        # Reading sleep scores:
        sleep_scores = read_sleep_scores(
            src_night, file_exists['Sleep_Scores'], org_dur, usability_scores.shape[0])
        # If sleep scores could not be read, the night will be skipped:
        if sleep_scores is None: continue

        # Aggregating sleep and usability scores:
        agg_scores = aggregate_scores(
            dest_night, sleep_scores, usability_scores, eeg_samp_rate)

#%% Step-4: Checking the degree of the participant's mobility (unless previously done):
    
    if Determine_TIB is True and ACC_Channels is not None and (not file_exists['lights_out_on'] or not file_exists['usability_scores_within_tib'] or not file_exists['artifact_rejected_scores_within_tib']):
        
        if acc_signals is None:
            acc_signals, acc_samp_rates, _ = read_data(src_night, 'ACC')
            if acc_signals is None: continue
            acc_signals, acc_samp_rate = adjust_samp_rate(
                acc_signals, acc_samp_rates, max(acc_samp_rates.values()))
            acc_signals = crop_n_make_nparray(acc_signals, acc_samp_rate)
            acc_signals_ok = check_acc_signals(src_night, acc_signals)
            if acc_signals_ok is False: continue

        # Reshaping to make samples with acc_samp_rate*Mobility_Epoch_Length datapoints each:
        samples_acc = make_samples(acc_signals, acc_samp_rate, 'Mobility')

        if 'lite' in eegMobility_Model:
            # Extracting features for eegMobility lite model:
            mob_samples = extract_pow_feats(samples_acc, acc_samp_rate)

        else:
            # Extracting statistical features for the default eegMobility model:
            stat_feats_acc = get_stat_feats(
                file_exists, samples_acc, stat_feats_acc, acc_samp_rate, 'ACC')

            # Creating samples for mobility classification:
            mob_samples = create_samples_mobility(stat_feats_acc)

        # Checking the degree of mobility:
        mobility_scores, pred_mat_data = predict_mobility(
            src_night, mob_samples, mobility_model, acc_samp_rate, pred_mat_data)

        if Save_Features is True:
            save_extracted_features(
                dest_night, stat_feats_eeg, stat_feats_acc_agg, stat_feats_acc)

        if Save_Prediction_Mats is True:
            save_pred_mat(dest_night, pred_mat_data)

        del samples_acc, mob_samples, stat_feats_eeg, stat_feats_acc_agg, stat_feats_acc

        # Determing the  from mobility scores:
        lights_out_ep, lights_on_ep = determine_lights_out_on(
            mobility_scores, dest_night, acc_samp_rate)

        # Saving sleep and usability scores within TIB:
        if lights_out_ep is not None:
            save_scores_within_TIB(src_night, file_exists, usability_scores,
                                   agg_scores, lights_out_ep, lights_on_ep, eeg_samp_rate, 
                                   org_dur)
    
    # updating the counter:
    num_done_nights = num_done_nights + 1

#%% Step-5: Calculating sleep statistics:
    
    if Calculate_Sleep_Stats is True:
        stats_ok = calculate_n_save_stats(
            src_night, file_exists, sleep_scores, agg_scores, lights_out_ep, lights_on_ep)

#%% Step-6: Plotting hypnogram:
    
    if Hypnogram_Flname is not None and Sleep_Scores_Flname is not None and Sleep_Scores_Flname != 'pseudo_scores':
        plot_ok = plot_hypnogram(src_night, eeg_signals, eeg_samp_rate, acc_agg, acc_samp_rate,
                                 agg_scores, lights_out_ep, lights_on_ep, mobility_scores,
                                 plot_data)

#%% End of processing:
    
    print(f"{Fore.GREEN}{Style.BRIGHT}\tThe night was processed successfully.\nOutputs were saved to:\n{dest_night}'{Style.RESET_ALL}")
    gc.collect()

    # Showing estimated remaining time:
    if num_night < len(all_nights):
        toc(start_time, num_done_nights, num_night, total_nights)

#%% Report:

print_report(Output_Dir, total_nights)
