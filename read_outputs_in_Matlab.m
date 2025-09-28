% This script shows how to read various outputs of the eegFloss package in MATLAB.

% The night's directory:
night_dir = "C:\Sciebo_files\eegFloss_v1.0\Out_dir\S1";

% Default eegFloss output file names:
Usability_Scores_Flname = 'usability_scores.csv';
Artifact_Rejected_Scores_Flname = 'atrifact_rejected_sleep_scores.csv';
Lights_Out_On_Flname = 'lights_out_lights_on_moments.csv';
Artifact_Rejected_Scores_within_TIB_Flname = 'atrifact_rejected_sleep_scores_TIB.csv';
Usability_Scores_within_TIB_Flname = 'usability_scores_TIB.csv';
Sleep_Stats_Flname = 'sleep_statistics.csv';

% Read the scores:
Usabilty_scores = readtable(fullfile(night_dir, Usability_Scores_Flname), "CommentStyle", "#");
Usabilty_scores_tib = readtable(fullfile(night_dir, Usability_Scores_within_TIB_Flname), "CommentStyle", "#");
artifact_rejected_scores = readtable(fullfile(night_dir, Artifact_Rejected_Scores_Flname), "CommentStyle", "#");
artifact_rejected_scores_tib = readtable(fullfile(night_dir, Artifact_Rejected_Scores_within_TIB_Flname), "CommentStyle", "#");
lights_out_on_time = readtable(fullfile(night_dir, Lights_Out_On_Flname), "CommentStyle", "#");
sleep_stats = readtable(fullfile(night_dir, Sleep_Stats_Flname), "CommentStyle", "#", ...
    "ReadVariableNames", false, "ReadRowNames", 1);

% Flip sleep_stats:
sleep_stats_row = array2table(sleep_stats{:,:}', 'VariableNames', sleep_stats.Properties.RowNames);


