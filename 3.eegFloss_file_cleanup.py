"""
Created by Niloy Sikder
Radboud University Medical Center, Donders Institute for Brain, Cognition and Behaviour, Nijmegen, The Netherlands
Faculty of Technology and Bionics, Rhine-Waal University of Applied Sciences, Kleve, Germany
niloy.sikder@donders.ru.nl, niloy.sikder@hochschule-rhein-waal.de,
Google Scholar: https://scholar.google.com/citations?user=0ALk5j4AAAAJ&hl=en
ORCID: 0000-0002-9016-6105
Copyright (c) 2024 Niloy Sikder
"""
#%% What does this script do?
"""
This script can delete a list of files (files_to_be_deleted) from a directory (data_dir).
Since eegFloss saves many forms of outputs to avoid re-calculation, it might be preferable to remove unnecessary files once the processing is complete.
Please note that this operation is not reversible. So, please check the input carefully.
"""
#%% Inputs:
    
data_dir = r'C:\Sciebo_files\eegFloss_v1.0\Full_test_data\Zmax\out\S2'

files_to_be_deleted = ['usability_graph.png', 
                    'usability_scores.csv',
                    'atrifact_rejected_sleep_scores.csv',
                    'atrifact_rejected_sleep_scores_TIB.csv',
                    'eegFloss_prediction_mats.pkl',
                    'hypnogram_with_usability.png',
                    'lights_out_lights_on_moments.csv',
                    'sleep_statistics.csv',
                    'usability_scores_TIB.csv',
                    ]

#%%
import os
def delete_files_recursive(root_dir, clean_files):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file_name in filenames:
            if file_name in clean_files:
                file_path = os.path.join(dirpath, file_name)
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"\tDeleted: {file_path}")
                    except Exception as e:
                        print(f"\tError deleting {file_path}: {e}")
    print("\nAll Files deleted!")

#%%
delete_files_recursive(data_dir, files_to_be_deleted)
