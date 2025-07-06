![License](https://img.shields.io/github/license/Niloy333/eegFloss?style=flat&color=brightgreen)
[![DOI](https://zenodo.org/badge/926585143.svg)](https://doi.org/10.5281/zenodo.15823969)
![Repo size](https://img.shields.io/github/repo-size/Niloy333/eegFloss)
![Last commit](https://img.shields.io/github/last-commit/Niloy333/eegFloss)
![Downloads](https://img.shields.io/github/downloads/Niloy333/eegFloss/total)

# *eegFloss*: A Python Package to *Floss out* Artifacts from Sleep EEG Recordings

**Table of Contents:** [Overview](#overview) | [Installation](#installation) | [Script Descriptions](#script-descriptions) | [Read Before Execution](#read-before-execution) | [Primary Artifacts](#primary-artifacts) | [eegUsability Models](#eegusability-models) | [Sample Outputs](#sample-outputs) | [Reference Paper](#reference-paper) | [Cite](#cite) | [People](#people)

## Overview
EEG data often contains artifacts caused by both internal factors (such as device issues) and external influences (such as movement or environmental noise). In sleep research, these artifacts frequently go unnoticed or undetected, which can impair the performance and reliability of data-driven models or analyses, especially automatic sleep-stage scoring models,  and reduce the credibility of study outcomes.

Most existing artifact detection methods rely on threshold-based techniques. While easy to implement, these methods often struggle to detect complex or unfamiliar artifacts and typically lack generalizability across datasets.

**eegFloss** addresses this challenge with *eegUsability*—a machine learning (ML) model designed to detect artifact-contaminated EEG segments in sleep recordings. The model was trained and evaluated on manually artifact-labeled EEG data from 15 participants, collected over 127 nights using the Zmax headband. However, it can also be applied to sleep EEG data from other devices to detect common artifacts and assess the usability of the data.

The package also includes *eegMobility*—an ML model that detects the degree of movement throughout the night based on **Zmax** accelerometer data. This information is used to automatically detect Time-in-Bed (TIB). For further details, please refer to the [associated paper](#reference-paper).

## Installation
It is recommended to use **eegFloss** within a dedicated Anaconda or Miniconda environment. Follow these steps:

1. Download and install [Anaconda or Miniconda](https://www.anaconda.com/download/success) for your operating system.
2. [Download eegFloss](https://github.com/Niloy333/eegFloss/archive/refs/heads/base.zip), extract the compressed file, and place it in a suitable and accessible directory.
3. On Linux, ensure that the appropriate graphics driver is installed and hardware acceleration is enabled.
4. Launch the Anaconda Prompt:
   - **Windows**: Search for "Anaconda Prompt" in the Start menu.
   - **Linux**: Open a terminal and run:
   `source ~/anaconda3/bin/activate` or `source ~/miniconda3/bin/activate`.
   - **macOS**: Open a terminal.
5. In the prompt, navigate to the extracted `eegFloss` directory. Example:  
   `cd C:\eegFloss_v1.0\`
6. Create a new environment named `eegFloss` with all the necessary packages:  
   `conda env create --name eegFloss --file eegFloss_dependencies.yml`
7. Once the environment is created, activate it:  
   `conda activate eegFloss`
8. To start coding:
   - Launch Spyder and manually open the scripts:
     `spyder`
   - Or, use your preferred code editor and run the script from the command line. Example:  
     `python 1.eegFloss_check_usability_mobility.py`

## Script Descriptions

### `1.eegFloss_check_usability_mobility.py`
- Detects artifacts in sleep EEG data using a chosen [eegUsability model](#eegusability-models).
- Aggregates the provided sleep scores with *usability scores* (the outcomes of artifact detection) using a majority rule (an epoch is marked unusable if more than half of its constituent segments are unusable) to generate *artifact-rejected sleep scores*.
- Automatically identifies *Lights Out* and *Lights On* moments using a chosen *eegMobility* model and computes Time-in-Bed (TIB).
- Computes common sleep statistics based on the artifact-rejected sleep scores and TIB.
- Generates visualizations such as *usability graphs* (shows channel-wise data usability) and *hypnograms* (shows the overall outcomes) to better illustrate the models' outputs.

### `2.eegFloss_spiky_noise_filter.py`
- Identifies the presence of Spiky artifacts in EEG recordings.
- If detected, it applies a custom filter to remove the artifact.
- Saves the cleaned data to a new file (which can then be sleep-scored).

### `3.eegFloss_file_cleanup.py`
- Deletes intermediate or unnecessary files generated during processing to reduce clutter.

### `eegFloss_functions.py`
- Contains all imports and helper functions required by the main scripts.
- Modify this file only if you need to customize core functionalities.

### `read_outputs_in_Matlab.m`
- Shows how to read various eegFloss output files in MATLAB.

**Please read the comments in the input-output cells of each script carefully before running the script.**

# Read before Execution

### File Type: EDF & BDF

- eegFloss currently supports only EDF and BDF files. Therefore, raw EEG signal(s) must be stored in the EDF/BDF format.
- If your data is in a different format, check whether the associated software suite of your recording device allows exporting or converting data to EDF.
- If not, you can manually convert data using Python libraries such as [PyEDFlib](https://pyedflib.readthedocs.io/en/latest/) or [MNE](https://github.com/mne-tools/mne-python).

### Sleep-Stage Scoring
- eegFloss **does not include** a built-in automatic sleep-stage scorer and cannot infer sleep stages from EEG or other signals.
- However, if you provide sleep scores alongside your data, it can generate artifact-rejected sleep scores by combining the provided sleep scores and the detected data usability.
- If your data is not manually scored, consider using open-source automatic sleep scorers such as [U-Sleep](https://sleep.ai.ku.dk/), [YASA](https://yasa-sleep.org/), or [Dreamento](https://github.com/dreamento/dreamento) (only for Zmax data).

### Sleep Score Format
- Sleep stages are expected to be labeled as 0 = Wake, 1 = N1, 2 = N2, 3 = N3, and 4 or 5 = REM. Deviating from this convention will result in incorrect visualizations and sleep statistics.
- The sleep scores must be stored in the first column of a TXT/CSV file located in the same directory as the corresponding EDF file. The number of epochs must match the recording duration.

### Data Organization
- Each recording should reside in a separate directory. Placing multiple recordings in the same directory will cause eegFloss outputs to overwrite one another. Provide the parent directory as the `Raw_Data_Dir`.

### Output Directory Management
- While it is possible to save eegFloss outputs in the same directory as the recordings by setting `Output_Dir = Raw_Data_Dir`, this is not recommended.
- eegFloss checks for prior outputs to skip redundant processing. So, consider saving the additional outputs, even if they are not needed for your analysis.
- If you later modify key settings (e.g., change the usability model or adjust TIB thresholds), the tool may incorrectly skip reprocessing due to existing outputs. To prevent this, use a separate output directory for each round of processing.
- Feature extraction is typically the most time-consuming step (unless using a ‘lite’ model). To avoid recomputation, copy the `eegFloss_stat_features.npz` files to the data directory alongside the EDF files after processing the data once.

### TIB Detection and Accelerometer Requirements
- Automatic TIB detection using the eegMobility model is validated only for Zmax data.
- If you want to test it for another device, ensure that the tri-axial accelerometer data is measured in units of g, falls within a range of ±2g (clip extreme values if needed), and includes gravitational acceleration (meaning the normalized data should center around 1g).
- Analyzing EEG data without accompanying accelerometer signals may lead to the removal of some arousals due to a lack of motion information.

### Miscellaneous
- For non-Zmax devices, verify that sampling rates are correct. If initial results are suboptimal, consider applying normalization techniques.
- Thoroughly read and **update all the fields** in the input-output cells according to your dataset and desired outputs before running the script.
- The package has been tested on Windows 10 and 11, Ubuntu 24.04.2, and macOS Sequoia 15.3.1 (MacBook Air, 2018).
- Known issue on Linux: **Could not initialize GLX**. To solve this, ensure that step 3 of [Installation](#Installation) was done correctly. Then try (one by one):
  ```bash
   pip install PyQtWebEngine
   QT_XCB_GL_INTEGRATION=none
   QT_DEBUG_PLUGINS=1
   QT_QPA_PLATFORM=wayland spyder
   QT_QPA_PLATFORM=xcb spyder
   QT_QPA_PLATFORM=offscreen spyder

## Primary Artifacts
eegUsability detects the following artifacts in raw sleep EEG data:
<img src="https://github.com/Niloy333/eegFloss/blob/base/figures/noise_classes_port.png" alt="primary_artifacts" width="600">

>Figure 1: (a) A windowed spectrogram (blue: low power, red: high power) of a sample Zmax EEG channel, highlighting segments containing different artifacts. The corresponding time-domain representations of these segments are shown for (b) Good Data, (c) No Data, (d) High Noise, (e) Spiky Noise, and (f) M-shaped Noise.

## eegUsability Models

| eegUsability version         | Feature set(s)              | Specialty                                                                                         | When to use                                                                                       | F1-score (%)      | Processing time (8-hr night)† |
|-----------------------------|-----------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------|-------------------------------|
| “v1.0” or “default”         | Spectrogram and statistical | Combines two feature sets for consistent outputs. Tested across datasets and the most dependable. | Best for general tasks requiring maximum data retention.                                         | 84.87             | ≈35 sec                      |
| “v0.8” or “weighted-M”      | Spectrogram and statistical | Better at identifying M-shaped Noise but sacrifices a bit more usable data.                      | Ideal when M-shaped Noise detection is crucial and slight data loss is acceptable.              | 86.35             | ≈36 sec                      |
| “v0.6” or “binary”          | Spectrogram and statistical | Only identifies whether the data is usable or not; does not differentiate noise types.           | If noise differentiation is entirely unnecessary or processing simplicity is prioritized.       | 89.4              | ≈35 sec                      |
| “v0.7” or “lite”            | Spectrogram                 | Uses only one feature set; similar to v1.0, but 12 times faster with comparable results.          | Suitable for quick results where minor inconsistencies are tolerable.                           | 84.94             | ≈3 sec                       |
| “v0.7.2” or “lite weighted-M” | Spectrogram               | Similar to v0.8, but works on only spectrogram features, hence is faster.                         | Optimal for quick, precise outputs.                                                              | 86.37             | ≈3 sec                       |
| “v0.7.3” or “lite binary”   | Spectrogram                 | Similar to v0.6 but works on only spectrogram features, hence is faster.                          | Handy when fast results are needed without noise type differentiation.                          | 89.29             | ≈3 sec                       |
| “v0.9” or “full”            | Spectrogram and statistical | Similar to v1.0 but is trained on the entire available dataset.                                   | Can be used if a more hypertuned model is needed.                                               | 90.11^            | ≈37 sec                      |

>^Results are from a test set that is a subset of the training data. †Tested on a Core i7, 8C/16T, 2.5–4.8 GHz processor with no resource-intensive processes running in parallel.

## Sample Outputs

### Usability Graph
<img src="https://github.com/Niloy333/eegFloss/blob/base/figures/usability_graph.png" alt="usability_graph_zmax" width="1000">

>Figure 2: The usability graph of a sample Zmax recording showing (a) a windowed spectrogram of the EEG Left channel, (b) its usability scores, (c) the normalized acceleration calculated from tri-axial ACC data, (d) a windowed spectrogram of the EEG Right channel, and (e) its usability scores.
 
### Hypnogram
<img src="https://github.com/Niloy333/eegFloss/blob/base/figures/hypnogram_with_usability.png" alt="hypnogram_zmax" width="1000">

>Figure 3: eegFloss outputs of a sample Zmax recording showing spectrograms of (a) EEG Left and (b) EEG Right channels, (c) the normalized acceleration, (d) hypnogram based on the artifact-rejected autoscores, and (e) the mobility labels with TIB bounded by Lights Out and Lights On moments.

**The `sample_output` folder contains a sample of all output files.**

## Reference Paper
More information on this package and the underlying models can be found in:

N. Sikder, P. Zerr, M. Dresler, & M. Krauledat, *eegFloss: A Python package for refining sleep EEG recordings with machine learning models* (submitted).

For now, you can see a [preprint version](TBA).

## Cite
If you find this package helpful and use it in your work, please cite the reference paper as:

TBA

And cite the package as:

N. Sikder, Niloy333/eegFloss: eegFloss v1.0. Zenodo, 2025. doi: 10.5281/ZENODO.15823969

## People
© [Niloy Sikder](https://scholar.google.com/citations?hl=en&user=0ALk5j4AAAAJ&view_op=list_works&sortby=pubdate)<sup>1,2,#</sup>, [Paul Zerr](https://scholar.google.com/citations?hl=en&user=9CldqFoAAAAJ&view_op=list_works&sortby=pubdate)<sup>1,$</sup>, [Martin Dresler](https://scholar.google.com/citations?hl=en&user=Y-hAEQYAAAAJ&view_op=list_works&sortby=pubdate)<sup>1,$</sup>, & [Matthias Krauledat](https://scholar.google.com/citations?hl=en&user=n9q-wxgAAAAJ&view_op=list_works&sortby=pubdate)<sup>2,$</sup>   
<sup>1</sup>Radboud University Medical Center, Donders Institute for Brain, Cognition and Behaviour, Nijmegen, The Netherlands.  
<sup>2</sup>Faculty of Technology and Bionics, Rhine-Waal University of Applied Sciences, Kleve, Germany.  
<sup>#</sup>Developer  
<sup>$</sup>Supervisor

**This package is provided *as is*, without any warranties, express or implied. eegFloss is released under the MIT License and is free to use, modify, and integrate with other software, provided that appropriate credit is given.**

For questions, assistance, suggestions, or further information: [contact the developer](mailto:niloy.sikder@hochschule-rhein-waal.de)
