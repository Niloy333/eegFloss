<!--   [![DOI](https://zenodo.org/badge/925334641.svg)](https://doi.org/10.5281/zenodo.xxxxx) -->

**The scripts will be added soon**

# *eegFloss*: A Python Package to floss out artifacts from sleep EEG data

## Overview
EEG data is often affected by artifacts caused by both internal factors (e.g., device issues) and external influences (e.g., movement or environmental noise). In sleep research, these artifacts frequently go undetected, which can impair the performance of automatic sleep-stage scoring models and reduce the reliability of study outcomes.

Most existing artifact detection methods rely on threshold-based techniques. While easy to implement, these methods struggle to detect complex or unfamiliar artifacts and typically lack generalizability across datasets.

**eegFloss** addresses this challenge with *eegUsability*—a machine learning (ML) model designed to detect artifact-contaminated EEG segments in sleep recordings. The model was trained and evaluated on manually labeled EEG data from 15 participants (127 nights) recorded using the Zmax headband, but it can also be applied to sleep EEG data from other devices to detect common artifacts and assess data usability.

The package also includes *eegMobility*—an ML model that detects the degree of movement throughout the night based on **Zmax** accelerometer data. This information is used to automatically detect Time-in-Bed (TIB). For further details, please refer to the [associated paper]().

## Installation
It is recommended to use **eegFloss** within a dedicated Anaconda or Miniconda environment. Follow these steps:

1. Download and install [Anaconda or Miniconda](https://www.anaconda.com/download/success) for your operating system.
2. Download eegFloss, extract the compressed folder, and place it in a suitable and accessible directory.
3. On Linux, ensure that the appropriate graphics driver is installed and hardware acceleration is enabled.
4. Launch the Anaconda Prompt:
   - **Windows**: Search for "Anaconda Prompt" in the Start menu.
   - **Linux**: Open a terminal and run
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
9. Known issue on Linux: **Could not initialize GLX**. To solve this, ensure that step 3 was done correctly. Then try:    
   -`pip install PyQtWebEngine`   
   -`QT_XCB_GL_INTEGRATION=none`   
   -`QT_DEBUG_PLUGINS=1`   
   -`QT_QPA_PLATFORM=wayland spyder`   
   -`QT_QPA_PLATFORM=xcb spyder`   
   -`QT_QPA_PLATFORM=offscreen spyder`
   
## Script Descriptions

### `1.eegFloss_check_usability_mobility.py`
- Detects artifacts in sleep EEG data using a chosen eegUsability model. See [eegUsability Models](#eegusability-models) for details on a specific model.
- Aggregates the provided sleep scores with *usability scores* (the outcomes of artifact detection) using a majority rule (an epoch is marked unusable if more than half of its constituent segments are unusable) to generate *artifact-rejected sleep scores*.
- Automatically identifies Lights Out and Lights On moments and computes Time-in-Bed (TIB) using a chosen *eegMobility* model.
- Computes common sleep statistics based on the artifact-rejected sleep scores and TIB.
- Generates visualizations such as *usability graphs* (shows channel-wise data usability) and *hypnograms* (shows the overall outcomes).

### `2.eegFloss_spiky_noise_filter.py`
- Identifies the presence of Spiky artifacts in EEG recordings.
- If detected, applies a custom filter to remove the artifact.
- Saves the cleaned data to a new file (which can then be sleep-scored).

### `3.eegFloss_file_cleanup.py`
- Deletes intermediate or unnecessary files generated during the processing pipeline to reduce clutter.

### `eegFloss_functions.py`
- Contains all imports and helper functions required by the main scripts.
- Modify this file only if you need to customize core functionalities.

**Please read the comments in the input/output cells of each script carefully before running the script.**

## Primary Artifacts:
<img src="https://hochschule-rhein-waal.sciebo.de/s/khVYFd3BbPnBCQS/download" alt="Artifacts" width="500">
Figure 1: (a) A windowed spectrogram (blue: low power, red: high power) of a sample Zmax EEG channel, highlighting segments containing different artifacts. The corresponding time-domain representations of these segments are shown for (b) Good Data, (c) No Data, (d) High Noise, (e) Spiky Noise, and (f) M-shaped Noise.

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
<img src="https://hochschule-rhein-waal.sciebo.de/s/MO0QuL2M0mYwM22/download" alt="Usability Graph" width="900">
Figure 2: The usability graph of a sample Zmax recording showing (a) a windowed spectrogram of the EEG Left channel, (b) its usability scores, (c) the normalized acceleration calculated from tri-axial ACC data, (d) a windowed spectrogram of the EEG Right channel, and (e) its usability scores.
 
### Hypnogram
<img src="https://hochschule-rhein-waal.sciebo.de/s/cCDp7AW8q8TdQiY/download" alt="Hypnogram" width="900">
Figure 3: eegFloss outputs of a sample Zmax recording showing spectrograms of (a) EEG Left and (b) EEG Right channels, (c) the normalized acceleration, (d) hypnogram based on the artifact-rejected autoscores, and (e) the mobility labels with TIB bounded by Lights Out and Lights On moments.

**See `sample_data\Zmax\output` for more sample outputs.**

## Reference
Sikder, N., Zerr, P., Krauledat, M., & Dresler, M. *eegFloss: A Python package for refining sleep EEG recordings with machine learning models* (in preparation).

## People
© [Niloy Sikder](https://scholar.google.com/citations?hl=en&user=0ALk5j4AAAAJ&view_op=list_works&sortby=pubdate)<sup>1,2,#</sup>, [Paul Zerr](https://scholar.google.com/citations?hl=en&user=9CldqFoAAAAJ&view_op=list_works&sortby=pubdate)<sup>1,$</sup>, [Matthias Krauledat](https://scholar.google.com/citations?hl=en&user=n9q-wxgAAAAJ&view_op=list_works&sortby=pubdate)<sup>2,$</sup>, & [Martin Dresler](https://scholar.google.com/citations?hl=en&user=Y-hAEQYAAAAJ&view_op=list_works&sortby=pubdate)<sup>1,$</sup>  
<sup>1</sup>Radboud University Medical Center, Donders Institute for Brain, Cognition and Behaviour, Nijmegen, The Netherlands.  
<sup>2</sup>Faculty of Technology and Bionics, Rhine-Waal University of Applied Sciences, Kleve, Germany.  
<sup>#</sup>Developer  
<sup>$</sup>Supervisor  

**This program is provided "as is" without any warranties, express or implied.**  
For questions, assistance, or further information: [contact the developer](mailto:niloy.sikder@donders.ru.nl)

