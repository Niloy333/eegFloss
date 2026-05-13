![License](https://img.shields.io/github/license/Niloy333/eegFloss?style=flat&color=brightgreen)
[![arXiv](https://img.shields.io/badge/arXiv-2507.06433-b31b1b.svg)](https://arxiv.org/abs/2507.06433)
![Python](https://img.shields.io/badge/Python-3.9.12-blue.svg)
[![Package DOI](https://img.shields.io/badge/Package_DOI-10.5281%2Fzenodo.15823969-blue)](https://doi.org/10.5281/zenodo.15823969)
<!--[![DOI](https://zenodo.org/badge/926585143.svg)](https://doi.org/10.5281/zenodo.15823969) -->
![Repo size](https://img.shields.io/github/repo-size/Niloy333/eegFloss)
![Last commit](https://img.shields.io/github/last-commit/Niloy333/eegFloss)

# *eegFloss*

**— A Python package to *floss out* artifacts from sleep EEG recordings**

<p align="center">
  <img src="https://github.com/Niloy333/eegFloss/blob/base/figures/eegFloss_logo.jpg" alt="eegFloss_logo" width="180">
</p>

**Table of Contents:** [Overview](#overview) | [Installation](#installation) | [Script Descriptions](#script-descriptions) | [Read before Execution](#read-before-execution) | [Primary Artifacts](#primary-artifacts) | [eegUsability Models](#eegusability-models) | [Sample Outputs](#sample-outputs) | [Reference Paper](#reference-paper) | [Cite](#cite) | [People](#people) | [Opportunities for Collaboration](#opportunities-for-collaboration)

## Overview
EEG data often contains artifacts caused by both internal factors (such as device issues) and external influences (such as movement or environmental noise). In sleep research, these artifacts frequently go unnoticed or unaddressed, which can impair the performance and reliability of data-driven models or analyses, especially automatic sleep-stage scoring models,  and reduce the credibility of study outcomes.

Most existing artifact detection methods rely on threshold-based techniques. While easy to implement, these methods often struggle to detect complex or unfamiliar artifacts and typically lack generalizability across datasets.

eegFloss addresses this challenge with *eegUsability*—a machine learning (ML) model designed to detect artifact-contaminated EEG segments in sleep recordings. The model was trained and evaluated on manually artifact-labeled EEG data from 15 participants, collected over 127 nights using the Zmax wearable sleep EEG headband. However, it can be applied to sleep EEG data from any device to detect common artifacts and assess data usability, provided the sleep data adhere to the American Academy of Sleep Medicine (AASM) standards and definitions.

The package also includes *eegMobility*—an ML model that detects the degree of movement throughout the night based on **Zmax** accelerometer data. This information is used to automatically detect Time-in-Bed (TIB). For further details, please refer to the [associated paper](#reference-paper).

eegFloss was developed as part of a doctoral research project at the [Donders Sleep & Memory Lab](https://dreslerlab.org/), within the [Donders Centre for Cognitive Neuroimaging](https://www.ru.nl/en/departments/institutes/donders-centre-for-cognitive-neuroimaging) at Radboud University (Nijmegen, The Netherlands), in collaboration with Hochschule Rhein-Waal (Kleve, Germany) and Radboud University Medical Center (Radboudumc; Nijmegen, The Netherlands).

## Installation
It is recommended to use eegFloss within a dedicated Anaconda or Miniconda environment. Follow these steps:

1. Download and install [Anaconda or Miniconda](https://www.anaconda.com/download/success) for your operating system.
2. [Download eegFloss](https://github.com/Niloy333/eegFloss/archive/refs/heads/base.zip), extract the compressed file, and place it in a suitable and accessible directory (with writing permission).
3. On Linux, ensure that the appropriate graphics driver is installed and hardware acceleration is enabled.
4. Launch the Anaconda Prompt:
   - **Windows**: Search for "Anaconda Prompt" in the Start menu.
   - **Linux**: Open a terminal and run:<br>
   `source ~/anaconda3/bin/activate` or `source ~/miniconda3/bin/activate`
   - **macOS**: Open a terminal.
5. In the prompt, navigate to the extracted `eegFloss` directory (the directory of the `.py` files). Example:<br>
   `cd D:\Folder1\eegFloss-base\`
6. Create a new environment named `eegFloss` with all the necessary packages:<br>
   `conda env create --name eegFloss --file eegFloss_dependencies.yml -y`<br>
   You may need to provide an administrative password.
7. Once the environment is created, activate it:<br>
   `conda activate eegFloss`
8. Set `PYTHONWARNINGS` before launching Spyder to suppress noisy `pkg_resources` deprecation warning and the Python 3.9 end-of-life warnings:<br>
   `set PYTHONWARNINGS=ignore:pkg_resources is deprecated as an API:UserWarning,ignore:You are using a Python version 3.9 past its end of life:FutureWarning`
9. To start coding:
   - Launch Spyder and manually open the scripts:<br>
     `spyder`
   - Or, use your preferred code editor and run the script from the Anaconda Prompt. Example:<br>
     `python 1.eegFloss_check_usability_mobility.py`

## Script Descriptions

### `1.eegFloss_check_usability_mobility.py`
- Detects artifacts in sleep EEG data using a chosen [eegUsability model](#eegusability-models).
- Aggregates the provided sleep scores with *usability scores* (the outcomes of artifact detection) using a majority rule (an epoch is marked unusable if more than half of its constituent segments are unusable) to generate *artifact-rejected sleep scores*.
- Automatically identifies *Lights Out* and *Lights On* moments using a chosen *eegMobility* model and computes Time-in-Bed (TIB).
- Computes common sleep statistics based on the artifact-rejected sleep scores and TIB.
- Generates visualizations such as *usability graphs* (shows channel-wise data usability) and *hypnograms* (shows the overall outcomes) to better illustrate the model's outputs.

Here is a streamlined overview of the script’s internal workflow:
<img src="https://github.com/Niloy333/eegFloss/blob/base/figures/package_workflow.png" alt="script workflow" width="600"> 

>Figure 1: A simplified workflow of 1.eegFloss_check_usability_mobility.py.

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

- eegFloss currently supports only European Data Format (EDF) and BioSemi Data Format (BDF) files. Therefore, raw EEG signal(s) must be stored in the EDF/BDF format.
- If your data is in a different format, check whether the associated software suite of your recording device allows exporting as or converting data to EDFs.
- If not, you can manually convert data using Python libraries such as [PyEDFlib](https://pyedflib.readthedocs.io/en/latest/) or [MNE](https://github.com/mne-tools/mne-python).

### Sleep-Stage Scoring
- eegFloss **does not include** a built-in automatic sleep-stage scorer and cannot infer sleep stages from EEG or other signals.
- However, if you provide sleep scores alongside your data, it can generate artifact-rejected sleep scores by combining the provided sleep scores and the detected data usability.
- If your data is not manually scored, consider using an open-source automatic sleep scorer, such as [U-Sleep](https://sleep.ai.ku.dk/), [YASA](https://yasa-sleep.org/), [SomnoBot](https://somnobot.fh-aachen.de/), or another similar model using [SleepyLand](https://github.com/biomedical-signal-processing/sleepyland) (for PSG data), or [Dreamento](https://github.com/dreamento/dreamento) or [ezscore-f](https://github.com/coonwg1/ezscore) (for Zmax data).

### Sleep Score Format
- Sleep stages are expected to be labeled as 0 = Wake, 1 = N1, 2 = N2, 3 = N3, and 4 or 5 = REM. Deviating from this convention will result in incorrect visualizations and sleep statistics.
- The sleep scores must be stored in the first column of a TXT/CSV file located in the same directory as the corresponding EDF file. The number of epochs must match the recording duration.

### Data Organization
- Each recording should reside in a separate directory. Placing multiple recordings in the same directory will result in only one file being processed by eegFloss, with the rest ignored. Provide the parent directory as the `Raw_Data_Dir`.

### Output Directory Management
- While it is possible to save eegFloss outputs in the same directory as the recordings by setting `Output_Dir = Raw_Data_Dir`, this is not recommended.
- eegFloss checks for prior outputs to skip redundant processing. So, consider saving the additional outputs, even if they are not needed for your analysis.
- If you later modify key settings (e.g., change the usability model or adjust TIB thresholds), the tool may incorrectly skip reprocessing due to existing outputs. To prevent this, use a separate output directory for each round of processing.
- Feature extraction is typically the most time-consuming step (unless using a ‘lite’ model). To avoid recomputation, copy the `eegFloss_stat_features.npz` files to the data directory alongside the EDF files after processing the data once.

### TIB Detection and Accelerometer Requirements
- Automatic TIB detection using the eegMobility model is validated only for Zmax data. The model's training data is publicly available at [Kaggle.com/datasets/niloy333/eegmobility-dataset](https://doi.org/10.34740/kaggle/dsv/12525688).
- If you want to test it for another device, ensure that the tri-axial accelerometer data is measured in units of g, falls within a range of ±2g (clip extreme values if needed), and includes gravitational acceleration (meaning the normalized data should center around 1g).
- Analyzing EEG data without accompanying accelerometer signals may lead to the removal of some arousals due to a lack of motion information.

### Miscellaneous
- Make sure your device is connected to the internet before executing the scripts.
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
  ```
- On non-Conda environments, if you face issues with missing packages, try installing the packages listed on `eegFloss_dependencies.yml` manually in the given order using `pip install [package_name]==[version]`.

## Primary Artifacts
eegUsability detects the following artifacts in raw sleep EEG data:
<img src="https://github.com/Niloy333/eegFloss/blob/base/figures/noise_classes_port.png" alt="primary_artifacts" width="600">

>Figure 2: (a) A windowed spectrogram (blue: low power, red: high power) of a sample Zmax EEG channel, highlighting segments containing different artifacts. The corresponding time-domain representations of these segments are shown for (b) Good Data, (c) No Data, (d) High Noise, (e) Spiky Noise, and (f) M-shaped Noise.

## eegUsability Models

| eegUsability version         | Feature set(s)              | Specialty                                                                                         | When to use                                                                                       | F1-score (%)      | Processing time (8-hr night)<sup>†</sup> |
|-----------------------------|-----------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------|-------------------------------|
| “v1.0” or “default”         | Spectrogram and statistical | Combines two feature sets for consistent outputs. Tested across datasets and the most dependable. | Best for general tasks requiring maximum data retention.                                         | 84.87             | ≈35 sec                      |
| “v0.8” or “weighted-M”      | Spectrogram and statistical | Better at identifying M-shaped Noise but sacrifices a bit more usable data.                      | Ideal when M-shaped Noise detection is crucial and slight data loss is acceptable.              | 86.35             | ≈36 sec                      |
| “v0.6” or “binary”          | Spectrogram and statistical | Only identifies whether the data is usable or not; does not differentiate noise types.           | If noise differentiation is entirely unnecessary or processing simplicity is prioritized.       | 89.4              | ≈35 sec                      |
| “v0.7” or “lite”            | Spectrogram                 | Uses only one feature set; similar to v1.0, but 12 times faster with comparable results.          | Suitable for quick results where minor inconsistencies are tolerable.                           | 84.94             | ≈3 sec                       |
| “v0.7.2” or “lite weighted-M” | Spectrogram               | Similar to v0.8, but works on only spectrogram features, hence is faster.                         | Optimal for quick, precise outputs.                                                              | 86.37             | ≈3 sec                       |
| “v0.7.3” or “lite binary”   | Spectrogram                 | Similar to v0.6 but works on only spectrogram features, hence is faster.                          | Handy when fast results are needed without noise type differentiation.                          | 89.29             | ≈3 sec                       |
| “v0.9” or “full”            | Spectrogram and statistical | Similar to v1.0 but is trained on the entire available dataset.                                   | Can be used if a more hypertuned model is needed.                                               | 90.11<sup>^</sup>            | ≈37 sec                      |

><sup>^</sup>Results are from a test set that is a subset of the training data.<br><sup>†</sup>Tested on a Core i7, 8C/16T, 2.5–4.8 GHz processor with no resource-intensive processes running in parallel.

## Sample Outputs

### Usability Graph
<img src="https://github.com/Niloy333/eegFloss/blob/base/figures/fig%206.png" alt="usability_graph_zmax" width="1000">

>Figure 3: The usability graph of a sample Zmax recording showing (a) a windowed spectrogram of the EEG Left channel, (b) its usability scores, (c) the normalized acceleration calculated from tri-axial ACC data, (d) a windowed spectrogram of the EEG Right channel, and (e) its usability scores.
 
### Hypnogram
<img src="https://github.com/Niloy333/eegFloss/blob/base/figures/fig%208.png" alt="hypnogram_zmax" width="1000">

>Figure 4: eegFloss outputs of a sample Zmax recording showing spectrograms of (a) EEG Left and (b) EEG Right channels, (c) the normalized acceleration, (d) hypnogram based on the artifact-rejected autoscores, and (e) the mobility labels with TIB bounded by Lights Out and Lights On moments.

**The `sample_output` folder contains a sample of all output files.**

## Reference Paper
More information on this package and the underlying models can be found in:

Sikder, N., Zerr, P., Jafarzadeh Esfahani, M., Dresler, M., & Krauledat, M. (2025). *eegFloss*: A Python package for refining sleep EEG recordings using machine learning models. arXiv. [https://doi.org/10.48550/arXiv.2507.06433](https://doi.org/10.48550/arXiv.2507.06433).

[Read on ResearchGate](https://www.researchgate.net/publication/393539640_eegFloss_A_Python_package_for_refining_sleep_EEG_recordings_using_machine_learning_models)

## Cite
If you find this package helpful and use it in your work, please cite the reference paper as:

```bibtex
@article{sikder2025eegfloss,
  title     = {eegFloss: A Python package for refining sleep EEG recordings using machine learning models},
  author    = {Sikder, Niloy and Zerr, Paul and Jafarzadeh Esfahani, Mahdad and Dresler, Martin and Krauledat, Matthias},
  journal   = {arXiv preprint arXiv:2507.06433},
  year      = {2025},
  doi       = {10.48550/arXiv.2507.06433},
  url       = {https://arxiv.org/abs/2507.06433},
}
```

And cite the package as:

```bibtex
@software{sikder2025eegflossv1,
  author    = {Niloy Sikder},
  title     = {eegFloss},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15823969},
  url       = {https://doi.org/10.5281/zenodo.15823969},
}
```

## People
© [Niloy Sikder](https://scholar.google.com/citations?hl=en&user=0ALk5j4AAAAJ&view_op=list_works&sortby=pubdate)<sup>1,2,#</sup>, [Paul Zerr](https://scholar.google.com/citations?hl=en&user=9CldqFoAAAAJ&view_op=list_works&sortby=pubdate)<sup>1,$</sup>, [Martin Dresler](https://scholar.google.com/citations?hl=en&user=Y-hAEQYAAAAJ&view_op=list_works&sortby=pubdate)<sup>1,$</sup>, & [Matthias Krauledat](https://scholar.google.com/citations?hl=en&user=n9q-wxgAAAAJ&view_op=list_works&sortby=pubdate)<sup>2,$</sup>   
<sup>1</sup>Radboud University Medical Center, Donders Institute for Brain, Cognition and Behaviour, Nijmegen, The Netherlands.  
<sup>2</sup>Faculty of Technology and Bionics, Rhine-Waal University of Applied Sciences, Kleve, Germany.  
<sup>#</sup>Developer  
<sup>$</sup>Supervisor

## Opportunities for Collaboration
**eegFloss can currently process only sleep EEG data**, and so far, it's been **validated only on recordings from the Zmax headband**. However, the package offers many exciting opportunities for improvement and expansion, including:
- Improving the artifact detection model with additional training data
- Integrating an automatic sleep-stage scorer to create a complete sleep analysis pipeline
- Validating performance on non-Zmax datasets
- Extending support to artifact detection in wake EEG
- Developing a user-friendly Graphical User Interface (GUI)

We are exploring some of these work packages, but we can achieve much more through active collaboration. If this sounds interesting, and you would like to get involved, please feel free to reach out.

**This package is provided *as is*, without any warranties, express or implied. eegFloss is released under the MIT License and is free to use, modify, and integrate with other software, provided that appropriate credit is given.**

For questions, assistance, suggestions, or further information: [contact the developer](mailto:niloy.sikder@donders.ru.nl).
