# How-Biased-Information-Environments-Influence-Belief-Updating-and-Long-term-Neural-Dynamics

## Neurocomputational Modeling of Belief Entrenchment in Biased Information Environments (BIEs)
# Overview
This repository contains the MATLAB codebase for investigating how exposure to Biased Information Environments (BIEs) influences belief updating and long-term neural dynamics. By integrating empirical EEG data with recurrent neural population simulations, this framework translates short-term neural and behavioral responses into long-term predictions about cognitive entrenchment across the human lifespan.

The project addresses developmental cognitive risks, specifically comparing the susceptibility of high-plasticity pediatric networks versus stable adult networks to algorithmic echo chambers.

# Repository Structure
The pipeline is divided into four sequential MATLAB scripts:

1_EEG_Processing.m
Phase 1: Feature Extraction and Preprocessing

* Automates the preprocessing of raw EEG recordings using EEGLAB.
* Performs downsampling, bandpass filtering, artifact removal via ICA, and epoching.
* Extracts trial-level neural features including Event-Related Potentials (P300, N200), spectral band power (Delta, Theta, Alpha, Beta, Gamma), and signal complexity (Shannon Entropy).
* Exports normalized feature tables for computational modeling.

2_Model_Construction.m
Phase 2: EEG-Driven Model Calibration

* Constructs an iterative prediction-error learning model driven by the extracted EEG features.
* Iterates through the processed data of 80 subjects across 180 trials each.
* Utilizes non-linear constrained optimization (fmincon) to calibrate empirical parameters, specifically isolating the epistemic weights assigned to visual evidence versus social consensus.
* Generates heatmaps and correlation matrices to validate the relationship between neural activity and behavioral belief shifts.

3_LongTerm_Forecast_Adult.m
Phase 3a: Asymptotic Entrenchment Simulation

* Translates the discrete trial-level updates into a continuous-time recurrent neural network (RNN) model representing adult cortical dynamics.
* Anchors learning rates and synaptic plasticity to the empirically optimized parameters from Script 2.
* Simulates belief trajectories over extended timescales (e.g., 80 years) under varying environmental biases (Neutral, Moderate, Strong).
* Quantifies "Time-to-Entrenchment" and long-term network signal entropy.

4_LongTerm_Sensitivity_Child.m
Phase 3b: Comparative Neurodevelopmental Impact

* Extends the adult baseline model to simulate the heightened neuroplasticity and reduced consolidation characteristic of pediatric brain development.
* Applies a developmental scaling factor to baseline learning rules to simulate increased sensitivity to external inputs.
* Conducts comparative sensitivity analyses to identify the "Tipping Point" where biological plasticity and algorithmic bias intersect, resulting in rapid cognitive radicalization compared to the adult baseline.

# Data Access
This pipeline is designed to analyze empirical EEG data obtained from the OpenNeuro dataset ds004067 (originally published in a study on social influence and belief updating).

To run the preprocessing script (1_EEG_Processing.m), you must first download the raw .vhdr/.eeg files from OpenNeuro and update the data_root directory path within the script.

# System Requirements
MATLAB (R2021a or newer recommended)
EEGLAB (v2021.1 or newer)
MATLAB Optimization Toolbox (Required for fmincon in Script 2)
MATLAB Statistics and Machine Learning Toolbox

# Usage Instructions
To replicate the findings or utilize the modeling framework, scripts must be executed sequentially:
Ensure EEGLAB is added to your MATLAB path.
Run 1_EEG_Processing.m to generate the .mat feature tables from the raw dataset.
Run 2_Model_Construction.m to calculate the optimized population-level parameters.
Run 3_LongTerm_Forecast_Adult.m and 4_LongTerm_Sensitivity_Child.m independently to view the respective longitudinal simulations and generate network visualizations.
