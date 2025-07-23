# CreepEventRheology

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Tags**: `fault mechanics` · `aseismic slip` · `creep events` · `rheology` · `geophysics` · `time series analysis`

This repository models the shape of creep events on faults using different rheological formulations. It evaluates the fit of each model and quantifies model selection uncertainty using information-theoretic criteria (e.g., Akaike Information Criterion, AIC).

The approach is particularly suited for studying aseismic slip transients and understanding near-surface fault mechanics.

## Features

- Fits multiple rheological models (linear viscous, power-law, and rate-and-state inspired forms)
- Computes AIC and delta-AIC for model comparison
- Flags events with model selection uncertainty
- Uses bootstrapping to assess the robustness of model support
- Includes plotting utilities for visualising model fits and confidence intervals

## Input Data

- **Creep events CSV**: Metadata for detected slip transients (e.g., timing, duration)
- **Creepmeter data `.txt` files**: Displacement time series from fault-monitoring instruments

## Usage

This project is primarily designed for use in **Jupyter notebooks**. The typical workflow is:

1. Load creep event metadata and creepmeter time series
2. Fit multiple rheological models to each event
3. Evaluate model performance using AIC
4. Use built-in functions to flag model uncertainty and summarise support
5. Visualise fits and statistical summaries
