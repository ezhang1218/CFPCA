# Contrastive Functional Principal Components Analysis (CFPCA)

# Overview
This repository hosts the code and data on CFPCA, a novel method designed to spotlight low-dimensional structures unique to or enriched in the foreground dataset relative to the background counterpart.

As functional data assumes a central role in contemporary data analysis, the search for meaningful dimension reduction becomes critical due to its inherent infinite-dimensional structure. Traditional methods, such as Functional Principal Component Analysis (FPCA), adeptly explore the overarching structures within the functional data. However, these methods may not sufficiently identify low-dimensional representations that are specific or enriched in a foreground dataset (case or treatment group) relative to a background dataset (control group). This limitation becomes critical in scenarios where the foreground dataset, such as a specific treatment group in biomedical applications, contains unique patterns or trends that are not as pronounced in the background dataset. Addressing this gap, we propose Contrastive Functional Principal Component Analysis (CFPCA). We supplement our method with theoretical guarantees on CFPCA estimates supported by multiple simulations. Through a series of applications, CFPCA successfully identifies these foreground-specific structures, thereby revealing distinct patterns and trends that traditional FPCA overlooks.

# Repository Structure
## data
This folder contains the gait cycle data used in [gait_cycle.ipynb](./gait_cycle.ipynb).

## simulations
This folder contains each simulation. Within each file, we indicate which figures correspond to those in the paper.

## berkeley_growth_study.ipynb
This notebook demonstrates the application of CFPCA on the Berkeley Growth Study dataset, a toy dataset that has height measurements for boys and girls over various ages.

## gait_cycle.ipynb
This notebook utilizes CFPCA to analyze the gait cycle dataset.

## stock_market.ipynb
This notebook utilizes CFPCA to analyze stock market data.

## main.py
This script contains the core CFPCA method along with other supporting functions used.
