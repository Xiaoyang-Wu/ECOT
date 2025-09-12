# Unified Conformalized Multiple Testing - Implementation

This repository contains the implementation code for the paper:

**"Unified Conformalized Multiple Testing with Full Data Efficiency"**

The code is organized into two main parts: **simulation** and **realdata**.

---

## Repository Structure


---

## Simulation

- **`simulation/main_paper/`**  
  Contains code to reproduce simulations presented in the main text of the paper.

- **`simulation/supplementary/`**  
  Contains code to reproduce simulations presented in the supplementary materials.

---

## Real Data

- **`realdata/`** folder contains code for five different datasets.  
- Each dataset has its own subfolder (`dataset1`, `dataset2`, ..., `dataset5`) with scripts to reproduce the corresponding analyses.

---

## Requirements

- Python ≥ 3.8 (or R depending on implementation)  
- Git LFS for large files: [https://git-lfs.github.com](https://git-lfs.github.com)  
- Standard data science packages (numpy, pandas, scipy, etc.)

---

## Usage

1. Clone the repository:

```bash
git clone git@github.com:Xiaoyang-Wu/ECOT.git
git lfs pull   # Pull large files managed by Git LFS

