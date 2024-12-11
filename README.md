# LofreeCP

LofreeCP repository consists of two main files: `run_generation.py` and `run_cp.py`. The goal of the project is to first sample model responses using `run_generation.py`, and then perform conformal prediction with the generated responses using `run_cp.py`.

## Overview

### Workflow
1. **First Step: Sampling Responses**
    - Run `run_generation.py` to generate and sample responses from a machine learning model. This will output a set of model responses based on a given prompt or input.
  
2. **Second Step: Conformal Prediction**
    - After collecting all the responses, run `run_cp.py` to apply conformal prediction. This step uses the responses generated in the first step to calculate confidence intervals and assess the reliability of the model predictions.

### Key Files
- **`run_generation.py`**: This script is used for generating and sampling responses from the model based on inputs.
  
- **`run_cp.py`**: This script takes the generated responses and applies conformal prediction methods to quantify uncertainty in the model's predictions.
