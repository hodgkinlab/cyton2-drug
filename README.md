# Effect of Drugs in Murine Cells
$\text{OT-I/Bcl2l11}^{-/-} \text{ CD8}^+$ T cells were stimulated with the following drugs, either individually or in combination:
- Rapamycin (Rapa)
- Mycophenolic Acid (MPA)
- Dexamethasone (Dex)
- Cyclosporine A (CsA)

# File Structure
```bash
root
 ├── requirements.txt   # List of dependencies
 └── code
      ├── fit-*.py      # Main Python scripts for fitting Cyton2 model
      ├── pred-*.ipynb  # Example Jupyter notebooks to generate predictions
      ├── data          # FACS data in Excel format
      ├── src           # Cython code for Cyton2 algorithm & other custom functions to import data and process modelling results
      └── out
           ├── Best-fit Parameters
           │    ├── 1. Single Drugs
           │    ├── 2. Different Timers
           │    ├── 3. Same Timer
           │    └── 4. Complex Interaction
           └── Predictions
                ├── 1. Different Timers
                ├── 2. Same Timer
                └── 3. Complex Interaction
```
Python version: 3.13.2

# Installation
To install dependencies:

```bash
pip install -r requirements.txt
```

Before running the python scripts to fit the Cyton2 model, compile the Cython code by executing:

```bash
cd code/src
python _setup.py build_ext --inplace
```

# Model Fitting Scripts
There are four versions of the model fitting scripts, corresponding to the following experimental cases:

1. `fit-SingleDrugs.py`
2. `fit-DiffTimer.py`
3. `fit-SameTimer.py`
4. `fit-ComplexInter.py`

Each script automatically imports the relevant dataset(s) and initiates fitting procedure using predefined configurations. It enumerates all experimental conditions (i.e. drug concentrations) within each dataset and utilises `multiprocessing` module to fit them in parallel. 

By default, each script generates:
- Two Excel files - one containing model outputs for direct plotting in external software, and another containing a table of fitted parameters.
- One PDF file with summary plots for quick evaluation of the modelling results.

The exact modelling results presented in the paper are located in the `out/Best-fit Parameters` directory.

# Predicting the Combined Effect of Drugs
Three example Jupyter notebooks demonstrate how to read the modelling results from the `out/Best-fit Parameters` directory and generate predictions for the following scenarios:

1. **Different timers** - e.g., division timers from Rapamycin and time to death from Dexamethasone
2. **Same timers** - e.g., division timers from Rapamycin and Mycophenolic Acid
3. **Complex interaction** - e.g., Cyclosporine A and Mycophenolic Acid

Each notebook produces model predictions for all combinations of the provided concentrations of the two drugs.
