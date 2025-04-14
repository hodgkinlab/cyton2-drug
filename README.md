# Effect of drugs in murine cells

OT-I/Bcl2l11$^{-/-}$ CD8$^+$ T cells

# File structure

```bash
root
├── requirements.txt  # List of dependencies
├── fit-*.py          # Scripts for fitting Cyton2 model
├── pred-*.ipynb      # Jupyter notebooks to generate predictions
└── code
     ├── data
     ├── src
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

Install dependencies:

```bash
pip install -r requirements.txt
```

Prior to running the python scripts to fit the Cyton2 model, run the following commands to compile the Cython code:

```bash
cd code/src
python _setup.py build_ext --inplace
```

There are four versions of the model fitting scripts, corresponding to the following cases:

1. `fit-SingleDrugs.py`
2. `fit-DiffTimer.py`
3. `fit-SameTimer.py`
4. `fit-ComplexInter.py`

Each script automatically imports the relevant data files and initiates fitting procedure using predefined configurations. It enumerates all conditions (i.e. drug concentrations) provided in each dataset and utilises `multiprocessing` to fit them in parallel. By default, the script will generate two Excel files-one containing model outputs for direct plotting in external software, and another with a table of fitted parameters-and one PDF file containing summary plots to quickly evaluate the modelling results.

The exact modelling results presented in the paper are provided in the `out/Best-fit Parameters` folder.

# Predicting the combined effect of drugs

There are three example Jupyter notebooks that read the modelling results from the `out/Best-fit Parameters` folder and generate predictions for the following scenarios: 

1. **Different timers** - e.g., division timers from Rapamycin and time to death from Dexamethasone
2. **Same timers** - e.g., division timers from Rapamycin and Mycophenolic Acid
3. **Complex interaction** - e.g., Cyclosporine A and Mycophenolic Acid

Each notebook produces model predictions for all combinations of the provided concentrations of the two drugs.
