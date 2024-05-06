# The Higgs Boson Machine Learning Challenge

__Team Member__: Robin Jaccard, Mathieu Despond, Eva Luvison

The Higgs Boson Machine Learning Challenge is a classification challenge where the goal is to predict whether a particle is a Higgs Boson or not. In order to solve this problem, we are using real data from the CERN particle accelerator. We implemented a machine learning model to solve this challenge.

## Usage

### Dependencies

#### Main
* Python 
* Numpy 
* Jupyter Notebook

#### Visualization
* Matplotlib 
* Seaborn 

#### Data
The data must be stored in a "data" folder at the root (same level as run.py). Those files must be stored as zip, exactly as in the repository.

### Submission file
To create a submission file for aicrowd.com, you must go in the scripts directory and run `python3 run.py` 

After running the command a file called `result.csv` will be created, and can directly be uploaded to aicrowd.com to make a submission.

## Files
The different files are :
* `ExploratoryDataAnalyis.ipynb` : allows us to visualized the datas and features distribution to then choose which feature to remove and where were the outliers to handle.
* `helpers.py` : contains the functions used load the datasets, give labels and predictions, compute accuracy, create final csv.
* `processing.py`:  contains the functions used to clean the dataset and preprocess it: split it in 6, delete the NaN values, apply transformations, poly-expansion, add bias and hadle outliers.
* `implementations.py` : contains the implementation of the 6 machine learning models asked for this project.
* `GridSearch.ipynb`: implements the different version of gridsearch for each method.
* `cross_validation.py` : contains the functions to perform a cross-validation.
* `run.py` : is used to get the .csv submission file, using the best hyper-parameters found for the chosen function.
