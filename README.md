# csci567
CSCI567 Course Project for H&amp;M Personalized Fashion Recommendations Kaggle Competition: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview

## Get Started
Create a virtual environment and install all required packages.
```
pip3 install virtualenv
virtualenv -p $(which python3) ./venv
source ./venv/bin/activate

# Clone the repository
git clone https://github.com/ShivinDass/csci567.git
cd csci567

# Install the package for namespace
pip3 install -e .
```
Create a data directory in the project folder to store all the training data, submissions etc. None of this will be uploaded on github, 
```
# Download all the csv files and save here
mkdir ./data
export DATA_DIR=/path/to/data
```
After this you can use the utility files and get started with your own code. Create a separate brnach before starting.

### Using Jupyter Notebook
If you want to use a jupyter notebook with the virtualenv then,
```
# Activate the virtualenv
source ./venv/bin/activate

# Install Jupyter notebook
pip install jupyter

# Check whether the correct jupyter is being used. It should show the bin directory in the virtual environment
which jupyter

# Use ipykernel to create a kernel of the virtual environment
pip install ipykernel
python -m ipykernel install --user --name=myvenv
```
After this you can launch ```jupyter notebook``` and choose your kernel when creating a new notebook.

### Two-Tower Training Plots
#### Naive Parameterization (unique embedding per customer and article)
![two tower naive](https://github.com/ShivinDass/csci567/blob/main/figures/two_tower_naive.png)
#### Small Network (feature-wise parameterization)
![two tower small](https://github.com/ShivinDass/csci567/blob/main/figures/two_tower_small.jpeg)
#### Medium Network (feature-wise parameterization)
![two tower medium](https://github.com/ShivinDass/csci567/blob/main/figures/two_tower_medium.jpeg)
#### Large Network (feature-wise parameterization)
![two tower large](https://github.com/ShivinDass/csci567/blob/main/figures/two_tower_large.jpeg)
