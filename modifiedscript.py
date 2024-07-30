import os
import pandas as pd

# Define the base directory relative to the script location
base_dir = os.path.dirname(__file__)

# Define the paths to the data files using relative paths
training_file_path = os.path.join(base_dir, 'data', 'twitter_training.csv')
validation_file_path = os.path.join(base_dir, 'data', 'twitter_validation.csv')

# Load the datasets
training_data = pd.read_csv(training_file_path)
validation_data = pd.read_csv(validation_file_path)

# Print the columns of the training data
print("Columns in Training Data:")
print(training_data.columns)

# Print the first few rows to inspect the data
print("\nFirst few rows of Training Data:")
print(training_data.head())

# Print the columns of the validation data
print("\nColumns in Validation Data:")
print(validation_data.columns)

# Print the first few rows to inspect the data
print("\nFirst few rows of Validation Data:")
print(validation_data.head())
