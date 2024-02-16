import random
# from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import csv
import joblib
import numpy as np

def convert_to_numeric(value):
    # if value.isdigit():
    #     return int(value)
    try:
        return float(value)
    except ValueError:
        if value.lower() == 'male':
            return 1
        elif value.lower() == 'female':
            return 0
        elif value.lower() == 'never':
            return 0
        elif value.lower() == 'current':
            return 1
        elif value.lower() == 'former':
            return 2
        elif value.lower() == 'ever':
            return 3
        elif value.lower() == 'not current':
            return 4
        elif value.lower() == 'no info':
            return -1
        else:
            return value
        

# Load the dataset
inputs = []
targets = []
with open('AI\Diabetes_prediction\diabetes_prediction_dataset.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row
    for row in reader:
        converted_row = [convert_to_numeric(value) for value in row[:-1]]
        inputs.append(converted_row)
        target = convert_to_numeric(row[-1])
        targets.append(target)

# Remove rows with non-numeric values from data and targets
cleaned_data = []
cleaned_targets = []
for i in range(len(inputs)):
    if all(isinstance(value, (int, float)) for value in inputs[i]) and isinstance(targets[i], (int, float)):
        cleaned_data.append(inputs[i])
        cleaned_targets.append(targets[i])

# Separate diabetes and non_diabetes data
diabetes_indices = [index for index, value in enumerate(cleaned_targets) if value == 1.0]
non_diabetes_indices = [index for index, value in enumerate(cleaned_targets) if value == 0]
print("Ratio of Patients without Diabetes : Total Data length",len(non_diabetes_indices)/len(cleaned_data))
non_diabetes_indices = [index for index, value in enumerate(cleaned_targets) if value == 0][:len(diabetes_indices)]
# Balance data
mixed_indices = diabetes_indices + non_diabetes_indices
random.seed(510)
random.shuffle(mixed_indices)

balanced_data = [cleaned_data[index] for index in mixed_indices]
balanced_targets = [cleaned_targets[index] for index in mixed_indices]

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(balanced_data)

# Split the dataset into features (X) and target (y)
X_train, X_test, y_train, y_test = train_test_split(normalized_data, balanced_targets, test_size=0.2, random_state=42)

# Split the data into training and testing sets
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Save the trained model to a file
joblib.dump(model, './trained_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

