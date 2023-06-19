from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import csv
import joblib


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
with open('diabetes_prediction_dataset.csv', 'r') as csvfile:
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



# Split the dataset into features (X) and target (y)
X_train, X_test, y_train, y_test = train_test_split(cleaned_data, cleaned_targets, test_size=0.2, random_state=42)

# Split the data into training and testing sets
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Save the trained model to a file
joblib.dump(model, './trained_model.joblib')

