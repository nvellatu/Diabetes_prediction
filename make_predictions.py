import joblib
from Diabetes_prediction import convert_to_numeric


model = joblib.load('trained_model.joblib')

#Predict if a patient has diabetes by entering new_data in this format:
    #[gender(male/female),age,hypertension(0/1),heart_disease(0/1),smoking_history(never/ever/former/current/not current/no info),
    # bmi,HbA1c_level,blood_glucose_level]

#Input data directly
# new_data = ['Female',67.0,0,0,'never',63.48,8.8,155]

# OR Ask user for individual data
print("Enter the following data: ")
gender = input("Gender (male/female): ")
age = input("Age: ")
hypertension = input("Does patient have Hypertension (1 for yes/0 for no): ")
heartdisease = input("Does patient have Heart Disease (1/0): ")
smoking_history = input("Patient smoking history (never/ever/former/current/not current/no info)")
bmi = input("BMI: ")
Hba1c_level = input("HbA1c_level: ")
blood_glucose_level = input("Blood Glucose Level: ")
new_data = [gender, age, hypertension, heartdisease, smoking_history, bmi, Hba1c_level, blood_glucose_level]

#process data and make prediction
preprocessed_data = [convert_to_numeric(value) for value in new_data]

predictions = model.predict([preprocessed_data])

print(predictions[0])

if predictions[0] == 1: print('Patient has Diabetes') 
else: print('Patient does not have Diabetes.')