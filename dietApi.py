label_column_name = 'VegNovVeg'  # TODO: Replace with your actual label column name

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from flask import jsonify
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report,mean_absolute_error

import utils.bmi as bmi
# Data Imports
data=pd.read_csv('dataset/food.csv')
dishes = pd.read_csv('dataset/dishes.csv')
diabetic_data=pd.read_csv('dataset/diabetic.csv')
bodyfat_data=pd.read_csv('dataset/bodyfatbmi.csv')
healthcare_data = pd.read_csv('dataset/healthcaredataset.csv')
diet_dataset=pd.read_csv('dataset/diet_dataset.csv')
cardio_train=pd.read_csv('dataset/cardio_train.csv')
## Data Cleaning process

healthcare_data_cleaned = healthcare_data.dropna(subset=['bmi'])
healthcare_data_cleaned['age'] = healthcare_data_cleaned['age'].astype('int')
healthcare_data_cleaned['gender'] = healthcare_data_cleaned['gender'].map({'Male': 1, 'Female': 0})
healthcare_data_cleaned['smoking_status'] = healthcare_data_cleaned['smoking_status'].map({'Unknown': -1, 'formerly smoked': 1,'never smoked':0,'smokes':2})
diet_dataset['Gender']=diet_dataset['Gender'].map({'M': 1, 'F': 0})
diet_dataset['Exercise'] = diet_dataset['Exercise'].map({'Lunge': 0, 'Resistance training': 1 ,'Low Cardio':2,'Medium Cardio':3,'High Cardio':4,'Running':5,'Weightlifting':6,
                                                         'Gentle Impact Wellness for Weight Loss':7,"Personalized Fusion Fitness for Weight Loss":8,
                                                         "150/Week Fusion Fitness for Weight Loss":9,"Tailored Fitness Fusion":10,
                                                         "Custom Blend Fitness Journey":11,"Balanced Fitness Fusion":12,
                                                         "Moderate Fitness Fusion for Weight Loss":13})
# Converting 'VegNonVeg' column to binary (0 for Veg, 1 for NonVeg)
data['VegNovVeg'] = data['VegNovVeg'].map({'Veg': 0, 'NonVeg': 1})




# Saving Cleaned Data for other purposes maybe 
healthcare_data_cleaned.to_csv('./dataset/cleaned/healthcaredataset_cleaned.csv', index=False)
diet_dataset.to_csv('./dataset/cleaned/diet_dataset_cleaned.csv', index=False)
def show_entry_fields(data):
    return "Age: %s  Veg-NonVeg: %s Weight: %s kg Hight: %s cm" % (data['age'], data['vegnveg'],data['weight'], data['height'])

############ UNTIL HERE , EVERYTHING IS WORKING FINE ################
##################################### ALGOS HERE #######################################
############### MODEL PREDICTION FOR DIABETES
scaler1 = StandardScaler()
X_normalized = scaler1.fit_transform(diabetic_data[['BMI', 'Height', 'Weight', 'AgeYears']])
import gc
gc.collect()
y = diabetic_data['is_diabetic']

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

model_d = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_d.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_d.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
loss, accuracy = model_d.evaluate(X_test, y_test)
print(f"Model Diabeties Accuracy: {accuracy}")
y_pred_d = model_d.predict(X_test)
y_pred_d = (y_pred_d > 0.5).astype(int)  # Convert probabilities to binary predictions

# Create confusion matrix
conf_matrix_d = confusion_matrix(y_test, y_pred_d)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_d, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Diabetic', 'Diabetic'],
            yticklabels=['Not Diabetic', 'Diabetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Diabetes Prediction')
plt.show()

############### MODEL PREDICTION FOR HeartDisease / Hypertension
scaler3 = StandardScaler()
# Fit the scaler with BMI and age data
x2 = healthcare_data_cleaned[['bmi','age']]
scaler3.fit(x2)
healthcare_scaled = scaler3.transform(x2)

# Prepare the target column
y2 = healthcare_data_cleaned['heart_disease']

# Split the data
X_train2, X_test2, y_train2, y_test2 = train_test_split(healthcare_scaled, y2, test_size=0.2, random_state=42)

# Define the model
model_h = Sequential([
    Dense(128, activation='relu', input_shape=(X_train2.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    
])
# Compile the model with binary_crossentropy
model_h.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_h.fit(X_train2, y_train2, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model_h.evaluate(X_test2, y_test2)


print(f"Model Heart Disease Accuracy: {accuracy}")

print("Confusion Matrix")
y_pred = model_h.predict(X_test2)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

# Create confusion matrix
conf_matrix = confusion_matrix(y_test2, y_pred)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Heart Disease', 'Heart Disease'],
            yticklabels=['No Heart Disease', 'Heart Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
############### MODEL PREDICTION For Blood presure

scaler5 = StandardScaler()
x5 = cardio_train[['age', 'gender', 'height', 'weight', 'smoke']]
# Systolic blood pressure

scaler5.fit(x5)
cardio_scaled = scaler5.transform(x5)
y5 = cardio_train['ap_hi']

X_train5, X_test5, y_train5, y_test5 = train_test_split(cardio_scaled, y5, test_size=0.2, random_state=42)

model_bp = Sequential([
    Dense(256, activation='relu', input_shape=(X_train5.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # No activation function for regression
])

model_bp.compile(optimizer='adam', loss='mean_squared_error')
model_bp.fit(X_train5, y_train5, epochs=20, batch_size=32, validation_data=(X_test5, y_test5))
loss = model_bp.evaluate(X_test5, y_test5)

print(f"Model Loss (Mean Squared Error): {loss}")

###########################################################################
############### MODEL Fitness Goal / Exercise Recommendation / BMR Prediction
scaler4 = StandardScaler()
# Fit the scaler with BMI and age data
x3 = diet_dataset[['BMI','Age','Gender']]
scaler4.fit(x3)
diet_dataset_scaled = scaler4.transform(x3)

# Prepare the target column
y3 = diet_dataset['BMR']

# Split the data
X_train3, X_test3, y_train3, y_test3 = train_test_split(diet_dataset_scaled, y3, test_size=0.2, random_state=42)

# Define the model
model_b = Sequential([
    Dense(128, activation='relu', input_shape=(3,)),  # Assuming 3 input features: BMI, Age, Gender
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # Linear activation for regression
])
model_b.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # Use mean squared error for regression
model_b.fit(X_train3, y_train3, epochs=10, batch_size=32)
loss, mae = model_b.evaluate(X_test3, y_test3)
y_pred_b = model_b.predict(X_test3)

# Calculate Mean Absolute Error
mae_b = mean_absolute_error(y_test3, y_pred_b)
print(f"Model BMR Prediction MAE: {mae_b}")

# Visualize predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test3, y_pred_b, color='blue', alpha=0.5)
plt.xlabel('Actual BMR')
plt.ylabel('Predicted BMR')
plt.title('Actual vs. Predicted BMR')
plt.show()


# Display classification reports
print("Classification Report:\n", classification_report(y_test, y_pred_d))
print("Classification Report:\n", classification_report(y_test2, y_pred))
############################## Predicting diabetes probability #####################################
##################################################################

#######################################################################
def predict_diabetes_probability(user_data,bmii):
    # Assuming user_data is a dictionary with keys like 'BMI', 'Height', 'Weight', 'Age'
    # Extract and order the values as expected by the model
    user_features = [bmii, user_data['height'], user_data['weight'], user_data['age']]

    
    # Convert to the format expected by the scaler (e.g., a 2D array)
    user_features_array = [user_features]  # This creates a 2D array-like structure

    # Normalize the input
    normalized_input = scaler1.transform(user_features_array)

    # Make the prediction
    probability = model_d.predict(normalized_input)
    print("THE PROBABILITY OF BEING DIABETIC", probability[0][0] * 100, "%")
    return probability[0][0]
############################## Predicting Heart disease probability #####################################
def predict_heartdisease_probability(user_data,bmii):
    # Assuming user_data is a dictionary with keys like 'BMI', 'Height', 'Weight', 'Age'
    # Extract and order the values as expected by the model
    user_features = [bmii , user_data['age']]
    
    # Convert to the format expected by the scaler (e.g., a 2D array)
    user_features_array = [user_features]  # This creates a 2D array-like structure

    # Normalize the input
    normalized_input = scaler3.transform(user_features_array)

    # Make the prediction
    probability = model_h.predict(normalized_input)
    print("THE PROBABILITY OF Having a heart attack", probability[0][0] * 100, "%")
    return probability[0][0]
##################################################################
def predict_bmr(user_data,bmii):
    # Assuming user_data is a dictionary with keys like 'BMI', 'Height', 'Weight', 'Age'
    # Extract and order the values as expected by the model
    user_features = [bmii , user_data['age'],user_data['gender']]
    # Convert to the format expected by the scaler (e.g., a 2D array)
    user_features_array = [user_features]  # This creates a 2D array-like structure

    # Normalize the input
    normalized_input = scaler4.transform(user_features_array)
    print("Normalized Input:", normalized_input)
    # Make the prediction
    bmr = model_b.predict(normalized_input)
    print("The value of your BMR is ", bmr[0][0] )
    return bmr[0][0]
##################################################################
def suggest_ideal_weight(height_cm):
    healthy_bmi_lower = 18.5
    healthy_bmi_upper = 24.9
    height_m = float(height_cm) / 100 
    # Calculate the ideal weight for the user
    ideal_weight_lower = healthy_bmi_lower * (height_m ** 2)
    ideal_weight_upper = healthy_bmi_upper * (height_m ** 2)
    return (round(ideal_weight_lower,2), round(ideal_weight_upper,2))
# print(f"The ideal weight range for a height of {height_cm} cm is between {ideal_weight_range[0]} kg and {ideal_weight_range[1]} kg.")

#####################################################################""
def predict_bloodpresure(user_data):
    # Ensure that the order and number of features match the model's training data
    user_features = [user_data['age'], user_data['gender'], user_data['height'], user_data['weight'], user_data['smoking_status']]  

    # Convert to a 2D array for the scaler
    user_features_array = [user_features]

    # Normalize the input
    normalized_input = scaler5.transform(user_features_array)


    # Make the prediction
    bloodpresure = model_bp.predict(normalized_input)
    print("The estimated value of your blood pressure is ", bloodpresure[0][0])
    return bloodpresure[0][0]

def Healthy(user_data):

    # Assuming bmiCalculation function returns a tuple, and the first element is the BMI value
    val,_,_,_,health_status,bmiValue = bmi.bmiCalculation(user_data)

    # Filter based on Veg-NonVeg preference
    veg_nonveg_filter = user_data['vegnveg']
    filtered_food = data[data['VegNovVeg'] == veg_nonveg_filter]
    # Further filtering based on BMI
    if bmiValue >= 25:  # BMI >= 25 is considered overweight
        print("Looking for the best healthy diet for you !")
        filtered_food = filtered_food[filtered_food['Calories'] < filtered_food['Calories'].median()]
        print("filtered food \n",filtered_food)


    # Get the food items from the filtered list
    filtered_food_items = filtered_food['Food_items'].tolist()
    # Find dishes that contain these food items
    recommended_dishes = dishes[dishes['Food_items'].isin(filtered_food_items)]['Dish'].tolist()
    print(recommended_dishes)
    diabetes_probability = predict_diabetes_probability(user_data, bmiValue)
    # bodyfact_probability=predict_bodyfat_percentage(bmiValue)
    heartdisease_probability=predict_heartdisease_probability(user_data, bmiValue)
    # Convert the NumPy float32 to a native Python float
    diabetes_probability_percentage = float(diabetes_probability) * 100
    heartdisease_probability_percentage=float(heartdisease_probability) * 100
    
    height_cm=user_data['height']
    predictbloodpresure=predict_bloodpresure(user_data)
    ideal_weight_lower, ideal_weight_upper = suggest_ideal_weight(height_cm)
    bmr_prediction=predict_bmr(user_data, bmiValue)
    suggested_weight_range = f"The ideal weight for you is between {ideal_weight_lower} kg and {ideal_weight_upper} kg."
    output= f"Health Status {health_status}",
     
    response = {
        "Predicted Blood presure":f"{predictbloodpresure}",
        "Suggested weight range ": f"{suggested_weight_range}",
        "dishes": recommended_dishes,
        "BMR Value":f"{bmr_prediction:.2f}",
        "Diabetic probability": f"{diabetes_probability_percentage:.2f}%",
        # "Body fat probability " : f"{bodyfact_probability_percentage:.2f}%",
        "Heart Disease probability Percentage":f"{heartdisease_probability_percentage:.2f}%"
    }
    return output,response




