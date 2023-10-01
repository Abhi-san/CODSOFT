import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('churn.csv')

X = data.drop(columns=['Exited', 'RowNumber', 'CustomerId', 'Surname', 'Geography', 'Gender'])
y = data['Exited']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression() #Logistic Regression Model
model.fit(X_scaled, y)

new_data = {}
new_data['CreditScore'] = int(input("Enter CreditScore: "))
new_data['Gender1'] = int(input("Enter Gender (1 for Female, 0 for Male): "))
new_data['Age'] = int(input("Enter Age: "))
new_data['Tenure'] = int(input("Enter Tenure: "))
new_data['Balance'] = float(input("Enter Balance: "))
new_data['NumOfProducts'] = int(input("Enter NumOfProducts: "))
new_data['HasCrCard'] = int(input("Enter HasCrCard (1 for Yes, 0 for No): "))
new_data['IsActiveMember'] = int(input("Enter IsActiveMember (1 for Yes, 0 for No): "))
new_data['EstimatedSalary'] = float(input("Enter EstimatedSalary: "))

new_data_df = pd.DataFrame(new_data, index=[0])

new_data_scaled = scaler.transform(new_data_df)

prediction = model.predict(new_data_scaled)

if prediction[0] == 0:
    print("Prediction: The customer is likely to churn -_-!")
else:
    print("Prediction: The customer is not likely to churn!!!")
