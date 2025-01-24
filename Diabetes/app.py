from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('D:/WebiSoftTech/ELASTIC NET REGRESSION/Diabetes/diabetes.csv')

# Define features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create and fit the ElasticNet model
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
elastic_net.fit(X_train_scaled, y_train)

# Get the coefficients
coefficients = elastic_net.coef_

# Create a DataFrame 
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
feature_importance['Absolute Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance = feature_importance.sort_values(by='Absolute Coefficient', ascending=False)

# Identify the most determining factor
most_determining_factor = feature_importance.iloc[0]

@app.route('/')
def index():
    return render_template('D:/WebiSoftTech/ELASTIC NET REGRESSION/Diabetes/template/index.html', 
                           most_determining_factor=most_determining_factor['Feature'], 
                           coefficient=most_determining_factor['Coefficient'])

if __name__ == '__main__':
    app.run(debug=True)