from flask import Flask, jsonify
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('D:/WebiSoftTech/ELASTIC NET REGRESSION/boston_houses.csv')

# Preprocess the data
X = data.drop('MEDV', axis=1)  
y = data['MEDV'] 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implement Regression
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)  
elastic_net.fit(X_train_scaled, y_train)

# Get feature 
importance = elastic_net.coef_
feature_importance = pd.Series(importance, index=X.columns).sort_values(ascending=False)

# Select the feature
most_important_feature = feature_importance.idxmax()
most_important_value = feature_importance.max()

@app.route('/feature-selection', methods=['GET'])
def feature_selection():
    return jsonify({
        'most_important_feature': most_important_feature,
        'importance_value': most_important_value
    })

if __name__ == '__main__':
    app.run(debug=True)