import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
import numpy as np

# Dataset: Age of the car (in years) vs Resale value (in lakhs)
dataset = {
    "age": [1, 3, 5, 7, 10],
    "resale_value": [8.5, 6.0, 4.2, 3.1, 1.5]
}

# Prepare input (X) and output (y)
x = [[age] for age in dataset["age"]]
y = dataset["resale_value"]

# Create and train the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(x, y)

# Predict resale value for a 1-year-old car
predicted_value = model.predict([[1]])
print(f"Predicted resale value for a 1-year-old car: {predicted_value[0]:.2f} lakhs")

# Plotting the Decision Tree
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=["Car Age"], filled=True, rounded=True)
plt.title("Decision Tree Regressor")
plt.show()

# Create high-resolution inputs for smooth prediction line
x_test = np.arange(0, 11, 0.1).reshape(-1, 1)
y_pred = model.predict(x_test)

# Plot the original data and model prediction
plt.figure(figsize=(8, 5))
plt.scatter(dataset["age"], y, color='red', label="Actual Data")
plt.plot(x_test, y_pred, color='blue', label="Model Prediction")
plt.xlabel("Car Age (years)")
plt.ylabel("Resale Value (lakhs)")
plt.title("Car Age vs Resale Value - Decision Tree Regression")
plt.legend()
plt.grid(True)
plt.show()


