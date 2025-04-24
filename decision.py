from sklearn.tree import DecisionTreeRegressor

# New dataset: Age of the car (years) vs Resale value (in lakhs)
dataset = {
    "age": [1, 3, 5, 7, 10],
    "resale_value": [8.5, 6.0, 4.2, 3.1, 1.5]
}

# Prepare input (X) and target (y)
x = [[value] for value in dataset["age"]]
y = dataset["resale_value"]

# Create and train the Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(x, y)

# Predict resale value for a car that's 4 years old
predicted_value = model.predict([[1]])
print(predicted_value)
