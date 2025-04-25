# 🚗 Decision Tree Regression - Car Age vs Resale Value

This project demonstrates how to use a **Decision Tree Regressor** from `sklearn` to predict the resale value of a car based on its age.

---
## 📷Visualizations
![Screenshot 2025-04-25 093309](https://github.com/user-attachments/assets/3f87b2ed-5348-4f07-ae5f-72db30022b7a)
![Screenshot 2025-04-25 093354](https://github.com/user-attachments/assets/9efc0737-b7df-4211-ba8b-a40d4e243ef8)


## 📊 Dataset

We are using a small dataset with the following format:

| Car Age (Years) | Resale Value (Lakhs) |
|------------------|-----------------------|
| 1                | 8.5                   |
| 3                | 6.0                   |
| 5                | 4.2                   |
| 7                | 3.1                   |
| 10               | 1.5                   |

---

## ⚙️ Libraries Used

- `scikit-learn`
- `matplotlib`
- `numpy`
- `pandas`

---

## 📌 Steps

1. **Import necessary libraries**.
2. **Create the dataset** (car age vs resale value).
3. **Train a Decision Tree Regressor** using `sklearn.tree`.
4. **Visualize the decision tree**.
5. **Plot predictions** vs actual values.
6. **Make predictions** for new car ages (e.g., 4 years old).

---

## 📈 Output

- You will see:
  - A **tree diagram** showing how the model makes decisions.
  - A **plot** showing actual vs predicted resale values.
  - Printed prediction result in the terminal.

---

## ▶️ How to Run

1. Make sure you have Python and pip installed.
2. Install the required libraries:
   ```bash
   pip install matplotlib scikit-learn numpy pandas
