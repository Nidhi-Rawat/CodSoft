import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 📂 Step 1: Load Dataset
df = pd.read_csv("advertising.csv")  # Your uploaded file
print("\n✅ Dataset Loaded Successfully!\n")
print(df.head())

# 📊 Step 2: Visualize Data
sns.pairplot(df)
plt.suptitle("📊 Relationships Between Advertising Budget and Sales", y=1.02)
plt.show()

# 🎯 Step 3: Define Features and Target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 🔀 Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Step 5: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📈 Step 6: Evaluate Model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Model Evaluation")
print(f"R² Score: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

# 💾 Step 7: Save the model
joblib.dump(model, "sales_model.pkl")
print("\n💾 Model saved as sales_model.pkl")

# 🧠 Step 8: Take User Input for Prediction
print("\n🔮 Let's Predict Sales from Your Advertising Budget")
try:
    tv = float(input("Enter TV budget (in thousands): "))
    radio = float(input("Enter Radio budget (in thousands): "))
    newspaper = float(input("Enter Newspaper budget (in thousands): "))

    input_data = pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "Radio", "Newspaper"])
    predicted_sales = model.predict(input_data)

    print(f"\n📈 Predicted Sales: {predicted_sales[0]:.2f} units")
except ValueError:
    print("❌ Invalid input! Please enter numerical values only.")
