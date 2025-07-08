import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv("IMDb Movies India.csv", encoding="ISO-8859-1")
df = df.dropna(subset=["Rating"])

# Clean columns
df["Year"] = df["Year"].str.extract(r"(\d{4})").astype(float)
df["Duration"] = df["Duration"].str.extract(r"(\d+)").astype(float)
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# Only keep needed columns
features = ["Year", "Duration", "Genre", "Votes"]
X = df[features]
y = df["Rating"]

# Preprocessing
num_features = ["Year", "Duration", "Votes"]
cat_features = ["Genre"]

num_transformer = SimpleImputer(strategy="mean")
cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

# Fast model pipeline
model = Pipeline([
    ("pre", preprocessor),
    ("lr", LinearRegression())
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# ==== 🔽 USER INPUT ====
print("Enter movie details to predict IMDB rating:\n")
year = float(input("Enter release year (e.g. 2023): "))
duration = float(input("Enter duration in minutes (e.g. 120): "))
genre = input("Enter genre (e.g. Drama): ")
votes = float(input("Enter number of votes (e.g. 1500): "))

# Create input DataFrame
input_data = {
    "Year": year,
    "Duration": duration,
    "Genre": genre,
    "Votes": votes
}
input_df = pd.DataFrame([input_data])

# Predict
predicted_rating = model.predict(input_df)[0]
print(f"\n⭐ Predicted IMDB Rating: {predicted_rating:.2f}")

# ==== 📊 GRAPH ====
average_rating = y.mean()
plt.figure(figsize=(6, 4))
bars = plt.bar(["Predicted", "Average IMDB"], [predicted_rating, average_rating], color=["blue", "gray"])
plt.title("Movie Rating Prediction")
plt.ylabel("IMDB Rating")
plt.ylim(0, 10)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{yval:.2f}", ha='center')

plt.tight_layout()
plt.show()