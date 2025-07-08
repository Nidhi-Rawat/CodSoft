import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Clean data
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Show both graphs using subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Survival Count
sns.countplot(ax=axes[0], x='Survived', data=df)
axes[0].set_title("Survival Count (0 = No, 1 = Yes)")

# Subplot 2: Survival by Gender
sns.countplot(ax=axes[1], x='Sex_male', hue='Survived', data=df)
axes[1].set_title("Survival by Gender (0 = Female, 1 = Male)")
axes[1].set_xticklabels(['Female', 'Male'])

plt.tight_layout()
plt.show()

# Prepare features & target
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save model
joblib.dump(model, "titanic_model.pkl")

# Get input from user
print("\n🔷 Enter passenger details to predict survival:")

pclass = int(input("Passenger Class (1, 2, or 3): "))
age = float(input("Age: "))
sibsp = int(input("No. of siblings/spouses aboard: "))
parch = int(input("No. of parents/children aboard: "))
fare = float(input("Fare paid: "))
sex = input("Sex (male/female): ").strip().lower()
embarked = input("Embarked (Q/S/C): ").strip().upper()

# Format inputs
sex_male = 1 if sex == "male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

sample_input = pd.DataFrame([{
    'PassengerId': 999,
    'Pclass': pclass,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Sex_male': sex_male,
    'Embarked_Q': embarked_Q,
    'Embarked_S': embarked_S
}])

# Predict
prediction = model.predict(sample_input)
print("\n🔍 Prediction Result:")
print("✅ Survived" if prediction[0] == 1 else "❌ Did not survive")
