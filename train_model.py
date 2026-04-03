import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the dataset
df = sns.load_dataset('titanic')

# 2. Basic Preprocessing
# Select a few simple features to keep the model basic
features = ['pclass', 'sex', 'age', 'fare']
X = df[features].copy()
y = df['survived']

# Handle missing values (fill age with median)
X['age'] = X['age'].fillna(X['age'].median())
X['fare'] = X['fare'].fillna(X['fare'].median())

# Convert categorical 'sex' into numerical (0 for male, 1 for female)
X['sex'] = X['sex'].map({'male': 0, 'female': 1})

# 3. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save the Model
joblib.dump(model, 'titanic_model.pkl')
print("Model trained and saved as titanic_model.pkl")