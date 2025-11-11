# Experiment 7: Mini Project — Titanic Survival Prediction
# Aim: Build a ML model to predict who survived the Titanic shipwreck.

# Step 1: Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Step 2: Reading Datasets
train = pd.read_csv(r'D:\Ml_Code\ML\ML\mini_project\train.csv')
test = pd.read_csv(r'D:\Ml_Code\ML\ML\mini_project\test.csv')
gender = pd.read_csv(r'D:\Ml_Code\ML\ML\mini_project\gender_submission.csv')

print("✅ Datasets Loaded Successfully!")

# Step 3: Cleaning the Data
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing values
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].dropna().mode()[0])
test['Fare'] = test['Fare'].fillna(test['Fare'].dropna().mean())

# Guess missing 'Age' values based on Pclass and Sex
guess_ages = np.zeros((2, 3))
combine = [train, test]

# Convert 'Sex' to numeric (female=1, male=0)
for ds in combine:
    ds['Sex'] = ds['Sex'].map({'female': 1, 'male': 0}).astype(int)

for ds in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = ds[(ds['Sex'] == i) & (ds['Pclass'] == j + 1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            ds.loc[(ds.Age.isnull()) & (ds.Sex == i) & (ds.Pclass == j + 1), 'Age'] = guess_ages[i, j]

    ds['Age'] = ds['Age'].astype(int)

# Step 4: Preparing Data for Model
X_train = pd.get_dummies(train.drop(['Survived', 'PassengerId'], axis=1))
y_train = train['Survived']
X_test = pd.get_dummies(test.drop(['PassengerId'], axis=1))

# Step 5: Training the Model
model = RandomForestClassifier(
    n_estimators=80,
    max_depth=5,
    max_features=8,
    min_samples_split=3,
    random_state=7
)
model.fit(X_train, y_train)

# Step 6: Predicting Survival
predictions = model.predict(X_test)

# Step 7: Evaluation Function
def print_scores(model, X_train, y_train, predictions, cv_splits=10):
    print("\n📊 Model Evaluation Results:")
    print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_splits)
    print(f"Cross-validation Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} ± {cv_scores.std() * 2:.2f}")

print_scores(model, X_train, y_train, predictions)

# Step 8: Combine Predictions with Passenger Info (for expected output)
output = test.copy()
output['Survived'] = predictions

# Select columns to match expected display
expected_output = output[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

print("\nOutput:")
print(expected_output.head())

# Step 9: Save Results (optional)
submission = gender.copy()
submission['Survived'] = predictions
submission.to_csv('titanic_submission.csv', index=False)
print("\n✅ Predictions saved to 'titanic_submission.csv'")
