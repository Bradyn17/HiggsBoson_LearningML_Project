import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Step 1: Load the data and separate features (X) and target (y)
data = pd.read_csv("training.csv")
data['approx_velocity'] = data['DER_pt_tot'] / data['DER_mass_MMC'] # creates ratio of particle momentum to energy

# Drop BOTH the 'Label' and the 'Weight' columns
X = data.drop(['Label', 'Weight'], axis=1)
y = data['Label'].apply(lambda x: 1 if x == 's' else 0)

# Step 2: Split the data BEFORE handling missing values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Step 3: Handle missing values for TRAINING data ONLY
X_train.replace(-999, np.nan, inplace=True)
X_train.fillna(X_train.mean(), inplace=True)

# Step 4: Handle missing values for TESTING data using the MEAN of the training data
X_test.replace(-999, np.nan, inplace=True)
X_test.fillna(X_train.mean(), inplace=True)

# Step 5: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=17)
model.fit(X_train, y_train)
final_model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=17)
final_model.fit(X_train, y_train)
# Step 6: Make predictions and evaluate
y_pred = model.predict(X_test)
y_pred = final_model.predict(X_test) # pred final model
accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 5, 10]
# }
# # Create a Random Forest model
# rf = RandomForestClassifier(random_state=17)

# # Create the GridSearchCV object
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# # Fit the grid search to your training data
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters found
# print("Best Hyperparameters:", grid_search.best_params_)

print(f"Final Optimized Model Accuracy: {accuracy:.2f}")
