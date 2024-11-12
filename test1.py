from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import pandas as pd

# Load data
data = pd.read_csv(r"C:\Users\d1411\Документы\Python Projects\Final Project\data\csv\df_processed\df_combined.csv")

# Set target and feature columns
target = 'Winner'
features = [col for col in data.columns if col != target]

# Split data into training and test sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Initialize AutoGluon Predictor
predictor = TabularPredictor(
    label=target, 
    eval_metric="f1"  # Set F1 as the evaluation metric
).fit(
    train_data=train_data,
    time_limit=15*60,   # Set a time limit in seconds
    presets='best_quality'
)

# Display the best model and its parameters
best_model = predictor.get_model_best()
print(f"Best model: {best_model}")

best_model_params = predictor.get_model_attribute(best_model, 'params')
print("Best Model Parameters:")
for param_name, param_value in best_model_params.items():
    print(f"{param_name}: {param_value}")

# Evaluate on the test set
y_test = test_data[target]
y_pred = predictor.predict(test_data)
y_pred_proba = predictor.predict_proba(test_data)[1]  # Probability for positive class

# Calculate performance metrics
auc_score = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nBest Model Performance on Test Data:")
print(f"AUC: {auc_score}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)

# Display leaderboard
leaderboard = predictor.leaderboard(test_data)
print("\nLeaderboard:")
print(leaderboard)
