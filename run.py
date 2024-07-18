import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dataset import generate_apple_sales_data_with_promo_adjustment
import os
import time
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.set_tracking_uri("https://tracking-server.a2n.finance")

# Sets the current active experiment to the "Apple_Models" experiment and
# returns the Experiment metadata
apple_experiment = mlflow.set_experiment(experiment_id="433043047588038494")

# Define a run name for this iteration of training.
# If this is not set, a unique name will be auto-generated for your run.
run_name = "apples_rf_test_2"

# Define an artifact path that the model will be saved to.
artifact_path = "rf_apples"

data = generate_apple_sales_data_with_promo_adjustment()

# Split the data into features and target and drop irrelevant date field and target field
X = data.drop(columns=["date", "demand"])
y = data["demand"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "bootstrap": True,
    "oob_score": False,
    "random_state": 888,
}

# Train the RandomForestRegressor
rf = RandomForestRegressor(**params)

print("Start the training process")
# Fit the model on the training data
rf.fit(X_train, y_train)

print("Start the evaluation process")
# Predict on the validation set
y_pred = rf.predict(X_val)

# Calculate error metrics
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

# Assemble the metrics we're going to write into a collection
metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
print("Model metrics:", metrics)

# Initiate the MLflow run context
with mlflow.start_run() as run:
    print("Log params to the tracking server")
    # Log the parameters used for the model fit
    mlflow.log_params(params)

    print("Log metrics to the tracking server")
    # Log the error metrics that were calculated during validation
    mlflow.log_metrics(metrics)

    print("Log model and artifacts to the tracking server")
    # Log an instance of the trained model for later use
    mlflow.sklearn.log_model(
        sk_model=rf, input_example=X_val, artifact_path=artifact_path
    )
    # System metrics are logged at 10-second intervals.
    # This line of code ensure metrics are logged.
    time.sleep(10)