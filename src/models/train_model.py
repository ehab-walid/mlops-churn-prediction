import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Keep your tracking URI the same as before!
mlflow.set_tracking_uri("file:///Users/Admin/Desktop/mlops-churn-prediction/mlruns") 
mlflow.set_experiment("Churn-Prediction-Experiment")

def train():
    print("Loading cleaned data...")
    df = pd.read_csv("data/processed/churn_processed.csv")
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"Yes": 1, "No": 0}) 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Find which columns are text (categorical)
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 2. Create a preprocessor that turns text into numbers
    # handle_unknown='ignore' IS THE MAGIC SAUCE. If the API sees new text, it won't crash!
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Leave the numbers (like MonthlyCharges) alone
    )
    
    # 3. Bundle the preprocessor and the model into a single Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # --- MLFLOW BLOCK ---
    with mlflow.start_run():
        print("Training Pipeline...")
        # Notice we just call .fit() on the pipeline! It does the preprocessing AND training in one go.
        pipeline.fit(X_train, y_train)
        
        print("Evaluating Pipeline...")
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        
        mlflow.log_param("model_type", "RandomForest_Pipeline")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        # 4. We are now saving the ENTIRE pipeline to MLflow
        mlflow.sklearn.log_model(pipeline, "churn_pipeline")
        
        print("Pipeline saved to MLflow!")

if __name__ == "__main__":
    train()