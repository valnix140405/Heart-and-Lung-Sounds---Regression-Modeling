
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def analyze_dataset(filepath):
    print(f"Loading dataset from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Dataset Shape:", df.shape)
    print("\nDataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Define target and features
    # Potential targets: 'Sleep Duration', 'Quality of Sleep'
    # Let's try 'Sleep Duration' first as a continuous variable suitable for regression
    target_col = 'Sleep Duration' 
    
    if target_col not in df.columns:
        print(f"\nTarget column '{target_col}' not found. Trying 'Quality of Sleep'...")
        target_col = 'Quality of Sleep'
        if target_col not in df.columns:
            print("\nNeither 'Sleep Duration' nor 'Quality of Sleep' found. Please specify target.")
            return

    print(f"\nTarget Variable: {target_col}")

    X = df.drop(columns=[target_col, 'Person ID']) # Drop Person ID as it's an identifier
    y = df[target_col]

    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    print(f"\nNumerical Features: {list(numeric_features)}")
    print(f"Categorical Features: {list(categorical_features)}")

    # Preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(alpha=0.1),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }

    results = {}

    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        
        print(f"\n{name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2: {r2:.4f}")

    return results

if __name__ == "__main__":
    filepath = "Sleep_health_and_lifestyle_dataset.csv"
    analyze_dataset(filepath)
