import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Add the parent directory to path to import utils when run as script
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils import get_base_path

def train_basic_model(year=2022, test_size=0.3, random_state=42):
    """Train a basic model to predict race winners"""
    base_path = get_base_path()
    data_path = os.path.join(base_path, "data", "raw", f'f1_data_{year}.csv')
    
    # Check if file exists, if not collect data
    if not os.path.exists(data_path):
        from src.data.collect import collect_and_save_data
        f1_data = collect_and_save_data(year)
    else:
        f1_data = pd.read_csv(data_path)
    
    # Create a target variable - is_winner (1 if position is 1, 0 otherwise)
    f1_data['is_winner'] = (f1_data['position'] == 1).astype(int)

    # Create basic features
    features = ['qualifying_position']

    # Split the data
    X = f1_data[features]
    y = f1_data['is_winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"Training accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Testing accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    # Check how often each qualifying position leads to a win
    win_rates = f1_data.groupby('qualifying_position')['is_winner'].mean()
    print("\nWin rate by qualifying position:")
    for pos, rate in win_rates.items():
        if rate > 0:
            print(f"Position {pos}: {rate:.1%}")
    
    # Save results directory if needed
    results_path = os.path.join(base_path, "results")
    os.makedirs(results_path, exist_ok=True)
    
    # Save model if needed
    models_path = os.path.join(base_path, "models")
    os.makedirs(models_path, exist_ok=True)
    
    # Return model and results for further use
    return {
        'model': model,
        'accuracy': accuracy_score(y_test, test_pred),
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

if __name__ == "__main__":
    train_basic_model()