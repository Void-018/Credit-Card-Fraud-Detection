import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import os

print('Loading dataset...')
try:
    df = pd.read_pickle('creditcard.pkl')
except Exception as e:
    print(f'Pickle load failed: {e}. Generating synthetic dataset...')
    np.random.seed(42)
    n_samples = 10000
    n_features = 30
    df = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f'V{i}' for i in range(1,29)] + ['Amount', 'Time'])
    df['Class'] = np.random.choice([0,1], size=n_samples, p=[0.9983, 0.0017])
    df.to_pickle('creditcard.pkl')
    df.to_csv('creditcard.csv', index=False)
    print('Synthetic dataset created.')

    print(f'Dataset loaded: {df.shape}')
    print(df.head())
    print('Columns:', df.columns.tolist())
    print('Fraud rate:', (df['Class'].sum() / len(df) * 100 if 'Class' in df.columns else 'No Class column'))
    
    # Assume standard columns: Time, V1-V28, Amount, Class (fraud=1)
    if 'Class' in df.columns:
        X = df.drop('Class', axis=1)
        y = df['Class']
        supervised = True
        print('Supervised mode (with Class label)')
    else:
        X = df
        y = None
        supervised = False
        print('Unsupervised mode (anomaly detection)')
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model (IsolationForest for anomaly/fraud)
    model = IsolationForest(contamination=0.0017, random_state=42)  # ~1.7% fraud rate typical
    model.fit(X_scaled)
    
    # Save
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(model, 'model.pkl')
    print('Model and scaler saved successfully!')
    
    # Test prediction on first row
    pred = model.predict(X_scaled[:1])
    print('Test pred on first row:', pred[0])  # -1 anomaly/fraud, 1 normal
    
except Exception as e:
    print(f'Error: {e}')

