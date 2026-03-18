from sklearn.ensemble import IsolationForest
import pickle
import os

def train_anomaly_detector(X_train):
    os.makedirs('models', exist_ok=True)
    
    print("Training Isolation Forest for Anomaly Detection...")
    # We focus on electrical characteristics for pure hardware anomaly detection
    # assuming X_train contains the relevant features (Voltage, Current, Power_Factor)
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_forest.fit(X_train)
    
    with open('models/isolation_forest.pkl', 'wb') as f:
        pickle.dump(iso_forest, f)
        
    return iso_forest

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.etl import ETLPipeline
    
    pipeline = ETLPipeline('../iiot_smart_grid_dataset.csv')
    df = pipeline.load_data()
    
    # Use key electrical parameters that indicate grid stress
    hardware_features = ['Voltage_V', 'Current_A', 'Power_Factor', 'Grid_Frequency_Hz']
    X = df[hardware_features].dropna().values
    
    train_anomaly_detector(X)
