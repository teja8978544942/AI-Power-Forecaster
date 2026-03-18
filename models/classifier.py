import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def train_classifier(X_train, y_train, X_test, y_test):
    os.makedirs('models', exist_ok=True)
    
    print("Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    print("Evaluating XGBoost...")
    xgb_preds = xgb_model.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
    
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
        
    return xgb_model

def train_svm_classifier(X_train, y_train, X_test, y_test):
    print("Training SVM Classifier...")
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X_train, y_train)
    
    print("Evaluating SVM...")
    svm_preds = svm_model.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, svm_preds):.4f}")
    
    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
        
    return svm_model

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pipeline.etl import ETLPipeline
    
    pipeline = ETLPipeline('../iiot_smart_grid_dataset.csv')
    df = pipeline.load_data()
    df = pipeline.feature_engineering(df)
    
    # We don't want sequences for static classification, we just want standard rows
    # The target is 'Peak_Load_Hour'
    features = [c for c in df.columns if c not in ['Peak_Load_Hour', 'Demand_Response_Event']]
    
    X = df[features].values
    y = df['Peak_Load_Hour'].values
    
    # Simple split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    train_classifier(X_train, y_train, X_test, y_test)
    # SVM can take a while to train on thousands of rows, leaving code here if needed
    # train_svm_classifier(X_train, y_train, X_test, y_test)
