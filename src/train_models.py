import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def train_position_classifiers(df_model):
    """Encodes targets/features and trains Decision Tree and Random Forest models."""
    feature_cols = ['age', 'height', 'contract_duration_days']
    df_clean = df_model.dropna(subset=feature_cols + ['main_position']).copy()
    
    # Encode categorical target
    le = LabelEncoder()
    df_clean['main_position_encoded'] = le.fit_transform(df_clean['main_position'])
    
    X = df_clean[feature_cols]
    y = df_clean['main_position_encoded']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_model.fit(X_train, y_train)
    dt_preds = dt_model.predict(X_test)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, dt_preds):.4f}")
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    
    return rf_model, dt_model, le, X_test, y_test