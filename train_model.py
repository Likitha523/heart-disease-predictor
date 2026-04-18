import os
import pandas as pd
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

def download_data():
    file_path = "cardio_train.csv"
    if not os.path.exists(file_path):
        print("Downloading dataset...")
        url = "https://raw.githubusercontent.com/SaneSky109/DATA606/main/Data_Project/Data/cardio_train.csv"
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    return file_path

def load_and_preprocess(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path, sep=';')
    
    print("Initial shape:", df.shape)
    
    # Drop id
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)
        
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Handle missing values
    df.dropna(inplace=True)
    
    # Handle outliers (ap_hi, ap_lo, height, weight)
    df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 50) & (df['ap_lo'] <= 150)]
    df = df[(df['ap_hi'] > df['ap_lo'])] 
    df = df[(df['height'] >= 130) & (df['height'] <= 220)]
    df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]
    
    print("Shape after outlier removal:", df.shape)
    
    # Feature Engineering
    # 1. Age in years
    df['age_years'] = (df['age'] / 365.25)
    
    # 2. BMI
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # 3. BP Risk Category
    def categorize_bp(ap_hi, ap_lo):
        if ap_hi < 120 and ap_lo < 80:
            return 0 # Normal
        elif 120 <= ap_hi <= 129 and ap_lo < 80:
            return 1 # Elevated
        elif 130 <= ap_hi <= 139 or 80 <= ap_lo <= 89:
            return 2 # Stage 1 HBP
        elif ap_hi >= 140 or ap_lo >= 90:
            return 3 # Stage 2 HBP
        else:
            return 0
            
    df['bp_risk'] = df.apply(lambda x: categorize_bp(x['ap_hi'], x['ap_lo']), axis=1)
    
    # Categorise Age Group
    def categorize_age(age):
        if age < 40: return 0
        elif 40 <= age < 50: return 1
        elif 50 <= age < 60: return 2
        else: return 3
    df['age_group'] = df['age_years'].apply(categorize_age)
    
    # Drop raw age since we have age_years and age_group
    df.drop('age', axis=1, inplace=True)
    
    return df

def main():
    file_path = download_data()
    df = load_and_preprocess(file_path)
    
    X = df.drop('cardio', axis=1)
    y = df['cardio']
    
    feature_cols = X.columns.tolist()
    
    # Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Class Imbalance Handling
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sm)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X) # For feature importance context
    
    # Save the scaler and feature names
    joblib.dump({"scaler": scaler, "features": feature_cols}, 'preprocessor.pkl')
    
    print("Training models...")
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    }
    
    # Hyperparameter tuning for XGBoost
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    print("Tuning XGBoost with RandomizedSearchCV...")
    xgb_search = RandomizedSearchCV(XGBClassifier(random_state=42, eval_metric='logloss'), 
                                    xgb_param_grid, n_iter=4, cv=5, scoring='roc_auc', random_state=42, n_jobs=-1)
    xgb_search.fit(X_train_scaled, y_train_sm)
    
    models['XGBoost'] = xgb_search.best_estimator_
    
    # We omit SVM because it is too slow with n_samples > 60k
    
    best_model = None
    best_auc = 0
    best_name = ""
    
    print()
    print("--- Model Evaluation ---")
    for name, model in models.items():
        if name != 'XGBoost':
            model.fit(X_train_scaled, y_train_sm)
            
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"[{name}]")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}\n")
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name
            
    print(f"Selected Best Model: {best_name}\n")
    
    # Detailed Eval for Best Model
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualizations
    importances = None
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
        
    if importances is not None:
        features_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
        features_df.sort_values(by='Importance', ascending=False, inplace=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=features_df, palette='viridis')
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Saved feature_importance.png")
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix: {best_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {best_auc:.4f}', color='darkorange', lw=2)
    plt.plot([0,1], [0,1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    print("Saved roc_curve.png")
    
    # Export Model
    joblib.dump(best_model, 'heart_model.pkl')
    print("Model saved as heart_model.pkl")

if __name__ == '__main__':
    main()
