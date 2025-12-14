import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ForestCoverClassifier:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.feature_importances_ = None
        
    def train(self, X_train, y_train):
        """Train the model"""
        print("Training the model...")
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        print("Training completed!")
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model's performance"""
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'report': report
        }
    
    def plot_feature_importances(self, feature_names, top_n=20):
        """Plot feature importances"""
        if self.feature_importances_ is None:
            raise ValueError("Model has not been trained yet.")
            
        # Create a DataFrame for visualization
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importances)
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importances.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize='true'):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=range(1, 8), 
                   yticklabels=range(1, 8))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Create directory if it doesn't exist
        import os
        os.makedirs('reports/figures', exist_ok=True)
        
        plt.savefig('reports/figures/confusion_matrix.png')
        plt.close()
    
    def save_model(self, filepath='models/forest_cover_model.pkl'):
        """Save the trained model to disk"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath='models/forest_cover_model.pkl'):
        """Load a trained model from disk"""
        model = cls()
        model.model = joblib.load(filepath)
        return model
