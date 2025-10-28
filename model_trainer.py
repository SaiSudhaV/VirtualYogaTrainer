import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import time

class YogaModelTrainer:
    def __init__(self, pose_names):
        self.pose_names = pose_names
        self.models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        self.param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5]
            },
            'SVM': {
                'C': [1, 10, 100],
                'kernel': ['rbf', 'linear']
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [0.001, 0.01]
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        }
        
        self.best_model = None
        self.best_score = 0
        self.best_model_name = ""
    
    def train_and_tune_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Train and tune multiple models"""
        print("Training and tuning models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                model, self.param_grids[name], 
                cv=3, scoring='accuracy', n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate on validation and test sets
            val_acc = accuracy_score(y_val, grid_search.predict(X_val))
            test_acc = accuracy_score(y_test, grid_search.predict(X_test))
            
            # Cross-validation score
            cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
            
            results[name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time
            }
            
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Validation accuracy: {val_acc:.3f}")
            print(f"  Test accuracy: {test_acc:.3f}")
            print(f"  CV score: {cv_scores.mean():.3f} (±{cv_scores.std()*2:.3f})")
            print(f"  Training time: {training_time:.2f}s")
            
            # Update best model based on validation accuracy
            if val_acc > self.best_score:
                self.best_score = val_acc
                self.best_model = grid_search.best_estimator_
                self.best_model_name = name
        
        return results
    
    def evaluate_best_model(self, X_test, y_test):
        """Evaluate the best model"""
        if self.best_model is None:
            print("No model trained yet!")
            return
        
        y_pred = self.best_model.predict(X_test)
        
        print(f"\n=== Best Model: {self.best_model_name} ===")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.pose_names, digits=3))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def save_model_info(self, filename='best_model_info.json'):
        """Save model information as JSON"""
        if self.best_model is None:
            print("No model to save!")
            return
        
        import json
        model_info = {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'accuracy': float(self.best_score),
            'parameters': self.best_model.get_params() if hasattr(self.best_model, 'get_params') else {},
            'pose_names': self.pose_names
        }
        
        with open(filename, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        print(f"Best model info ({self.best_model_name}) saved to {filename}")
        print(f"Model accuracy: {self.best_score:.3f}")