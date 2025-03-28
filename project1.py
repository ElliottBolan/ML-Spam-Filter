import os
import sys
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Download the stopwords and punctuation
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Make paths to the datasets in enron1, enron2, and enron4, which are in the same folder as this file
script_dir = os.path.dirname(__file__)
datasets = {
    "enron1": {
        "train": {
            "spam": f"{script_dir}/Datasets/enron1/train/spam",
            "ham": f"{script_dir}/Datasets/enron1/train/ham"
        },
        "test": {
            "spam": f"{script_dir}/Datasets/enron1/test/spam",
            "ham": f"{script_dir}/Datasets/enron1/test/ham"
        }
    },
    "enron2": {
        "train": {
            "spam": f"{script_dir}/Datasets/enron2/train/spam",
            "ham": f"{script_dir}/Datasets/enron2/train/ham"
        },
        "test": {
            "spam": f"{script_dir}/Datasets/enron2/test/spam",
            "ham": f"{script_dir}/Datasets/enron2/test/ham"
        }
    },
    "enron4": {
        "train": {
            "spam": f"{script_dir}/Datasets/enron4/train/spam",
            "ham": f"{script_dir}/Datasets/enron4/train/ham"
        },
        "test": {
            "spam": f"{script_dir}/Datasets/enron4/test/spam",
            "ham": f"{script_dir}/Datasets/enron4/test/ham"
        }
    }
}

stop_words = set(stopwords.words('english'))

def preprocess(text):
    """Preprocess text by lowercasing, removing punctuation and stopwords."""
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum() and word not in stop_words]
    return words

def load_all_emails(folder_paths):
    """Load all emails from the given folder paths and return their content and labels."""
    emails, labels = [], []
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found -> {folder_path}")
            continue

        folder_name = os.path.basename(folder_path).lower()
        is_spam = folder_name == "spam"

        for root, _, files in os.walk(folder_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                        if content:
                            emails.append(preprocess(content))
                            labels.append(1 if is_spam else 0)
                        else:
                            print(f"Skipping empty file: {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return emails, labels

def create_datasets():
    """Create and save the BoW and Bernoulli datasets for all three Enron datasets."""
    # First, build the vocabulary from all training datasets
    print("Building vocabulary from training datasets...")
    vocabularies = {}
    
    for dataset in datasets:
        vocabulary = set()
        folder_paths = [datasets[dataset]["train"]["spam"], datasets[dataset]["train"]["ham"]]
        emails, _ = load_all_emails(folder_paths)
        for email in emails:
            vocabulary.update(email)
        vocabularies[dataset] = sorted(vocabulary)
        print(f"Vocabulary size for {dataset}: {len(vocabulary)} words")
    
    # Generate CSV files
    print("Generating CSV files...")
    for dataset in datasets:
        vocabulary = vocabularies[dataset]
        
        for split in ["train", "test"]:
            folder_paths = [datasets[dataset][split]["spam"], datasets[dataset][split]["ham"]]
            emails, labels = load_all_emails(folder_paths)
            
            data_bow = []
            data_bernoulli = []
            
            for email, label in zip(emails, labels):
                word_counts = Counter(email)
                
                # Bag of Words representation
                row_bow = [word_counts.get(word, 0) for word in vocabulary]
                row_bow.append(label)
                data_bow.append(row_bow)
                
                # Bernoulli representation
                row_bernoulli = [1 if word in email else 0 for word in vocabulary]
                row_bernoulli.append(label)
                data_bernoulli.append(row_bernoulli)
            
            # Create and save DataFrames
            df_bow = pd.DataFrame(data_bow, columns=vocabulary + ["label"])
            df_bernoulli = pd.DataFrame(data_bernoulli, columns=vocabulary + ["label"])
            
            bow_filename = f"{dataset}_bow_{split}.csv"
            bernoulli_filename = f"{dataset}_bernoulli_{split}.csv"
            
            df_bow.to_csv(bow_filename, index=False)
            df_bernoulli.to_csv(bernoulli_filename, index=False)
            
            print(f"Generated {bow_filename} with shape {df_bow.shape}")
            print(f"Generated {bernoulli_filename} with shape {df_bernoulli.shape}")

def load_dataset(file_path):
    """Load a dataset from a CSV file."""
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

class MultinomialNaiveBayes:
    """Multinomial Naive Bayes implementation for text classification."""
    
    def __init__(self):
        self.class_log_priors = None
        self.feature_log_probs = None
        
    def fit(self, X, y):
        """
        Train the Multinomial Naive Bayes model based on training data.
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
           Training vectors representing document word counts
        y: array-like of shape (n_samples,)
           Target values (class labels)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Calculate class prior probabilities
        class_counts = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            class_counts[i] = np.sum(y == c)
        
        self.class_log_priors = np.log(class_counts / n_samples)
        
        # Calculate conditional probabilities for features
        feature_counts = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            feature_counts[i] = np.sum(X[y == c], axis=0)
        
        # Calculate total counts per class for normalization
        total_counts = np.sum(feature_counts, axis=1).reshape(-1, 1)
        
        # Apply Laplace smoothing and convert to log probabilities
        self.feature_log_probs = np.log((feature_counts + 1) / (total_counts + n_features))
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
           Test vectors representing document word counts
           
        Returns:
        y_pred: array-like of shape (n_samples,)
                Predicted class labels
        """
        return self.classes[np.argmax(self.predict_log_proba(X), axis=1)]
    
    def predict_log_proba(self, X):
        """
        Return log-probability estimates for samples in X.
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
           Test vectors representing document word counts
           
        Returns:
        log_proba: array-like of shape (n_samples, n_classes)
                   Log-probability estimates for each class
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        log_proba = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes):
            log_proba[:, i] = self.class_log_priors[i]
            # Multiply by counts (equivalent to summing log probabilities)
            log_proba[:, i] += np.sum(X * self.feature_log_probs[i], axis=1)
            
        return log_proba

class BernoulliNaiveBayes:
    """Bernoulli Naive Bayes implementation for binary text classification."""
    
    def __init__(self):
        self.class_log_priors = None
        self.feature_log_probs = None
        
    def fit(self, X, y):
        """
        Train the Bernoulli Naive Bayes model based on training data.
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
           Training vectors representing binary word presence
        y: array-like of shape (n_samples,)
           Target values (class labels)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Calculate class prior probabilities
        class_counts = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            class_counts[i] = np.sum(y == c)
        
        self.class_log_priors = np.log(class_counts / n_samples)
        
        # Calculate conditional probabilities for features
        # Count occurrences of each word in each class
        feature_counts = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            feature_counts[i] = np.sum(X_c, axis=0)
            
        # Calculate document counts per class for normalization
        doc_counts = class_counts.reshape(-1, 1)
        
        # Apply Laplace smoothing and convert to log probabilities
        # P(x_i | y) = (count(x_i, y) + 1) / (count(y) + 2)
        self.feature_log_probs = np.log((feature_counts + 1) / (doc_counts + 2))
        self.feature_log_neg_probs = np.log(1 - np.exp(self.feature_log_probs))
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
           Test vectors representing binary word presence
           
        Returns:
        y_pred: array-like of shape (n_samples,)
                Predicted class labels
        """
        return self.classes[np.argmax(self.predict_log_proba(X), axis=1)]
    
    def predict_log_proba(self, X):
        """
        Return log-probability estimates for samples in X.
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
           Test vectors representing binary word presence
           
        Returns:
        log_proba: array-like of shape (n_samples, n_classes)
                   Log-probability estimates for each class
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        log_proba = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes):
            # Start with class prior
            log_proba[:, i] = self.class_log_priors[i]
            
            # Add contributions from present features (X_j = 1)
            present = X.astype(bool)
            log_proba[:, i] += np.sum(present * self.feature_log_probs[i], axis=1)
            
            # Add contributions from absent features (X_j = 0)
            absent = ~present
            log_proba[:, i] += np.sum(absent * self.feature_log_neg_probs[i], axis=1)
            
        return log_proba

class LogisticRegressionL2:
    """Logistic Regression with L2 regularization implemented using gradient ascent."""
    
    def __init__(self, learning_rate=0.01, max_iter=100, tol=1e-3):
        """
        Initialize the logistic regression model.
        
        Parameters:
        learning_rate: float, default=0.01
                      The learning rate for gradient ascent
        max_iter: int, default=1000
                 Maximum number of iterations for convergence
        tol: float, default=1e-4
            Tolerance for stopping criteria
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        """Apply the sigmoid function."""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, lambda_):
        """
        Fit the model according to the given training data using gradient ascent.
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
           Training vector
        y: array-like of shape (n_samples,)
           Target values
        lambda_: float
                L2 regularization parameter
        
        Returns:
        self: object
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.max_iter):
            # Calculate current predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (lambda_ / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters with gradient ascent (note the negative sign for maximization)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if np.linalg.norm(dw) < self.tol and np.abs(db) < self.tol:
                break
                
        return self
    
    def predict_proba(self, X):
        """
        Probability estimates for samples in X.
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
           Test vectors
           
        Returns:
        proba: array-like of shape (n_samples,)
              Probability of the sample for the positive class
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
           Test vectors
           
        Returns:
        y_pred: array-like of shape (n_samples,)
                Predicted class labels
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance using accuracy, precision, recall, and F1-score.
    
    Parameters:
    y_true: array-like of shape (n_samples,)
           True class labels
    y_pred: array-like of shape (n_samples,)
           Predicted class labels
           
    Returns:
    metrics: dict
            Dictionary containing all metrics
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    }

def run_multinomial_nb():
    """Run MultinomialNaiveBayes on all datasets with BoW representation."""
    print("\n===== Multinomial Naive Bayes (Bag of Words) =====")
    results = {}
    
    for dataset in datasets:
        train_file = f"{dataset}_bow_train.csv"
        test_file = f"{dataset}_bow_test.csv"
        
        X_train, y_train = load_dataset(train_file)
        X_test, y_test = load_dataset(test_file)
        
        print(f"\nDataset: {dataset}")
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
        
        model = MultinomialNaiveBayes()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_model(y_test, y_pred)
        results[dataset] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return results

def run_bernoulli_nb():
    """Run BernoulliNaiveBayes on all datasets with Bernoulli representation."""
    print("\n===== Discrete Naive Bayes (Bernoulli) =====")
    results = {}
    
    for dataset in datasets:
        train_file = f"{dataset}_bernoulli_train.csv"
        test_file = f"{dataset}_bernoulli_test.csv"
        
        X_train, y_train = load_dataset(train_file)
        X_test, y_test = load_dataset(test_file)
        
        print(f"\nDataset: {dataset}")
        print(f"Training on {X_train.shape[0]} samples, testing on {X_test.shape[0]} samples")
        
        model = BernoulliNaiveBayes()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_model(y_test, y_pred)
        results[dataset] = metrics
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return results

def run_logistic_regression():
    """Run LogisticRegression on all datasets with both BoW and Bernoulli representations."""
    print("\n===== Logistic Regression with L2 Regularization =====")
    results = {}
    
    for dataset in datasets:
        for feature_type in ["bow", "bernoulli"]:
            train_file = f"{dataset}_{feature_type}_train.csv"
            test_file = f"{dataset}_{feature_type}_test.csv"
            
            X_train_full, y_train_full = load_dataset(train_file)
            X_test, y_test = load_dataset(test_file)
            
            # Split training set for hyperparameter tuning
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.3, random_state=42)
            
            print(f"\nDataset: {dataset}, Feature Type: {feature_type}")
            print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples")
            
            # Hyperparameter tuning
            lambdas = [0.1, 1.0, 10.0]
            best_lambda = None
            best_f1 = -np.inf
            
            for lambda_ in lambdas:
                model = LogisticRegressionL2(learning_rate=0.1, max_iter=100)
                model.fit(X_train, y_train, lambda_)
                y_val_pred = model.predict(X_val)
                val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_lambda = lambda_
            
            print(f"Best lambda: {best_lambda}")
            
            # Train on full training set with best lambda
            final_model = LogisticRegressionL2(learning_rate=0.1, max_iter=100)
            final_model.fit(X_train_full, y_train_full, best_lambda)
            y_pred = final_model.predict(X_test)
            
            metrics = evaluate_model(y_test, y_pred)
            results[f"{dataset}_{feature_type}"] = metrics
            
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return results

def main():
    print("Welcome to the Email Spam Classification Project!")
    
    # Check if datasets exist, otherwise create them
    first_dataset = "enron1_bow_train.csv"
    if not os.path.exists(first_dataset):
        print("Datasets not found. Creating datasets...")
        create_datasets()
    else:
        print("Datasets found. Skipping dataset creation.")
    
    # Run experiments
    nb_multinomial_results = run_multinomial_nb()
    nb_bernoulli_results = run_bernoulli_nb()
    lr_results = run_logistic_regression()
    
    print("\n===== Summary =====")
    print("Experiment complete! All results have been printed above.")
    print("Please refer to the CSV files for the generated datasets.")

if __name__ == "__main__":
    main()