import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from collections import Counter
import joblib
import os
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

class PredictionEngine:
    def __init__(self, model_dir: str = "saved_models"):
        # Initialize class attributes
        self.model_dir = model_dir
        self.embedding_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dict = {}
        self.pca_dict = {}
        self.party_distribution = None
        self.datasets = {}
        self.train_embeddings = None
        self.test_embeddings = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Constants
        self.factor_scores = [
            'location_score', 'education_score', 'event_coverage_score',
            'echo_chamber_score', 'news_coverage_score', 'malicious_account_score'
        ]
        self.count_columns = [
            "barely_true_counts", "false_counts", "half_true_counts",
            "mostly_true_counts", "pants_on_fire_counts"
        ]
        
        # Initialize NLP model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize sentence transformer (more efficient than BERT)
        self.initialize_embedding_model()

    def initialize_embedding_model(self):
        """Initialize the sentence transformer model or load from cache"""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_model.to(self.device)

    def save_model(self, model_type: str):
        """Save trained models and their configurations"""
        model_path = os.path.join(self.model_dir, f"{model_type}_model.joblib")
        pca_path = os.path.join(self.model_dir, f"{model_type}_pca.joblib")
        
        joblib.dump(self.models_dict[model_type], model_path)
        joblib.dump(self.pca_dict[model_type], pca_path)
        
        # Save party distribution if it's the overall model
        if model_type == 'overall':
            party_dist_path = os.path.join(self.model_dir, "party_distribution.joblib")
            joblib.dump(self.party_distribution, party_dist_path)

    def load_model(self, model_type: str) -> bool:
        """Load trained models and their configurations"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_type}_model.joblib")
            pca_path = os.path.join(self.model_dir, f"{model_type}_pca.joblib")
            
            self.models_dict[model_type] = joblib.load(model_path)
            self.pca_dict[model_type] = joblib.load(pca_path)
            
            # Load party distribution if it's the overall model
            if model_type == 'overall':
                party_dist_path = os.path.join(self.model_dir, "party_distribution.joblib")
                self.party_distribution = joblib.load(party_dist_path)
            
            return True
        except Exception as e:
            print(f"Error loading model {model_type}: {str(e)}")
            return False

    def get_embeddings(self, statements: list) -> np.ndarray:
        """
        Get embeddings using SentenceTransformer (faster than BERT)
        """
        batch_size = 64  # Larger batch size due to more efficient model
        embeddings = []
        
        for i in tqdm(range(0, len(statements), batch_size), desc="Extracting embeddings"):
            batch = statements[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)

    def train_model(self, target_score='overall'):
        """Train model with early stopping and model saving"""
        if target_score == 'overall':
            y_train = self.datasets['train']['label'].values
            y_test = self.datasets['test']['label'].values
            rf_model = RandomForestClassifier(
                n_estimators=150, 
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
        else:
            y_train = self.datasets['train'][target_score].values
            y_test = self.datasets['test'][target_score].values
            rf_model = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

        # Dimensionality reduction
        pca = PCA(n_components=0.95)
        
        # Prepare features
        if target_score == 'overall':
            additional_features = self.datasets['train'][self.factor_scores].values
            test_additional_features = self.datasets['test'][self.factor_scores].values
        else:
            other_scores = [f for f in self.factor_scores if f != target_score]
            additional_features = self.datasets['train'][other_scores].values
            test_additional_features = self.datasets['test'][other_scores].values

        X_train = pca.fit_transform(np.hstack([self.train_embeddings, additional_features]))
        X_test = pca.transform(np.hstack([self.test_embeddings, test_additional_features]))

        # Train model
        rf_model.fit(X_train, y_train)

        # Evaluate and save model
        y_test_pred = rf_model.predict(X_test)
        if target_score == 'overall':
            test_accuracy = accuracy_score(y_test, y_test_pred)
            report = classification_report(y_test, y_test_pred)
            print(f"Test Accuracy for {target_score}: {test_accuracy:.4f}")
            print(report)
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, y_test_pred)
            r2 = r2_score(y_test, y_test_pred)
            print(f"MSE for {target_score}: {mse:.4f}")
            print(f"R2 score for {target_score}: {r2:.4f}")

        # Write accuracy results to file
        log_path = os.path.join(self.model_dir, f"{target_score}_accuracy_report.txt")
        with open(log_path, 'w') as f:
            f.write(f"Test Accuracy for {target_score}: {test_accuracy:.4f}\n")
            f.write('=====================\n\n')
            f.write(report)
            f.write('\n')
        
        return rf_model, pca

    def load_dataset_and_prepare_models(self, re_train=False):
        """Load dataset and prepare models with caching"""
        print("Loading dataset...")
        scores = []
        
        # Check if models are already trained and saved
        if self.load_model('overall') and not re_train:
            print("Loaded pre-trained models successfully!")
            return
        
        # Load and process datasets
        for dtype in ['train', 'val', 'test']:
            df = pd.read_csv(f"../data/{dtype}2.tsv", sep="\t", 
                           header=None, dtype=str).drop(columns=[0])
            df.columns = ["ID", "label", "statement", "subjects", "speaker", 
                         "speaker_job_title", "state_info", "party_affiliation", 
                         "barely_true_counts", "false_counts", "half_true_counts", 
                         "mostly_true_counts", "pants_on_fire_counts", "context", 
                         "justification"]
            df[self.count_columns] = df[self.count_columns].apply(
                pd.to_numeric, errors='coerce'
            ).fillna(0).astype(int)
            df.dropna(inplace=True)
            self.datasets[dtype] = df
        
        party_counts = self.datasets['train']['party_affiliation'].value_counts(normalize=True)
        self.party_distribution = party_counts.to_dict()

        # Calculate scores for each dataset
        for dtype, data in self.datasets.items():
            print(f"Calculating scores for {dtype} data...")
            self.datasets[dtype] = self.parallel_apply_scores(data)

        # Export datasets to TSV files with all features
        for dtype, data in self.datasets.items():
            data.to_csv(f"PredictiveAI/{dtype}_data_full.tsv", sep='\t', index=False)
            print(f"{dtype.capitalize()} dataset saved to {dtype}_data_full.tsv.")

        # Calculate and save average scores to a TSV file
        for factor in self.factor_scores:
            for dtype, data in self.datasets.items():
                average_score = data[factor].mean()
                scores.append([factor, dtype, average_score])

        scores_df = pd.DataFrame(scores, columns=['factor', 'source', 'score'])
        scores_df.to_csv('PredictiveAI/average_scores.tsv', sep='\t', index=False)
        print("Average scores saved to 'average_scores.tsv'.")

        # Process embeddings
        print("Generating embeddings for training data...")
        self.train_embeddings = self.get_embeddings(self.datasets['train']['statement'].tolist())
        print("Generating embeddings for test data...")
        self.test_embeddings = self.get_embeddings(self.datasets['test']['statement'].tolist())
        
        # Train and save models
        print("\nTraining overall model...")
        rf_model, pca = self.train_model('overall')
        self.models_dict['overall'] = rf_model
        self.pca_dict['overall'] = pca
        
        # Save trained models
        print("Saving trained models...")
        self.save_model('overall')
        print("Models saved successfully!")

    def predict_new_example(self, new_data: pd.Series) -> pd.DataFrame:
        """Predict scores for a new example"""
        if isinstance(new_data, pd.Series):
            new_data = pd.DataFrame([new_data])
            new_data.columns = ["ID", "label", "statement", "subjects", "speaker", 
                              "speaker_job_title", "state_info", "party_affiliation", 
                              "barely_true_counts", "false_counts", "half_true_counts", 
                              "mostly_true_counts", "pants_on_fire_counts", "context", 
                              "justification"]
        
        # Process the new data
        processed_df = self.process_new_datapoint(new_data)
        
        # Get embeddings for the new data
        embeddings = self.get_embeddings(processed_df['statement'].tolist())
        
        # Prepare features and predict
        additional_features = processed_df[self.factor_scores].values
        X = self.pca_dict['overall'].transform(np.hstack([embeddings, additional_features]))
        predictions = {'overall': self.models_dict['overall'].predict(X)}
        
        return pd.DataFrame(predictions)

    # Other methods remain unchanged...
    def calculate_location_score(self, statement):
        doc = self.nlp(statement)
        location_tokens = sum([len(ent) for ent in doc.ents if ent.label_ in ["GPE", "LOC"]])
        total_tokens = len(doc)
        return location_tokens / total_tokens if total_tokens > 0 else 0

    def calculate_education_score(self, row):
        total_counts = row[self.count_columns].sum()
        return row['mostly_true_counts'] / total_counts if total_counts > 0 else 0

    def calculate_event_coverage(self, statement):
        event_keywords = ['conference', 'summit', 'meeting', 'election', 'protest', 
                         'war', 'tournament', 'concert', 'festival']
        words = statement.lower().split()
        event_count = sum(word in event_keywords for word in words)
        return event_count / len(words) if words else 0

    def calculate_echo_chamber(self, party_affiliation):
        return 1 - self.party_distribution.get(party_affiliation.lower(), 0)

    def calculate_news_coverage(self, subjects):
        if not subjects:
            return 0
        subject_list = subjects.split(',')
        count = Counter(subject_list)
        n = len(subject_list)
        sum_of_squares = sum(freq**2 for freq in count.values())
        diversity_index = 1 - (sum_of_squares / (n * n))
        return diversity_index

    def calculate_malicious_account(self, row):
        total_counts = row[self.count_columns].sum()
        return row['pants_on_fire_counts'] / total_counts if total_counts > 0 else 0

    def parallel_apply_scores(self, df):
        df['location_score'] = df['statement'].apply(self.calculate_location_score)
        df['education_score'] = df.apply(self.calculate_education_score, axis=1)
        df['event_coverage_score'] = df['statement'].apply(self.calculate_event_coverage)
        df['echo_chamber_score'] = df['party_affiliation'].apply(self.calculate_echo_chamber)
        df['news_coverage_score'] = df['subjects'].apply(self.calculate_news_coverage)
        df['malicious_account_score'] = df.apply(self.calculate_malicious_account, axis=1)
        return df

    def process_new_datapoint(self, df):
        df[self.count_columns] = df[self.count_columns].apply(
            pd.to_numeric, errors='coerce'
        ).fillna(0).astype(int)
        return self.parallel_apply_scores(df)