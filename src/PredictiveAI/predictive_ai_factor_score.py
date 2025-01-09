import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel
import concurrent.futures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import warnings
import math
from collections import Counter

warnings.filterwarnings("ignore")

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_sm")

def calculate_location_score(statement):
    doc = nlp(statement)
    location_tokens = sum([len(ent) for ent in doc.ents if ent.label_ in ["GPE", "LOC"]])
    total_tokens = len(doc)
    return location_tokens / total_tokens if total_tokens > 0 else 0

def calculate_education_score(row):
    total_counts = row[['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']].sum()
    return row['mostly_true_counts'] / total_counts if total_counts > 0 else 0

def calculate_event_coverage(statement):
    event_keywords = ['conference', 'summit', 'meeting', 'election', 'protest', 'war', 'tournament', 'concert', 'festival']
    words = statement.lower().split()
    event_count = sum(word in event_keywords for word in words)
    return event_count / len(words) if words else 0

def calculate_echo_chamber(party_affiliation, party_distribution):
    return 1 - party_distribution.get(party_affiliation.lower(), 0)

def calculate_news_coverage(subjects):
    from collections import Counter
    if not subjects:
        return 0
    subject_list = subjects.split(',')
    count = Counter(subject_list)
    n = len(subject_list)
    sum_of_squares = sum(freq**2 for freq in count.values())
    diversity_index = 1 - (sum_of_squares / (n * n))
    return diversity_index

def calculate_malicious_account(row):
    total_counts = row[['barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts']].sum()
    return row['pants_on_fire_counts'] / total_counts if total_counts > 0 else 0

def parallel_apply_scores(df, party_distribution):
    df['location_score'] = df['statement'].apply(calculate_location_score)
    df['education_score'] = df.apply(calculate_education_score, axis=1)
    df['event_coverage_score'] = df['statement'].apply(calculate_event_coverage)
    df['echo_chamber_score'] = df['party_affiliation'].apply(lambda x: calculate_echo_chamber(x, party_distribution))
    df['news_coverage_score'] = df['subjects'].apply(calculate_news_coverage)
    df['malicious_account_score'] = df.apply(calculate_malicious_account, axis=1)
    return df

# Load and prepare the dataset
print("Loading dataset...")
datasets = {}
scores = []
for dtype in ['train', 'val', 'test']:
    df = pd.read_csv(f"LIAR-PLUS/dataset/tsv/{dtype}2.tsv", sep="\t", header=None, dtype=str).drop(columns=[0])
    df.columns = ["ID", "label", "statement", "subjects", "speaker", "speaker_job_title", "state_info", "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context", "justification"]
    # Convert count columns to integers
    count_columns = ["barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts"]
    df[count_columns] = df[count_columns].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    df.dropna(inplace=True)
    datasets[dtype] = df

# Calculate party distribution from the training set for echo chamber score
party_counts = datasets['train']['party_affiliation'].value_counts(normalize=True)
party_distribution = party_counts.to_dict()

# Calculate scores for each dataset
for dtype, data in datasets.items():
    print(f"Calculating scores for {dtype} data...")
    datasets[dtype] = parallel_apply_scores(data, party_distribution)

# Export datasets to TSV files with all features
for dtype, data in datasets.items():
    data.to_csv(f"{dtype}_data_full.tsv", sep='\t', index=False)
    print(f"{dtype.capitalize()} dataset saved to {dtype}_data_full.tsv.")

# Calculate and save average scores to a TSV file
factors = ['location_score', 'education_score', 'event_coverage_score', 'echo_chamber_score', 'news_coverage_score', 'malicious_account_score']
for factor in factors:
    for dtype, data in datasets.items():
        average_score = data[factor].mean()
        scores.append([factor, dtype, average_score])

scores_df = pd.DataFrame(scores, columns=['factor', 'source', 'score'])
scores_df.to_csv('average_scores.tsv', sep='\t', index=False)
print("Average scores saved to 'average_scores.tsv'.")

# Machine Learning Model Preparation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_bert_embeddings(statements):
    """
    Extracts BERT embeddings for a list of statements.
    """
    model.to(device)
    model.eval()
    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(statements), batch_size), desc="Extracting BERT embeddings"):
            batch_statements = statements[i:i+batch_size]
            encoded_input = tokenizer(batch_statements, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            outputs = model(**encoded_input)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Process embeddings for training and testing
train_embeddings = get_bert_embeddings(datasets['train']['statement'].tolist())
test_embeddings = get_bert_embeddings(datasets['test']['statement'].tolist())

# Dimensionality reduction
pca = PCA(n_components=0.95)
X_train = pca.fit_transform(np.hstack([train_embeddings, datasets['train'][['location_score']].values]))
X_test = pca.transform(np.hstack([test_embeddings, datasets['test'][['location_score']].values]))

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)
rf_model.fit(X_train, datasets['train']['label'].values)

# Make predictions and evaluate
y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(datasets['test']['label'].values, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(classification_report(datasets['test']['label'].values, y_test_pred))
