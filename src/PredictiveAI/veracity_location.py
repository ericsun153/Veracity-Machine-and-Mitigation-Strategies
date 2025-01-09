import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import spacy
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch
import concurrent.futures
import xgboost as xgb
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# Load spaCy's English NER model
nlp = spacy.load("en_core_web_sm")

def contains_location(statement):
    """
    Uses spaCy's NER to detect if there is a location entity in the given statement.
    Returns 1 if a location is detected, otherwise 0.
    """
    doc = nlp(statement)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return 1
    return 0

def parallel_apply_contains_location(statements):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(contains_location, statements), total=len(statements), desc="Detecting geographic features"))
    return results

def get_bert_embeddings(statements, tokenizer, model, batch_size=32, device='cuda'):
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(statements), batch_size), desc="Extracting BERT embeddings"):
            batch_statements = statements[i:i+batch_size]
            encoded_input = tokenizer(batch_statements, padding=True, truncation=True, max_length=128, return_tensors='pt')
            encoded_input = {key: val.to(device) for key, val in encoded_input.items()}
            outputs = model(**encoded_input)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Load the dataset files
print("Loading dataset...")
train_data = pd.read_csv("LIAR-PLUS/dataset/tsv/train2.tsv", sep="\t", header=None, dtype=str).drop(columns=[0])
val_data = pd.read_csv("LIAR-PLUS/dataset/tsv/val2.tsv", sep="\t", header=None, dtype=str).drop(columns=[0])
test_data = pd.read_csv("LIAR-PLUS/dataset/tsv/test2.tsv", sep="\t", header=None, dtype=str).drop(columns=[0])

# Assign column names
column_names = [
    "ID", "label", "statement", "subjects", "speaker", "speaker_job_title", 
    "state_info", "party_affiliation", "barely_true_counts", "false_counts", 
    "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", 
    "context", "justification"
]
train_data.columns = val_data.columns = test_data.columns = column_names

# Drop rows with missing values from each dataset
train_data = train_data.dropna()
val_data = val_data.dropna()
test_data = test_data.dropna()

# Define a dictionary for mapping the label values
label_mapping = {
    'pants-fire': 0,  # Changed to zero-based index
    'false': 1,
    'barely-true': 2,
    'half-true': 3,
    'mostly-true': 4,
    'true': 5
}

# Map the original labels to numerical values
train_data['label'] = train_data['label'].map(label_mapping)
val_data['label'] = val_data['label'].map(label_mapping)
test_data['label'] = test_data['label'].map(label_mapping)

# Reverse mapping for decoding
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Preprocess the categorical data
def preprocess_categorical(data):
    data['speaker'] = data['speaker'].str.replace('-', ' ')
    data[['speaker_first_name', 'speaker_last_name']] = data['speaker'].str.split(n=1, expand=True)
    data['party_affiliation'] = data['party_affiliation'].str.replace('-', '_')
    data['subjects'] = data['subjects'].str.replace('-', '_')
    return data

train_data = preprocess_categorical(train_data)
val_data = preprocess_categorical(val_data)
test_data = preprocess_categorical(test_data)

# Adding a feature for the presence of location entities in the statement
print("Checking for geographic features in statements...")
train_data['geo_feature'] = parallel_apply_contains_location(train_data['statement'].tolist())
val_data['geo_feature'] = parallel_apply_contains_location(val_data['statement'].tolist())
test_data['geo_feature'] = parallel_apply_contains_location(test_data['statement'].tolist())

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train_statement = get_bert_embeddings(train_data['statement'].tolist(), tokenizer, model, device=device)
X_val_statement = get_bert_embeddings(val_data['statement'].tolist(), tokenizer, model, device=device)
X_test_statement = get_bert_embeddings(test_data['statement'].tolist(), tokenizer, model, device=device)

# MultiLabelBinarizer for subjects
mlb = MultiLabelBinarizer()
X_train_subjects = mlb.fit_transform(train_data['subjects'].apply(lambda x: x.split(',')))
X_val_subjects = mlb.transform(val_data['subjects'].apply(lambda x: x.split(',')))
X_test_subjects = mlb.transform(test_data['subjects'].apply(lambda x: x.split(',')))

# OneHotEncoder for other categorical features
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(train_data[['speaker_first_name', 'speaker_last_name', 'party_affiliation']])

X_train_categorical = np.hstack([
    X_train_subjects,
    encoder.transform(train_data[['speaker_first_name', 'speaker_last_name', 'party_affiliation']]).toarray()
])
X_val_categorical = np.hstack([
    X_val_subjects,
    encoder.transform(val_data[['speaker_first_name', 'speaker_last_name', 'party_affiliation']]).toarray()
])
X_test_categorical = np.hstack([
    X_test_subjects,
    encoder.transform(test_data[['speaker_first_name', 'speaker_last_name', 'party_affiliation']]).toarray()
])

# Combine all features
X_train = np.hstack([X_train_statement, X_train_categorical, train_data['geo_feature'].values.reshape(-1, 1)])
X_val = np.hstack([X_val_statement, X_val_categorical, val_data['geo_feature'].values.reshape(-1, 1)])
X_test = np.hstack([X_test_statement, X_test_categorical, test_data['geo_feature'].values.reshape(-1, 1)])

# PCA for dimensionality reduction
pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
X_test = pca.transform(X_test)

# Encoding labels
y_train = train_data['label'].values
y_val = val_data['label'].values
y_test = test_data['label'].values

# Regularization: Set higher regularization (lower C) to reduce overfitting
logistic_model = LogisticRegression(max_iter=1000, C=0.01, class_weight='balanced')

# XGBoost without early stopping
xgb_model = xgb.XGBClassifier(
    eval_metric='mlogloss',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4
)

# Random Forest with limited depth and higher number of estimators
rf_model = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)

# SGDClassifier with increased regularization and early stopping
sgd_model = SGDClassifier(loss='hinge', alpha=0.001, class_weight='balanced', max_iter=1000, tol=1e-3, early_stopping=True, validation_fraction=0.1)

def train_and_evaluate_model(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    return accuracy_score(y_train, y_train_pred), accuracy_score(y_val, y_val_pred)

# Train models
log_train_acc, log_val_acc = train_and_evaluate_model(logistic_model, X_train, X_val, y_train, y_val)
print(f"Logistic Regression - Train Accuracy: {log_train_acc:.4f}, Validation Accuracy: {log_val_acc:.4f}")

xgb_train_acc, xgb_val_acc = train_and_evaluate_model(xgb_model, X_train, X_val, y_train, y_val)
print(f"XGBoost - Train Accuracy: {xgb_train_acc:.4f}, Validation Accuracy: {xgb_val_acc:.4f}")

rf_train_acc, rf_val_acc = train_and_evaluate_model(rf_model, X_train, X_val, y_train, y_val)
print(f"Random Forest - Train Accuracy: {rf_train_acc:.4f}, Validation Accuracy: {rf_val_acc:.4f}")

sgd_train_acc, sgd_val_acc = train_and_evaluate_model(sgd_model, X_train, X_val, y_train, y_val)
print(f"SGDClassifier (approximated SVM) - Train Accuracy: {sgd_train_acc:.4f}, Validation Accuracy: {sgd_val_acc:.4f}")

# Make predictions for the test set using logistic regression
y_test_pred_proba = logistic_model.predict_proba(X_test)
y_test_pred = np.argmax(y_test_pred_proba, axis=1)

# Decode the predicted and actual labels back to their original classes
y_test_pred_labels = [reverse_label_mapping[label] for label in y_test_pred]
y_test_actual_labels = [reverse_label_mapping[label] for label in y_test]

# Create the JSON output structure
results = []
for i in range(len(y_test)):
    result = {
        "ID": test_data.iloc[i]["ID"],
        "statement": test_data.iloc[i]["statement"],
        "predicted_veracity_score": int(y_test_pred[i]),
        "actual_veracity_score": int(y_test[i]),
        "predicted_label": y_test_pred_labels[i],
        "actual_label": y_test_actual_labels[i],
        "prediction_probabilities": y_test_pred_proba[i].tolist()
    }
    results.append(result)

# Write the results to a JSON file
output_filename = "veracity_predictions_with_location.json"
with open(output_filename, "w") as json_file:
    json.dump(results, json_file, indent=4)

print(f"Predictions saved to {output_filename}.")
