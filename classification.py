#%% Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from chefboost import Chefboost as chef
import matplotlib.pyplot as plt
from collections import Counter
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# %%
print('STEP1: loading dataset...')
dataset = pd.read_csv('adult_income_dataset.csv')
dataset.replace('?', np.nan, inplace=True)
print('STEP2: Handling missing values...')
# %%
print('=================Dataset Info (before cleaning)=======================')
info = dataset.info()
print(info)
#%%
print('=================Dataset Describe (before cleaning)=======================')
describe = dataset.describe(include='all')
print(describe)
#%%
print('Replace NaN values with mode (for object) or mean (for numeric) columns')
for col in tqdm(dataset.columns, desc='Cleaning...'):
    if dataset[col].isnull().sum() > 0:
        if dataset[col].dtype == 'object':
            dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
        else:
            dataset[col] = dataset[col].fillna(dataset[col].mean())

# %%
print('=================Dataset Info (after cleaning)=======================')
info = dataset.info()
print(info)
#%%
print('=================Dataset Describe (after cleaning)=======================')
describe = dataset.describe(include='all')
print(describe)
#%%
#%% Encode categorical features with train-test alignment
def encode_categorical_features(dataset):
    print('STEP 1: Encoding target variable')

    # Binary target mapping
    if 'income' in dataset.columns:
        dataset['income'] = dataset['income'].map({'<=50K': 0, '>50K': 1})

    print('STEP 2: Identifying categorical columns')
    object_cols = dataset.select_dtypes(include='object').columns

    # Ordinal mappings
    ordinal_mappings = {
        'education': ['Preschool', '1st-4th', '5th-6th', '7th-8th',
                      '9th', '10th', '11th', '12th', 'HS-grad',
                      'Some-college', 'Assoc-acdm', 'Assoc-voc',
                      'Bachelors', 'Masters', 'Doctorate', 'Prof-school']
    }

    print('STEP 3: Encoding ordinal features')
    for col in ordinal_mappings:
        if col in dataset.columns:
            mapping = {val: idx for idx, val in enumerate(ordinal_mappings[col])}
            dataset[col] = dataset[col].map(mapping)

    # Nominal columns (exclude ordinals)
    nominal_cols = [col for col in object_cols if col not in ordinal_mappings]

    print(f'STEP 4: One-hot encoding nominal features: {nominal_cols}')
    if nominal_cols:
        dataset = pd.get_dummies(
            dataset,
            columns=nominal_cols,
            drop_first=True
        )

    print('=================Dataset Info (after encoding)=======================')
    info = dataset.info()
    print(info)
    
    print('=================Dataset Describe (after encoding)==================')
    describe = dataset.describe(include='all')
    print(describe)
    
    return dataset


#%% Split dataset
train_val, test = train_test_split(dataset, test_size=0.15, stratify=dataset['income'], random_state=42)
train, val = train_test_split(train_val, test_size=0.1765, stratify=train_val['income'], random_state=42)
#70% train 15% val 15% test 
#%% Encode dataset for Naive Bayes
nb_dataset = encode_categorical_features(dataset)
nb_train_val, test_nb = train_test_split(nb_dataset, test_size=0.15, stratify=dataset['income'], random_state=42)
train_nb, val_nb = train_test_split(nb_train_val, test_size=0.1765, stratify=train_val['income'], random_state=42)
#%% Train models on full training set
def chefboost_feature_importance(rules_file, feature_names):
    with open(rules_file, 'r') as f:
        rules = f.read()
    
    counts = Counter()
    for feature in feature_names:
        pattern = rf"\b{re.escape(feature)}\b"
        counts[feature] = len(re.findall(pattern, rules))
    
    total = sum(counts.values())
    importance = {k: v/total if total > 0 else 0 for k, v in counts.items()}
    
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

final_models = {}
feature_importances = {}

X_features = train.drop('income', axis=1).columns.tolist()

for algo in ['ID3', 'C4.5', 'CART']:
    model = chef.fit(train, config={'algorithm': algo}, target_label='income')
    final_models[algo] = model
    rules_path = f"outputs/rules/rules.py"  
    
    feature_importances[algo] = chefboost_feature_importance(
        rules_path,
        X_features
    )
#%%
# Train Naive Bayes 
X_train_nb = train_nb.drop('income', axis=1)
y_train_nb = train_nb['income']
nb = GaussianNB()
nb.fit(X_train_nb, y_train_nb)
final_models['NaiveBayes'] = nb

importances_nb = {}
for i, col in enumerate(X_train_nb.columns):
    mean0 = X_train_nb[y_train_nb==0][col].mean()
    mean1 = X_train_nb[y_train_nb==1][col].mean()
    importances_nb[col] = abs(mean1 - mean0)
feature_importances['NaiveBayes'] = importances_nb
#%% Cross Validation (5 folds) for all models
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
algorithms = ['ID3', 'C4.5', 'CART'] 
X_train = train.drop('income', axis=1) 
y_train = train['income'] 
cv_results = {} 
for algo in algorithms: 
    fold_accuracies = [] 
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1): 
        fold_train = train.iloc[train_idx] 
        fold_val = train.iloc[val_idx] 
        model = chef.fit(fold_train, config={'algorithm': algo}, target_label='income') 
        y_true = fold_val['income'] 
        y_pred = [chef.predict(model, row) for _, row in fold_val.iterrows()] 
        fold_accuracies.append(accuracy_score(y_true, y_pred)) 
        cv_results[algo] = fold_accuracies
#%%
_val = val_nb.drop('income' , axis = 1)
val_labels = val_nb['income']     
scores = cross_val_score(
    nb,
    _val,
    val_labels,
    cv=kf,
    scoring='accuracy'
)
cv_results['NaiveBayes'] = scores


#%% Test 
metrics = {}
for algo, model in final_models.items():
    if algo != 'NaiveBayes':
        X_test = test.drop('income', axis=1)
        y_test = test['income']
        y_pred = [chef.predict(model, row) for _, row in X_test.iterrows()]
        y_test_num = y_test.map({'<=50K':0, '>50K':1})
        y_pred_num = pd.Series(y_pred).map({'<=50K':0, '>50K':1})
        y_true, y_pred_final = y_test_num, y_pred_num
    else:
        X_test = test_nb.drop('income', axis=1)
        y_test = test_nb['income']
        y_pred_final = model.predict(X_test)
        y_true = y_test
    
    metrics[algo] = {
        'Accuracy': accuracy_score(y_true, y_pred_final),
        'Precision': precision_score(y_true, y_pred_final),
        'Recall': recall_score(y_true, y_pred_final),
        'F1-Score': f1_score(y_true, y_pred_final)
    }

#%% Display metrics table
metrics_df = pd.DataFrame(metrics).T
print("\n===== Test Set Metrics =====")
print(metrics_df)

#%% Display feature importance
for algo, feats in feature_importances.items():
    print(f"\n===== Feature Importance: {algo} =====")
    sorted_feats = dict(sorted(feats.items(), key=lambda x: x[1], reverse=True))

    top_3 = list(sorted_feats.items())[:3]

    for i, (feat, val) in enumerate(top_3, 1):
        print(f"{i}. {feat}: {val:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    features = [item[0] for item in top_3]
    importances = [item[1] for item in top_3]

    bars = ax.bar(features, importances, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])

    ax.set_title(f'Top 3 Feature Importance: {algo}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_ylim(0, max(importances) * 1.1)  

    for bar, val in zip(bars, importances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

num_algorithms = len(feature_importances)
fig, axes = plt.subplots(1, num_algorithms, figsize=(5*num_algorithms, 6))

if num_algorithms == 1:
    axes = [axes]

for ax, (algo, feats) in zip(axes, feature_importances.items()):
    sorted_feats = dict(sorted(feats.items(), key=lambda x: x[1], reverse=True))
    top_3 = list(sorted_feats.items())[:3]
    
    features = [item[0] for item in top_3]
    importances = [item[1] for item in top_3]

    bars = ax.bar(features, importances, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax.set_title(f'{algo}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Features', fontsize=10)
    ax.set_ylabel('Importance', fontsize=10)
    ax.tick_params(axis='x', rotation=45)

    for bar, val in zip(bars, importances):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

plt.suptitle('Top 3 Feature Importances for All Algorithms', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()

#%% Plot CV accuracies
plt.figure(figsize=(10,5))
for algo, accs in cv_results.items():
    plt.plot(range(1, len(accs)+1), accs, marker='o', label=f'{algo} CV folds')
plt.title('Cross Validation Accuracies')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.ylim(0.7,1.0)
plt.grid(True)
plt.legend()
plt.show()

#%% Plot Test Metrics
plt.figure(figsize=(10,6))
metrics_types = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for i, metric in enumerate(metrics_types):
    plt.bar(
        [x + i*0.2 for x in range(len(metrics))],
        [metrics[algo][metric] for algo in metrics],
        width=0.2,
        label=metric
    )
plt.xticks([r + 0.3 for r in range(len(metrics))], metrics.keys())
plt.ylim(0,1.0)
plt.title('Test Set Classification Metrics')
plt.ylabel('Score')
plt.legend()
plt.show()
# %%
