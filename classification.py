#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from C45 import C45Classifier
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
for col in tqdm(dataset.columns , desc='cleaning...'):
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
print('STEP3: Encoding categorical features')
object_cols = dataset.select_dtypes(include='object').columns

label_encoders = {}

ordinal_mappings = {
    'education': ['Preschool', '1st-4th', '5th-6th', '7th-8th', 
                  '9th', '10th', '11th', '12th', 'HS-grad', 
                  'Some-college', 'Assoc-acdm', 'Assoc-voc', 
                  'Bachelors', 'Masters', 'Doctorate', 'Prof-school']
}

for col in tqdm(object_cols, desc='Encoding...'):
    if col in ordinal_mappings:
        mapping = {val: idx for idx, val in enumerate(ordinal_mappings[col])}
        dataset[col] = dataset[col].map(mapping)
    else:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])
        label_encoders[col] = le
        
# %%
print('=================Dataset Info (after encoding)=======================')
info = dataset.info()
print(info)
#%%
print('=================Dataset Describe (after encoding)=======================')
describe = dataset.describe(include='all')
print(describe)
# %%
print('STEP4: split dataset to train decision tree')
data = dataset.drop('income', axis=1)
labels = dataset['income']
data_train, data_tmp, labels_train, labels_tmp = train_test_split(data, labels, test_size=0.5, random_state=42)
data_val, data_test, labels_val, labels_test = train_test_split(data_tmp, labels_tmp, test_size=0.2, random_state=42)
#%%
print('STEP5: training  and validation models..')
print('--training id3 decision tree')
dt_id3 = DecisionTreeClassifier(criterion='entropy', max_depth=5)
dt_id3.fit(data_train, labels_train)
#%%
print('--training c4.5 decision tree')
dt_c45 = C45Classifier()
dt_c45.fit(data_train, labels_train)
#%%
print('--training cart decision tree')
dt_cart = DecisionTreeClassifier(criterion='gini', max_depth=5)
dt_cart.fit(data_train, labels_train)
#%%
print('--training Naive Bayesian decision tree')
nb = GaussianNB()
nb.fit(data_train, labels_train)
#%% 
print('--Cross-Validation')
def manual_cross_val(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Clone model by re-fitting
        if isinstance(model, type):
            # If model is a class, create new instance
            model_copy = model.__class__()
        else:
            # Otherwise, create new instance with same parameters
            model_copy = model.__class__(**model.get_params() if hasattr(model, 'get_params') else {})
        
        model_copy.fit(X_train_fold, y_train_fold)
        y_pred = model_copy.predict(X_val_fold)
        score = accuracy_score(y_val_fold, y_pred)
        scores.append(score)
    
    return np.array(scores)

models = {'ID3': dt_id3, 'C4.5': dt_c45, 'CART': dt_cart, 'Naive Bayes': nb}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Handle cross-validation differently for non-sklearn models
    if name == 'C4.5':
        # Manual cross-validation for C4.5
        cv_scores = manual_cross_val(model, data_val, labels_val, cv=5)
    else:
        # Standard cross-validation for sklearn models
        cv_scores = cross_val_score(model, data_val, labels_val, cv=kfold, scoring='accuracy')
    
    results[name] = {
        'CV_Accuracy': cv_scores.mean(),
        'CV_Std': cv_scores.std()
    }
    
    # Make predictions on test set
    labels_pred = model.predict(data_test)
    
    # Calculate metrics
    results[name]['Accuracy'] = accuracy_score(labels_test, labels_pred)
    results[name]['Precision'] = precision_score(labels_test, labels_pred)
    results[name]['Recall'] = recall_score(labels_test, labels_pred)
    results[name]['F1'] = f1_score(labels_test, labels_pred)
#%%  
print('STEP6: Analysis models')  
feature_importance = pd.DataFrame({
    'feature': data.columns,
    'importance_id3': dt_id3.feature_importances_,
    #'importance_c4.5': dt_c45,
    'importance_cart': dt_cart.feature_importances_,
})
top_features = feature_importance['feature'][:10]

plt.figure(figsize=(10,6))
plt.plot(top_features, feature_importance['importance_id3'][:10], label='ID3' , c = 'black')
#plt.plot(top_features, feature_importance['importance_c4.5'][:10], label='C4.5', c = 'blue')
plt.plot(top_features, feature_importance['importance_cart'][:10], label='CART', c = 'red')

plt.legend()
plt.xticks(rotation=45)
plt.title('Feature Importance Comparison')
plt.show()

#%%
results_df = pd.DataFrame(results).T
print(results_df)

results_df[['Accuracy', 'Precision', 'Recall', 'F1']].plot(kind='bar', figsize=(12,6))
plt.title('Model Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.show()
# %%
