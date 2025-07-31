import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from google.colab import files
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# train id column is named "id" and test "ID"
def preprocess_data(df, is_train=True):
    identifiers = ['track_href', 'uri', 'type',
                 'track_album_release_date', 'analysis_url', 'id']
    if not is_train:
        identifiers = [col if col != 'id' else 'ID' for col in identifiers]

    df_clean = df.drop(columns=identifiers)

    if is_train:
        X = df_clean.drop(columns=['Popularity_Type'])
        y = df_clean['Popularity_Type'].map({'High': 1, 'Low': 0})
        return X, y
    return df_clean

X_train, y_train = preprocess_data(train_df)

X_test = preprocess_data(test_df, is_train=False)

test_ids = test_df['ID']

model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    learning_rate=0.05,
    max_depth=6,
    n_estimators=2000,
    subsample=0.7,
    colsample_bytree=0.7,
    early_stopping_rounds=11,
    random_state=12,
)


# class_ratio = np.mean(y_train)
# print(f"Positive class ratio: {class_ratio:.2f}")


# model.set_params(scale_pos_weight=(1-class_ratio)/class_ratio)

param_grid = {
    'learning_rate': [0.03, 0.05],
    'max_depth': [6, 7],
    'objective': ['binary:logistic'],
    'eval_metric': ['auc'],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9],
    'early_stopping_rounds': [10, 12],
    'random_state': [12],
    'n_estimators': [2000, 2400]
}


# Add this after getting X_train/X_test but before model training
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Use these scaled versions for all model operations
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.3, random_state=12
)

# search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
# search.fit(X_train, y_train, **{'eval_set': [(X_val_split, y_val_split)]})

# print(f"Best params: {search.best_params_}")
# print(f"Best CV AUC: {search.best_score_:.4f}")

# model = search.best_estimator_



# verbose prints auc each boost stage
model.fit(X_train_split, y_train_split,
          eval_set=[(X_val_split, y_val_split)],
          verbose=1)

y_pred_proba = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    'ID': test_ids,
    'Popularity_Type': y_pred_proba
})

submission.to_csv('submission.csv', index=False)
print(submission.head())

plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=10, importance_type="weight")
plt.title("Top 10 Feature Importances")
plt.show()

features_to_plot = ['danceability', 'energy', 'speechiness', 'tempo', 'loudness']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=train_df, x=feature, hue='Popularity_Type', bins=30, kde=True, palette="coolwarm")
    plt.title(f'Distribution of {feature} by Popularity')
plt.tight_layout()
plt.show()