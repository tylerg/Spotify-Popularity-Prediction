# Spotify Popularity Prediction

This repository provides a Python script that uses an **XGBoost Classifier** to predict whether a song on Spotify will be popular. The prediction is based on the song's audio features. The script preprocesses the data, trains the model, performs hyperparameter tuning to maximize the **Area Under the ROC Curve (AUC)**, and generates predictions.

## Dataset Requirements

The script expects the following CSV files in the working directory:

* `train.csv`: Contains song audio features and the `Popularity_Type` target label.
* `test.csv`: Contains song audio features for which predictions are to be made.

### Sample `train.csv` Structure:

| danceability | energy | loudness | speechiness | ... | Popularity_Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.812 | 0.641 | -7.222 | 0.0408 | ... | High |
| 0.588 | 0.764 | -4.948 | 0.0461 | ... | Low |

### Sample `test.csv` Structure:

| ID | danceability | energy | loudness | speechiness | ... |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.742 | 0.816 | -3.896 | 0.0516 | ... |
| 2 | 0.499 | 0.523 | -8.771 | 0.0381 | ... |

## How to Run

1.  Ensure that the required datasets (`train.csv` and `test.csv`) and the script are in the same directory.
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn xgboost scikit-learn
    ```
3.  Run the Python script:
    ```bash
    python your_script_name.py
    ```

## Model Overview

* **Data Preprocessing:** Loads the training and testing data, drops irrelevant identifier columns, and maps the `Popularity_Type` target variable to binary values (`High`: 1, `Low`: 0).
* **Model Training:** Splits the training data into training and validation sets. Trains an `XGBoost Classifier` using early stopping to prevent overfitting.
* **Hyperparameter Tuning:** Includes a `GridSearchCV` setup to find the optimal model parameters based on the `roc_auc` score.
* **Prediction & Output:** Generates popularity predictions for the test set and saves the results to `submission.csv`.
* **Visualization:** Creates and displays two plots: a feature importance plot showing the model's key drivers and feature distribution plots comparing popular and non-popular tracks.

## Key Audio Features

The model uses several audio features to make its predictions, including:
* `danceability`
* `energy`
* `loudness`
* `speechiness`
* `acousticness`
* `instrumentalness`
* `liveness`
* `valence`
* `tempo`

## Output Structure

* **Submission File (`submission.csv`):** A CSV file containing the `ID` for each test track and the corresponding predicted probability of it being popular.
* **Feature Importance Plot:** A bar chart displaying the top 10 most important features according to the trained model.
* **Feature Distribution Plots:** A set of histograms showing the distribution of key audio features, separated by popularity type.

## Example Output:
| ID | Popularity_Type |
| :--- | :--- |
| 1 | 0.8934 |
| 2 | 0.2145 |

## Implementation Notes

* The model is an **XGBoost Classifier** configured for a binary classification task (`binary:logistic`).
* The primary evaluation metric used for hyperparameter tuning and model evaluation is the **Area Under the ROC Curve (AUC)**.
* The commented-out code sections for `GridSearchCV` and feature scaling (`StandardScaler`) are provided for users who wish to extend the model's tuning and preprocessing steps.
