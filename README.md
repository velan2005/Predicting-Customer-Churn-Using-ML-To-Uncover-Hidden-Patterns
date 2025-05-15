
# Customer Churn Prediction Using Machine Learning

## Project Overview

This project uses machine learning to predict customer churn, helping businesses identify customers at risk of leaving. A RandomForestClassifier model is trained on a processed dataset, providing insights into key factors influencing customer retention.

## Features

* Customer churn prediction using machine learning.
* RandomForestClassifier model for accurate classification.
* Data preprocessing with encoding for categorical features.
* Binary representation of churn (1 for churn, 0 for not).
* Model evaluation using a confusion matrix and classification report.
* Visualization of feature importance for better understanding.

## Project Structure

* **customer\_churn.csv:** Dataset used for training and testing the model.
* **code.py:** Main script for data processing, model training, and evaluation.

## Installation and Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the model:

   ```bash
   python code.py
   ```

## Methodology

* Data is loaded and preprocessed using Pandas.
* Categorical features are encoded using one-hot encoding.
* The dataset is split into training and testing sets.
* A RandomForestClassifier is trained on the training set.
* Model performance is evaluated using a confusion matrix and classification report.
* Feature importance is visualized using Seaborn.

## Results

* The model provides a classification report with precision, recall, and F1-score.
* Feature importance graph highlights the most influential factors.

---

