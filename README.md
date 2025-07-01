# Credit Card Fraud Detection

This project demonstrates a machine learning workflow for detecting fraudulent credit card transactions using a public dataset. The analysis is performed in a Jupyter Notebook and leverages Python's data science libraries.

## Dataset
- **File:** `creditcard.csv`
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Description:** Contains transactions made by European cardholders in September 2013. The dataset presents transactions that occurred in two days, with 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.
- **Features:**
  - `Time`, `Amount`, `V1`-`V28` (anonymized principal components), `Class` (0: normal, 1: fraud)

## Workflow
1. **Data Loading & Exploration:**
   - Load the dataset and inspect its structure, missing values, and class distribution.
2. **Data Preprocessing:**
   - Under-sample the majority class to balance the dataset for model training.
   - Visualize the distribution of classes and transaction characteristics.
3. **Feature Selection:**
   - Separate features (`X`) and target (`Y`).
4. **Train-Test Split:**
   - Split the balanced data into training and testing sets (80/20 split, stratified).
5. **Model Training:**
   - Train a Random Forest Classifier on the training data.
6. **Evaluation:**
   - Evaluate the model's accuracy on both training and test sets.

## Results
- **Training Accuracy:** ~98.9%
- **Test Accuracy:** ~92.4%

## Requirements
- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required packages using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage
1. Place `creditcard.csv` in the project directory.
2. Open `creditcard.ipynb` in Jupyter Notebook.
3. Run the notebook cells sequentially to reproduce the analysis and results.

## Notes
- The dataset is highly imbalanced; this notebook uses under-sampling for demonstration. For production systems, consider more advanced techniques (e.g., SMOTE, ensemble methods).
- The features `V1`-`V28` are anonymized due to confidentiality.

## License
This project is for educational purposes. Please refer to the [Kaggle dataset license](https://www.kaggle.com/mlg-ulb/creditcardfraud) for dataset usage rights. 
