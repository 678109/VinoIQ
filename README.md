# 🍷 VinoIQ – Wine Quality Intelligence

VinoIQ is an interactive machine learning web application that predicts red wine quality based on physicochemical measurements.

The project was developed as part of the DAT158 Machine Learning course.

### 🌐 Live Application

**Try VinoIQ:** https://vinoiq-dat158.streamlit.app/

## Project Overview

VinoIQ uses supervised machine learning to estimate wine quality from 11 chemical measurements.

The application allows users to explore different wine profiles, generate predictions, and investigate how the trained model performs.

The project follows a machine learning workflow that includes:

1. Data collection and preparation
2. Model training and validation
3. Model comparison and selection
4. Evaluation on an independent test set
5. Deployment through an interactive web application

## Features

### Dashboard
An introduction to VinoIQ, including an overview of the dataset and model performance.

### Wine Analyzer
- Adjust 11 physicochemical measurements using interactive sliders.
- Generate predicted wine quality scores.
- Load example wines with recorded quality scores of 3, 5, and 7.
- Compare predictions with recorded scores for unchanged examples.
- Explore how different chemical measurements affect predictions.

### Model Insights
- Compare the performance of different regression models.
- View evaluation metrics, including MAE, RMSE, and R².
- Explore feature importance.
- Compare actual and predicted wine quality.
- Analyze prediction errors across quality scores.
- Investigate model performance for high-quality wines.

### About
Information about the project, machine learning methodology, dataset, and model limitations.

## Dataset

The project uses the **Wine Quality Dataset** from the UCI Machine Learning Repository.

The red wine dataset contains:

- **1,599 wine samples**
- **11 physicochemical input features**
- **1 target variable:** wine quality
- **Observed quality scores:** 3–8

The original quality scale ranges from 0 to 10.

Dataset: https://archive.ics.uci.edu/dataset/186/wine+quality

The dataset was introduced by Cortez et al. (2009).

## Machine Learning

Three regression models were compared:

| Model | Validation MAE |
|---|---:|
| Dummy Regressor (Baseline) | 0.686 |
| Ridge Regression | 0.505 |
| Random Forest Regressor | **0.439** |

**Random Forest Regressor** was selected because it achieved the lowest validation Mean Absolute Error (MAE).

The final model was retrained using the combined training and validation sets before being evaluated on an independent test set.

### Final Test Results

| Metric | Result |
|---|---:|
| MAE | **0.431** |
| RMSE | **0.558** |
| R² | **0.524** |

The model predicts numerical quality scores rather than classifying wines into fixed categories.

### Model Limitations

The model is trained on a relatively small dataset of Portuguese red wines.

Most samples have recorded quality scores of 5 or 6, while extreme scores are less common. Consequently, the model may underestimate high-quality wines and overestimate low-quality wines.

Predictions are statistical estimates based on chemical measurements and should not be interpreted as objective judgments of taste.

## Technologies

- **Python** – Programming language
- **Scikit-learn** – Machine learning models and evaluation
- **Streamlit** – Interactive web application
- **Pandas** – Data processing
- **NumPy** – Numerical operations
- **Plotly** – Interactive visualizations
- **Joblib** – Model serialization

## Project Structure

```text
VinoIQ/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── winequality-red.csv
│
├── models/
│   ├── wine_model.joblib
│   ├── metrics.json
│   └── test_predictions.csv
│
├── reports/
│   └── ...
│
└── src/
    ├── __init__.py
    └── train_model.py
```

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/678109/VinoIQ.git
cd VinoIQ
```

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will display a local address where the application can be opened in a web browser.

## Reproducing the Machine Learning Results

The repository includes the training script and dataset required to reproduce the machine learning workflow.

Run the following command from the project root:

```bash
python -m src.train_model
```

The training script:

1. Loads the red wine dataset.
2. Checks the data for missing values.
3. Splits the data into training, validation, and test sets.
4. Trains and compares three regression models.
5. Selects the model with the lowest validation MAE.
6. Retrains the selected model on the combined training and validation data.
7. Evaluates the final model on the independent test set.
8. Saves the trained model and evaluation results.

The generated model and result files are stored in the `models/` directory.

A fixed random seed is used to support reproducibility. Small differences in results may occur across software versions.

## Deployment

VinoIQ is deployed using **Streamlit Community Cloud**.

**Live application:** https://vinoiq-dat158.streamlit.app/

The application is connected to this GitHub repository and uses the saved machine learning model to generate predictions.

## Academic Context

This project was developed for the DAT158 Machine Learning course.

The main objective is to demonstrate a complete machine learning workflow, from data preparation and model evaluation to a publicly accessible web application.

## References

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.

UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/186/wine+quality

Scikit-learn documentation: https://scikit-learn.org/stable/

Streamlit documentation: https://docs.streamlit.io/

Plotly documentation: https://plotly.com/python/

## AI Tool Usage

OpenAI's ChatGPT was used to assist with code development, debugging, interface design, documentation, and report writing.

The project group is responsible for reviewing the generated material, testing the application, and verifying the final results.