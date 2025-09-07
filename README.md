# Lumpy Skin Disease (LSD) Prediction using Machine Learning & Deep Learning

## Project Overview

This system leverages meteorological and geospatial data to forecast the probability of LSD occurrence, transitioning from a reactive to a **proactive disease prevention strategy**. The project is structured in three distinct phases, moving from traditional machine learning models to a more complex deep learning architecture, and finally culminating in an advanced ensemble-based prediction system with a practical veterinary decision support dashboard.

**Dataset:** [Lumpy Skin Disease Dataset](https://www.kaggle.com/datasets/brsdincer/lumpy-skin-disease-dataset)

---

## Methodology: A Three-Phase Approach

The project was developed sequentially to ensure a robust and well-validated final model.

### Phase 1: Traditional Machine Learning Baseline

* **Objective:** To establish a strong performance baseline using classical machine learning algorithms.
* **Exploratory Data Analysis (EDA):** A thorough analysis was conducted to understand the relationships between environmental factors (temperature, humidity, precipitation) and LSD cases. Key insights included the identification of seasonal patterns and geospatial hotspots.
* **Data Preprocessing:** The pipeline included handling missing values, robust categorical feature encoding (`OneHotEncoder`), and addressing significant class imbalance in the target variable using the **SMOTE (Synthetic Minority Over-sampling TEchnique)**.
* **Models Trained:**
    * Logistic Regression
    * K-Nearest Neighbors
    * Support Vector Machine (SVM)
    * Random Forest
    * Gradient Boosting
    * XGBoost
* **Results:** Tree-based models like **Random Forest** and **XGBoost** emerged as the top performers, effectively capturing the non-linear relationships in the data. Feature importance analysis consistently highlighted **humidity** and **temperature** as the most significant predictors.

### Phase 2: Deep Learning Implementation

* **Objective:** To explore if a deep learning model could capture more intricate patterns and outperform the traditional models.
* **Architecture:** A **Deep Neural Network (DNN)**, specifically a Multi-Layer Perceptron (MLP), was constructed using TensorFlow and Keras. The architecture included:
    * Multiple `Dense` hidden layers with `ReLU` activation.
    * `Dropout` layers for regularization to prevent overfitting.
    * A final `Sigmoid` activation layer for binary classification output.
* **Training:** The model was trained with an `Adam` optimizer and a `binary_crossentropy` loss function. The **EarlyStopping** callback was used to halt training when validation performance plateaued, ensuring an optimal and efficient training process.
* **Comparison:** The DNN showed competitive performance, comparable to the top traditional models, demonstrating its viability for this type of epidemiological data.

### Phase 3: Advanced Analytics & Prediction System

* **Objective:** To create a production-ready, practical tool for veterinary decision support by combining the strengths of the previous phases.
* **Ensemble Modeling:** A powerful **ensemble model** was created by averaging the prediction probabilities (soft voting) of the best-performing models from Phase 1 (Random Forest) and Phase 2 (DNN). This approach leverages the diverse learning patterns of both models to create a more robust and often more accurate final predictor.
* **Risk Assessment System:** A function, `assess_risk_and_generate_report`, was developed to encapsulate the entire pipeline. This function:
    1.  Takes new environmental data as input.
    2.  Applies the complete, saved preprocessing pipeline.
    3.  Uses the best-performing model (the ensemble) to predict the probability of an LSD outbreak.
    4.  Translates this probability into a clear, actionable risk level: **Low**, **Medium**, or **High**.

---

## Performance & Results

The models were evaluated on a variety of metrics, with a focus on **F1-Score** and **Recall** to ensure a good balance between precision and the critical need to identify as many true positive cases as possible.

| Model                   | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ----------------------- | -------- | --------- | ------ | -------- | ------- |
| **Ensemble (RF + DNN)** | **~96%** | **~95%** | **~97%** | **~0.96** | **~0.99** |
| Random Forest           | ~95%     | ~94%      | ~96%   | ~0.95    | ~0.98   |
| Deep Neural Network     | ~94%     | ~93%      | ~95%   | ~0.94    | ~0.98   |
| XGBoost                 | ~93%     | ~92%      | ~94%   | ~0.93    | ~0.97   |

*Note: Performance metrics are approximate based on a typical run and may vary slightly.*


## How to Use This Project

The entire project is contained within a single Kaggle/Jupyter Notebook. The cells are numbered and structured according to the three phases described above.

1.  **Ensure Dataset:** Make sure the `Lumpy skin disease data.csv` file is in the correct input path.
2.  **Run Sequentially:** Execute the cells in order from top to bottom.
3.  **View Results:** The output of the final cell demonstrates the decision support system with both high-risk and low-risk example scenarios.

By leveraging machine learning, this system provides a powerful tool for modern, data-driven veterinary medicine.
