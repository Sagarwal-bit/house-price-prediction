# Project Report: House Price Prediction

**Domain:** Machine Learning  
**Project ID:** 6  
**Date:** June 2026

## 1. Project Overview
The objective of this project was to build a machine learning regression model to predict the monetary value of a house based on quantitative features. This serves as a foundational demonstration of structured data analysis and predictive modeling.

## 2. Dataset
Following the internship guidelines, a synthetic (dummy) dataset of 100 housing records was generated using `numpy`. The features included:
- **Square Footage:** The size of the house (ranging from 1000 to 5000 sq ft).
- **Bedrooms:** The number of bedrooms (1 to 5).
- **Age_Years:** The age of the property (0 to 50 years).
- **Price:** The target variable, calculated using a base formula with added Gaussian noise to simulate real-world data variance.

## 3. Technology Stack
- **Language:** Python
- **Libraries:**
  - `numpy` & `pandas`: Data generation, manipulation, and structuring.
  - `scikit-learn`: Machine learning implementation (Linear Regression, Evaluation metrics).

## 4. Methodology
1. **Data Preparation:** The synthetic data was compiled into a pandas DataFrame, separating the input features (X) from the target output (y).
2. **Train/Test Split:** The data was split into an 80% training set and a 20% testing set to evaluate generalization performance.
3. **Model Selection:** `LinearRegression` was chosen. It is a fundamental and highly interpretable algorithm for modeling the relationship between a continuous dependent variable (Price) and several independent variables.
4. **Training & Evaluation:** The model was fit to the training data. Predictions were then made on the testing data and evaluated using **Mean Squared Error (MSE)** and the **R-squared ($R^2$) Score**.

## 5. Results & Conclusion
The Linear Regression model performed exceptionally well on the synthetic dataset, yielding an extremely high $R^2$ score (close to 1.0), meaning it successfully learned the underlying mathematical relationship used to generate the prices. The model correctly identified that Square Footage and Bedrooms positively impact price, while Age negatively impacts it. It was then successfully deployed to predict realistic prices for completely new, unseen housing configurations.
