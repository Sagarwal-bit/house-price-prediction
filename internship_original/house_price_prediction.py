import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    print("--- Skillbit Project: House Price Prediction ---")
    
    # 1. Create dummy dataset
    np.random.seed(42)
    num_samples = 100
    
    # Features: Square Footage (1000 to 5000), Bedrooms (1 to 5), Age in years (0 to 50)
    sqft = np.random.randint(1000, 5000, num_samples)
    bedrooms = np.random.randint(1, 6, num_samples)
    age = np.random.randint(0, 51, num_samples)
    
    # Base price calculation with some random noise
    # $150 per sqft + $10000 per bedroom - $2000 per year of age + random noise
    base_price = (sqft * 150) + (bedrooms * 10000) - (age * 2000)
    noise = np.random.normal(0, 20000, num_samples) # Add some realistic variation
    prices = base_price + noise
    
    data = {
        'Square_Footage': sqft,
        'Bedrooms': bedrooms,
        'Age_Years': age,
        'Price': prices
    }
    
    df = pd.DataFrame(data)
    print("\nDataset loaded (Dummy Data - First 5 rows):")
    print(df.head())
    
    # 2. Split dataset into Features (X) and Target (y)
    X = df[['Square_Footage', 'Bedrooms', 'Age_Years']]
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train the Model (Linear Regression)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 4. Evaluate the Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Mean Squared Error (MSE): ${mse:,.2f}")
    print(f"R-squared Score: {r2:.4f}")
    
    print("\nModel Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f" - {feature}: ${coef:,.2f}")
    
    # 5. Predict on new, unseen data
    print("\nPredicting prices for new listings:")
    new_houses = pd.DataFrame({
        'Square_Footage': [2500, 1500, 4000],
        'Bedrooms': [3, 2, 5],
        'Age_Years': [10, 30, 5]
    })
    
    predictions = model.predict(new_houses)
    
    for i, (idx, row) in enumerate(new_houses.iterrows()):
        print(f"House {i+1} ({row['Square_Footage']} sqft, {row['Bedrooms']} beds, {row['Age_Years']} years old) -> Predicted Price: ${predictions[i]:,.2f}")

if __name__ == "__main__":
    main()
