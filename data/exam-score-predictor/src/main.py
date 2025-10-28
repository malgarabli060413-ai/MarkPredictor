# main.py

import pandas as pd
from src.data.processing import preprocess_data

def main():
    # Load the dataset
    df = pd.read_csv('data.csv')  # Replace with the actual path to your dataset
    print("✅ Dataset loaded successfully!")

    # Preprocess the data
    processed_df = preprocess_data(df)

    # Here you can add logic to train your model or make predictions
    # For example:
    # model = train_model(processed_df)
    # predictions = model.predict(new_data)

    print("🎉 Processing and model execution completed successfully!")

if __name__ == "__main__":
    main()