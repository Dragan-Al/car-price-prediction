from pathlib import Path
import joblib
import pandas as pd

from src.data_cleaning import prepare_prediction_features

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "car_price_model.joblib"

def get_car_data():
    # Ask the user to enter the car characteristics.
    print("\nEnter the car characteristics")
    car = {
        "make": input("Make (car brand): ").strip(),
        "model": input("Model: ").strip(),
        "year": int(input("Year of manufacture: ")),
        "condition": input(
            "Condition (with mileage / with damage / for parts): "
        ).strip(),
        "mileage(kilometers)": float(
            input("Mileage in kilometers: ")
        ),
        "fuel_type": input(
            "Fuel type (petrol / diesel / electrocar): "
        ).strip(),
        "volume(cm3)": float(
            input("Engine volume in cm3: ")
        ),
        "color": input(
            "Color (black / silver / blue / gray / white / green / "
            "other / red / burgundy / brown / purple / yellow / orange): "
        ).strip(),
        "transmission": input(
            "Transmission (mechanics / auto): "
        ).strip(),
        "drive_unit": input(
            "Drive unit (front-wheel drive / rear drive / "
            "all-wheel drive / part-time four-wheel drive): "
        ).strip(),
        "segment": input(
            "Segment (A / B / C / D / E / F / J / M / S): "
        ).strip(),
    }


    car_df = pd.DataFrame([car])

    return prepare_prediction_features(car_df)

def main():
    #Load the trained model and predict the car price.

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run car_price_prediction.py first."
        )

    model = joblib.load(MODEL_PATH)

    car_features = get_car_data()

    predicted_price = model.predict(car_features)[0]

    print("\n" + "=" * 40)
    print(f"Estimated car price: ${predicted_price:,.2f}")
    print("=" * 40)


if __name__ == "__main__":
    main()