# nse_stock_predictor_streamlit.py

import yfinance as yf
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def fetch_nifty100_symbols():
    try:
        cleaned_file = "cleaned_nifty100list.csv"
        try:
            df = pd.read_csv(cleaned_file)
        except FileNotFoundError:
            st.warning("Cleaned symbol list not found. Running validation...")
            df = pd.read_csv("ind_nifty100list.csv")
            valid_symbols = []
            for symbol in df['Symbol'].dropna().unique():
                if validate_symbol(symbol):
                    valid_symbols.append(symbol)
            df = pd.DataFrame(valid_symbols, columns=['Symbol'])
            df.to_csv(cleaned_file, index=False)
            st.info(f"Saved cleaned symbol list to {cleaned_file}")
        return [symbol + '.NS' for symbol in df['Symbol'].tolist()]
    except Exception as e:
        st.error(f"Error reading or validating Nifty 100 symbols: {e}")
        return []


def validate_symbol(symbol):
    try:
        test = yf.download(symbol + ".NS", period="1d", interval="1d", progress=False)
        return not test.empty
    except:
        return False


def fetch_stock_data(symbol):
    try:
        data = yf.download(symbol, period="6mo", interval="1d", progress=False, auto_adjust=False)
        if data.empty:
            raise ValueError("Empty data fetched")
        data.dropna(inplace=True)
        return data
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return None


def prepare_features(data):
    data['Return'] = data['Close'].pct_change()
    data['Lag1'] = data['Return'].shift(1)
    data['Lag2'] = data['Return'].shift(2)
    data['Lag3'] = data['Return'].shift(3)
    data.dropna(inplace=True)
    return data


def train_and_predict_all_models(data):
    X = data[['Lag1', 'Lag2', 'Lag3']]
    y = data['Return']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    models = {
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression()
    }

    predictions = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        latest_features = X.iloc[-1].values.reshape(1, -1)
        pred = round(model.predict(latest_features)[0] * 100, 2)
        predictions.append((name, pred))

    return predictions


def get_model_wise_predictions(threshold):
    stocks = fetch_nifty100_symbols()
    model_results = {"GradientBoosting": [], "RandomForest": [], "LinearRegression": []}

    for stock in stocks:
        data = fetch_stock_data(stock)
        if data is not None and len(data) > 20:
            data = prepare_features(data)
            if len(data) > 10:
                preds = train_and_predict_all_models(data)
                for model, pred in preds:
                    if pred >= threshold:
                        model_results[model].append({"Stock": stock, "Predicted Return": pred})

    for model in model_results:
        df = pd.DataFrame(model_results[model])
        if not df.empty and "Predicted Return" in df.columns:
            df.sort_values(by="Predicted Return", ascending=False, inplace=True)
            model_results[model] = df.head(5)
        else:
            model_results[model] = pd.DataFrame()

    return model_results


def main():
    st.set_page_config(page_title="NSE Stock Predictor", layout="wide")
    st.title("📈 NSE Stock Predictor")
    st.markdown("Enter your target gain percentage and view top 5 stock picks by each model.")

    threshold = st.number_input("Minimum Expected Gain (%)", min_value=0.0, value=5.0, step=0.5)

    if st.button("Get Predictions"):
        st.info("Fetching predictions... please wait")
        results = get_model_wise_predictions(threshold)

        if any(not df.empty for df in results.values()):
            for model, df in results.items():
                if not df.empty:
                    st.subheader(f"📌 Top Picks by {model}")
                    st.dataframe(df.reset_index(drop=True))
                else:
                    st.info(f"No suitable picks by {model}.")
        else:
            st.warning("No suitable stocks found this week.")

    st.button("Exit", on_click=st.stop)


if __name__ == "__main__":
    main()
