# train_model.py - UPDATED TO USE A SIMPLER ARIMA MODEL

import pandas as pd
import sqlite3
import pickle
import logging
# MODIFIED: Import ARIMA instead of SARIMAX
from statsmodels.tsa.arima.model import ARIMA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "documents.db"
MODEL_PATH = "incident_forecaster.pkl"


def train_and_save_model():
    """
    Loads historical incident data from the database,
    trains a non-seasonal forecasting model (ARIMA),
    and saves the trained model to a file.
    """
    logger.info("Starting model training process...")

    # 1. Load data from the database
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT report_date, total_incidents FROM incident_time_series ORDER BY report_date ASC",
                               conn)
        conn.close()
    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        return

    # MODIFIED: ARIMA needs fewer data points to start, e.g., 6.
    if df.empty or len(df) < 6:
        logger.warning(f"Not enough data to train ARIMA model. Found {len(df)} data points. Need at least 6.")
        return

    # 2. Prepare data for time-series modeling
    df['report_date'] = pd.to_datetime(df['report_date'])
    df.set_index('report_date', inplace=True)
    df = df.asfreq('MS')
    df['total_incidents'] = df['total_incidents'].interpolate()

    logger.info(f"Training ARIMA model on {len(df)} data points from {df.index.min()} to {df.index.max()}")

    # 3. Train an ARIMA model (non-seasonal)
    # The (p,d,q) order is a common starting point.
    try:
        model = ARIMA(df['total_incidents'], order=(2, 1, 1))
        results = model.fit()
    except Exception as e:
        logger.error(f"Failed to train ARIMA model: {e}")
        return

    # 4. Save the trained model to a file
    try:
        with open(MODEL_PATH, 'wb') as pkl_file:
            pickle.dump(results, pkl_file)
        logger.info(f"Model trained successfully and saved to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to save model file: {e}")


if __name__ == "__main__":
    train_and_save_model()