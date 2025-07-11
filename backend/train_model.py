# train_model.py - UPDATED FOR MULTI-CRIME SUPPORT

import pandas as pd
import sqlite3
import pickle
import logging
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "documents.db"
MODEL_PATH = "incident_forecaster.pkl"


def train_and_save_model(crime_type: Optional[str] = None):
    """
    Loads historical incident data from the database,
    trains an ARIMA forecasting model,
    and saves the trained model to a file.

    Args:
        crime_type: Optional specific crime type to train on.
                   If None, aggregates all crime types.
    """
    logger.info("Starting model training process...")

    # 1. Load data from the database
    try:
        conn = sqlite3.connect(DB_PATH)

        if crime_type:
            # Train on specific crime type
            logger.info(f"Training model for specific crime type: {crime_type}")
            df = pd.read_sql_query("""
                                   SELECT report_date, total_incidents
                                   FROM incident_time_series
                                   WHERE crime_type = ?
                                   ORDER BY report_date ASC
                                   """, conn, params=[crime_type])
            model_suffix = f"_{crime_type.replace(' ', '_').lower()}"
        else:
            # Aggregate all crime types
            logger.info("Training model on aggregated data from all crime types")
            df = pd.read_sql_query("""
                                   SELECT report_date, SUM(total_incidents) as total_incidents
                                   FROM incident_time_series
                                   GROUP BY report_date
                                   ORDER BY report_date ASC
                                   """, conn)
            model_suffix = "_all_crimes"

        conn.close()

    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        return

    # Check if we have data
    if df.empty:
        logger.warning(f"No data found for crime type: {crime_type or 'any'}")
        return

    logger.info(f"Found {len(df)} data points")
    logger.info(f"Date range: {df['report_date'].min()} to {df['report_date'].max()}")
    logger.info(f"Incident range: {df['total_incidents'].min()} to {df['total_incidents'].max()}")

    # ARIMA needs fewer data points to start, e.g., 6.
    if len(df) < 6:
        logger.warning(f"Not enough data to train ARIMA model. Found {len(df)} data points. Need at least 6.")
        return

    # 2. Prepare data for time-series modeling
    df['report_date'] = pd.to_datetime(df['report_date'])
    df.set_index('report_date', inplace=True)

    # Check for duplicate dates (shouldn't happen now, but good to verify)
    if df.index.duplicated().any():
        logger.warning("Found duplicate dates. Aggregating...")
        df = df.groupby(df.index).sum()

    # Set frequency and interpolate missing values
    df = df.asfreq('MS')  # Month start frequency
    df['total_incidents'] = df['total_incidents'].interpolate()

    logger.info(f"Training ARIMA model on {len(df)} data points from {df.index.min()} to {df.index.max()}")

    # 3. Train an ARIMA model (non-seasonal)
    try:
        model = ARIMA(df['total_incidents'], order=(2, 1, 1))
        results = model.fit()

        logger.info("ARIMA model training completed successfully")
        logger.info(f"AIC: {results.aic:.2f}")

    except Exception as e:
        logger.error(f"Failed to train ARIMA model: {e}")
        logger.info("Trying alternative ARIMA parameters...")

        # Try simpler model if the first one fails
        try:
            model = ARIMA(df['total_incidents'], order=(1, 1, 1))
            results = model.fit()
            logger.info("Alternative ARIMA model training completed successfully")
            logger.info(f"AIC: {results.aic:.2f}")
        except Exception as e2:
            logger.error(f"Alternative ARIMA model also failed: {e2}")
            return

    # 4. Save the trained model to a file
    try:
        model_filename = f"incident_forecaster{model_suffix}.pkl"
        with open(model_filename, 'wb') as pkl_file:
            pickle.dump(results, pkl_file)
        logger.info(f"Model trained successfully and saved to {model_filename}")

        # Also save the default model for backward compatibility
        if not crime_type:
            with open(MODEL_PATH, 'wb') as pkl_file:
                pickle.dump(results, pkl_file)
            logger.info(f"Also saved as default model: {MODEL_PATH}")

    except Exception as e:
        logger.error(f"Failed to save model file: {e}")


def train_all_crime_models():
    """Train individual models for each crime type plus an aggregated model"""
    logger.info("Training models for all available crime types...")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Get all available crime types
        cursor.execute("SELECT DISTINCT crime_type FROM incident_time_series WHERE crime_type IS NOT NULL")
        crime_types = [row[0] for row in cursor.fetchall()]

        conn.close()

        logger.info(f"Found crime types: {crime_types}")

        # Train aggregated model (all crimes combined)
        logger.info("Training aggregated model...")
        train_and_save_model(crime_type=None)

        # Train individual models for each crime type
        for crime_type in crime_types:
            logger.info(f"Training model for: {crime_type}")
            train_and_save_model(crime_type=crime_type)

        logger.info("All models trained successfully!")

    except Exception as e:
        logger.error(f"Failed to train all models: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            train_all_crime_models()
        else:
            crime_type = sys.argv[1]
            train_and_save_model(crime_type=crime_type)
    else:
        # Default: train aggregated model
        train_and_save_model(crime_type=None)