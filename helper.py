import pandas as pd
import os
import json
import numpy as np

def clean(df):

    # Forward-fill 'timestamp' and 'activity_type'
    df['timestamp'] = df['timestamp'].ffill()
    df['activity_type'] = df['activity_type'].ffill()

    # Drop rows where 'heart_rate' is missing
    df = df[df['heart_rate'].notnull()].copy()

    # Convert 'timestamp' to datetime if it's not already
    df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'])
    df.loc[:, 'adjusted_timestamp'] = df.apply(resolve_timestamp_16, axis=1)
    # df['adjusted_timestamp'] = df['adjusted_timestamp'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')

    # Drop the old 'timestamp' column if you no longer need it
    df = df.drop(columns=['timestamp'])

    df = df[['adjusted_timestamp', 'activity_type', 'heart_rate']]

    return df


def resolve_timestamp_16(row):
    base_ts = int(row['timestamp'].timestamp())
    ts16 = int(row['timestamp_16'])
    # Adjust for overflow if needed
    base_mod = base_ts % 65536
    offset = ts16 - base_mod
    if offset < 0:
        offset += 65536
    corrected_ts = base_ts + offset
    return pd.to_datetime(corrected_ts, unit='s')



def calculate_rhr(df: pd.DataFrame) -> float:
    """
    Calculates resting heart rate (RHR) as the lowest average heart rate 
    over any continuous 30-minute window in the DataFrame.

    Assumes 'timestamp' and 'heart_rate' columns exist.
    Returns the RHR as a float (bpm).
    """


    df = df.copy()
    df = df.sort_values('adjusted_timestamp')

    # Set timestamp as index for rolling
    df = df.set_index('adjusted_timestamp')

    # Resample to 1-minute intervals to normalize spacing
    df = df.resample('1min').mean(numeric_only=True)

    # Apply rolling 30-minute window, compute average HR
    rolling_mean = df['heart_rate'].rolling('30min').mean()

    # Return the minimum of those rolling averages
    rhr = rolling_mean.min()

    return round(rhr, 1) if rhr is not None else None



def extract_garmin_daily_data(folder_path):
    health_data = []
    stress_data = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)

                        # handle JSON arrays or objects
                        if isinstance(content, list):
                            records = content
                        else:
                            records = [content]

                        for entry in records:
                            # --- Extract General Health Summary ---
                            if "calendarDate" in entry:
                                row = {
                                    "calendarDate": entry.get("calendarDate"),
                                    "totalKilocalories": entry.get("totalKilocalories"),
                                    "activeKilocalories": entry.get("activeKilocalories"),
                                    "bmrKilocalories": entry.get("bmrKilocalories"),
                                    "totalSteps": entry.get("totalSteps"),
                                    "dailyStepGoal": entry.get("dailyStepGoal"),
                                    "totalDistanceMeters": entry.get("totalDistanceMeters"),
                                    "minAvgHeartRate": entry.get("minAvgHeartRate"),
                                    "maxAvgHeartRate": entry.get("maxAvgHeartRate"),
                                    "minHeartRate": entry.get("minHeartRate"),
                                    "maxHeartRate": entry.get("maxHeartRate"),
                                    "restingHeartRate": entry.get("restingHeartRate"),
                                    "currentDayRestingHeartRate": entry.get("currentDayRestingHeartRate"),
                                    "highlyActiveSeconds": entry.get("highlyActiveSeconds"),
                                    "activeSeconds": entry.get("activeSeconds"),
                                    "moderateIntensityMinutes": entry.get("moderateIntensityMinutes"),
                                    "vigorousIntensityMinutes": entry.get("vigorousIntensityMinutes"),
                                    "userIntensityMinutesGoal": entry.get("userIntensityMinutesGoal"),
                                    "userFloorsAscendedGoal": entry.get("userFloorsAscendedGoal"),
                                }
                                health_data.append(row)

                            # --- Extract Stress Summary if present ---
                            if "allDayStress" in entry:
                                stress_entry = entry["allDayStress"]
                                calendar_date = stress_entry.get("calendarDate")
                                for agg in stress_entry.get("aggregatorList", []):
                                    if agg.get("type") == "TOTAL":
                                        stress_data.append({
                                            "calendarDate": calendar_date,
                                            "averageStressLevel": agg.get("averageStressLevel"),
                                            "averageStressLevelIntensity": agg.get("averageStressLevelIntensity"),
                                            "maxStressLevel": agg.get("maxStressLevel")
                                        })

                except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
                    continue

    # Convert to DataFrames
    df_health = pd.DataFrame(health_data)
    df_stress = pd.DataFrame(stress_data)

    # Merge on calendarDate
    if not df_health.empty:
        df_health["calendarDate"] = pd.to_datetime(df_health["calendarDate"], errors='coerce')
    if not df_stress.empty:
        df_stress["calendarDate"] = pd.to_datetime(df_stress["calendarDate"], errors='coerce')

    df = pd.merge(df_health, df_stress, on="calendarDate", how="outer").sort_values("calendarDate").reset_index(drop=True)

    return df


# def rolling_outliers_zscore(series, window=14, threshold=1):
#     rolling_mean = series.rolling(window=window, center=True).mean()
#     rolling_std = series.rolling(window=window, center=True).std()
#     z_scores = (series - rolling_mean) / rolling_std
#     return np.abs(z_scores) > threshold


def rolling_outliers_zscore(series, window=14, threshold=2, debug=False):
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()
    rolling_std = rolling_std.replace(0, np.nan)  # Prevent div by 0

    z_scores = (series - rolling_mean) / rolling_std
    outliers = np.abs(z_scores) > threshold

    if debug:
        print(f"Total outliers found: {outliers.sum()}")
        print(outliers.value_counts(dropna=False))

    return outliers, z_scores