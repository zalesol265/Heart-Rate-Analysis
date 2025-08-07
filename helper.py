import pandas as pd

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
