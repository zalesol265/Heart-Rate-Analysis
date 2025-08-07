import os
from garmin_fit_sdk import Decoder, Stream
import pandas as pd
from helper import clean, calculate_rhr

data_dir = "data"
cleaned_dfs = []

for filename in os.listdir(data_dir):
    if filename.endswith("WELLNESS.fit"):
        filepath = os.path.join(data_dir, filename)
        print(f"Processing {filename}...")

        stream = Stream.from_file(filepath)
        decoder = Decoder(stream)
        messages, errors = decoder.read()

        monitoring_list = messages.get('monitoring_mesgs')
        if not monitoring_list:
            print(f"No monitoring_mesgs in {filename}, skipping.")
            continue

        df = pd.DataFrame(monitoring_list)

        cleaned_df = clean(df)
        cleaned_dfs.append(cleaned_df)


if cleaned_dfs:
    final_df = pd.concat(cleaned_dfs, ignore_index=True)
    rhr = calculate_rhr(final_df)
    print(f"Resting Heart Rate (RHR): {rhr} bpm")
    print(max(final_df['heart_rate']))
    # final_df.to_csv("monitoring_data.csv", index=False)
    # print("All files processed and saved to monitoring_data.csv")
else:
    print("No valid files found.")
