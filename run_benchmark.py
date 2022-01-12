import os
import pandas as pd

# from yolo_cv import run_tracking;     method = "yolo_cv"
from yolo_cv_ph import run_tracking;  method = "yolo_cv_ph"
# from yolo_ph_ema import run_tracking; method = "yolo_ph_ema"
# from yolo_ph_ma import run_tracking;  method = "yolo_ph_ma"


data_dir = "data/"
output_dir = "benchmark-results"


if __name__ == "__main__":
    keys = ["id", "camera", "speed", "number_of_shuffles", "cup", "detected", "solved"]
    results = {k: [] for k in keys}

    for filename in os.listdir(f"{data_dir}"):
        # Check if video
        if filename[-4:] != ".mp4":
            continue

        print(f"Processing {filename}")

        # Try to split by underscore and ensure the video belongs to benchmark
        config = filename[:-4].split("_")
        if len(config) != 5:
            continue

        # 'config' contains [id, camera, speed, number_of_shuffles, cup]
        detected, solved = run_tracking(f"{data_dir}/{filename}")

        # Append results
        for i in range(5):
            results[keys[i]].append(config[i])
        results["detected"].append(detected)
        results["solved"].append(solved)


    # Build the dataframe and save it to csv
    df = pd.DataFrame.from_dict(results).set_index("id")
    df.to_csv(f"{output_dir}/{method}.csv")
