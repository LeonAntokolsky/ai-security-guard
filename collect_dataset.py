from src.data_collector import PoseDataCollector

if __name__ == "__main__":
    collector = PoseDataCollector()

    # Path where the combined big dataset will be saved
    csv_dataset_path = "data/pose_dataset.csv"

    # 1. Process all .avi files from the fights folder
    collector.collect_from_directory("data/fights", csv_dataset_path, label_name="fight")

    # 2. Process all .mpg files from the noFights folder
    collector.collect_from_directory("data/noFights", csv_dataset_path, label_name="normal")

    print("\n[INFO] Global Dataset collection complete! Check your new data/pose_dataset.csv")