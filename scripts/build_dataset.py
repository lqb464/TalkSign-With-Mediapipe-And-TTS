import argparse
import sys

from src.data.collect_webcam import main as collect_webcam_data
from src.data.import_external import main as step_import_external
from src.data.raw_to_processed import main as step_raw_to_processed


def main():
    parser = argparse.ArgumentParser(
        description="Run dataset pipeline or collect webcam data."
    )

    parser.add_argument(
        "--collect-webcam",
        action="store_true",
        help="Collect raw data from webcam instead of running the dataset pipeline.",
    )

    parser.add_argument(
        "--source",
        choices=["raw", "external", "all"],
        default="all",
        help="Chọn nguồn dữ liệu: 'raw', 'external', hoặc 'all'."
    )

    args = parser.parse_args()

    if args.collect_webcam:
        collect_webcam_data()
        return

    print("\n" + "=" * 50)
    print(f"BẮT ĐẦU PIPELINE - CHẾ ĐỘ: {args.source.upper()}")
    print("=" * 50)

    original_argv = sys.argv

    if args.source in ["external", "all"]:
        print("\n[STEP 1] Importing EXTERNAL VIDEOS to RAW...")
        sys.argv = [original_argv[0]]
        step_import_external()

    if args.source in ["raw", "external", "all"]:
        print("\n[STEP 2] Converting RAW to PROCESSED (.npz)...")
        sys.argv = [original_argv[0]]
        step_raw_to_processed()

    sys.argv = original_argv

    print("\n" + "=" * 50)
    print("PIPELINE HOÀN TẤT!")
    print("=" * 50)


if __name__ == "__main__":
    main()