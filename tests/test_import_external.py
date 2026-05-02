import sys
from src.data.import_external import main as import_external_videos

if __name__ == "__main__":
    # Giả lập việc gõ "--test" trên Terminal
    sys.argv = ["src/data/import_external.py", "--test"]
    
    # Bây giờ gọi hàm main, nó sẽ nhận diện được args.test = True
    import_external_videos()