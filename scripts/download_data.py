import sys

def main():
    try:
        import kagglehub
    except ImportError:
        print("Error: 'kagglehub' is required to download datasets.")
        print("Please install it via: pip install kagglehub")
        return 1

    path = kagglehub.dataset_download("dennisho/blackjack-hands")
    print("Path to dataset files:", path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
