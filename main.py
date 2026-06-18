import os
import sys

if __name__ == "__main__":
    print("🚀 Starting CineMatch Movie Recommendation System...")
    print("Running command: streamlit run app.py")
    try:
        os.system("streamlit run app.py")
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
        sys.exit(0)