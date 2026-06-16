from experiments.test_real_inference import test_comprehensive_metrics
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    try:
        test_comprehensive_metrics()
    except Exception as e:
        print(f"Error: {e}")
