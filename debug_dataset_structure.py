import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from baseline.logiqa_data import load_logiqa_dataset

def inspect_dataset():
    print("Loading dataset...")
    dataset = load_logiqa_dataset()
    
    if len(dataset) > 0:
        print("\n--- Example 0 Structure ---")
        example = dataset[0]
        print(f"Keys: {list(example.keys())}")
        print(f"Example content: {example}")
        
        # Check specific fields we suspected
        print(f"\n'label' value: {example.get('label')}")
        print(f"'answer' value: {example.get('answer')}")
        print(f"'correct_answer' value: {example.get('correct_answer')}")
    else:
        print("Dataset is empty.")

if __name__ == "__main__":
    inspect_dataset()
