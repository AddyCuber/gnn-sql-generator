#!/usr/bin/env python3
"""
Helper script to add more training examples to sql_training_data.json
"""

import json
import os
from datetime import datetime

def load_training_data():
    """Load existing training data"""
    if os.path.exists('sql_training_data.json'):
        with open('sql_training_data.json', 'r') as f:
            return json.load(f)
    else:
        return {
            "training_data": [],
            "negative_examples": [],
            "schema_info": {
                "tables": [],
                "relationships": []
            },
            "metadata": {
                "total_examples": 0,
                "total_negative_examples": 0,
                "categories": [],
                "complexity_levels": ["simple", "medium", "complex"],
                "created_date": datetime.now().strftime("%Y-%m-%d"),
                "version": "1.0"
            }
        }

def save_training_data(data):
    """Save training data to JSON file"""
    with open('sql_training_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("✅ Training data saved!")

def add_training_example():
    """Add a new training example"""
    data = load_training_data()
    
    print("\n📝 Adding New Training Example")
    print("=" * 40)
    
    question = input("Question: ")
    sql = input("Expected SQL: ")
    category = input("Category (filtering/aggregation/subquery/joins_aggregation/subquery_aggregation): ")
    complexity = input("Complexity (simple/medium/complex): ")
    
    example = {
        "question": question,
        "sql": sql,
        "category": category,
        "complexity": complexity
    }
    
    data["training_data"].append(example)
    data["metadata"]["total_examples"] = len(data["training_data"])
    
    if category not in data["metadata"]["categories"]:
        data["metadata"]["categories"].append(category)
    
    save_training_data(data)
    print(f"✅ Added training example: {question}")

def add_negative_example():
    """Add a new negative example"""
    data = load_training_data()
    
    print("\n❌ Adding New Negative Example")
    print("=" * 40)
    
    question = input("Invalid Question: ")
    sql = input("Invalid SQL (with -- Invalid: comment): ")
    category = input("Category (invalid_entity/invalid_column/security_violation/invalid_data/invalid_filter): ")
    
    example = {
        "question": question,
        "sql": sql,
        "category": category,
        "complexity": "invalid"
    }
    
    data["negative_examples"].append(example)
    data["metadata"]["total_negative_examples"] = len(data["negative_examples"])
    
    save_training_data(data)
    print(f"✅ Added negative example: {question}")

def view_statistics():
    """View training data statistics"""
    data = load_training_data()
    
    print("\n📊 Training Data Statistics")
    print("=" * 40)
    print(f"Total training examples: {data['metadata']['total_examples']}")
    print(f"Total negative examples: {data['metadata']['total_negative_examples']}")
    print(f"Categories: {', '.join(data['metadata']['categories'])}")
    print(f"Complexity levels: {', '.join(data['metadata']['complexity_levels'])}")
    
    # Show examples by category
    print("\n📋 Examples by Category:")
    categories = {}
    for example in data["training_data"]:
        cat = example["category"]
        if cat not in categories:
            categories[cat] = 0
        categories[cat] += 1
    
    for cat, count in categories.items():
        print(f"  {cat}: {count} examples")

def main():
    """Main menu"""
    while True:
        print("\n🎯 SQL Training Data Manager")
        print("=" * 40)
        print("1. Add training example")
        print("2. Add negative example")
        print("3. View statistics")
        print("4. Exit")
        
        choice = input("\nChoose an option (1-4): ")
        
        if choice == "1":
            add_training_example()
        elif choice == "2":
            add_negative_example()
        elif choice == "3":
            view_statistics()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 