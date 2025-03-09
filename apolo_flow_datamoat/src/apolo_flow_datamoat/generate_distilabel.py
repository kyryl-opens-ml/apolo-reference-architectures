import os
import json
import pandas as pd
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import GenerateTextClassificationData
from distilabel.models import OpenAILLM
import random

def generate_insurance_fraud_dataset(
    output_csv="insurance_fraud_dataset.csv",
    output_json="insurance_fraud_dataset.json",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    base_url="https://job-45868b5e-356c-4d93-a58a-2f6b0202d6c2.jobs.scottdc.org.apolo.scottdata.ai/v1/",
    api_key="any",
    examples_per_task=5
):
    """
    Generate a synthetic dataset for insurance fraud classification using distilabel.
    
    Args:
        output_csv: Path to save the CSV output
        output_json: Path to save the JSON output
        model: The LLM model to use
        base_url: API base URL for the model
        api_key: API key for the model
        examples_per_task: Number of examples to generate per task configuration
        
    Returns:
        pandas.DataFrame: The generated dataset
    """
    # Define the labels for our classification task
    labels_fraud = ["fraud", "not fraud"]

    # Define task templates for variety
    task_templates = [
        "Determine the insurance claim as {}",
        "Classify insurance claim as {}",
        "Identify the insurance claim as {}",
        "Categorize the insurance claim as {}",
        "Label the insurance claim using {}",
        "Annotate the insurance claim based on {}",
        "Determine if an insurance claim is {}",
        "Recognize if the insurance claim is {}"
    ]

    # Create classification tasks with different combinations of labels
    classification_tasks = [
        {"task": action.format(" or ".join(random.sample(labels_fraud, 2)))}
        for action in task_templates for _ in range(5)
    ]

    # Define different difficulties and clarity levels for diverse data
    difficulties = ["college", "high school", "PhD"]
    clarity = ["clear", "understandable with some effort", "ambiguous"]

    # Create and run the pipeline
    with Pipeline("insurance-fraud-generation-pipeline") as pipeline:
        # Load the tasks
        tasks_generator = LoadDataFromDicts(data=classification_tasks)

        # Create generation tasks with different parameters
        generate_data = []
        for difficulty in difficulties:
            for clarity_level in clarity:
                task = GenerateTextClassificationData(
                    language="English",
                    difficulty=difficulty,
                    clarity=clarity_level,
                    num_generations=examples_per_task,
                    llm=OpenAILLM(
                        model=model,
                        base_url=base_url,
                        api_key=api_key 
                    ),
                    input_batch_size=2,
                )
                generate_data.append(task)

        # Connect the tasks to the pipeline
        for task in generate_data:
            tasks_generator.connect(task)

        # Run the pipeline
        print("Running the pipeline to generate insurance fraud classification data...")
        distiset = pipeline.run()

    # Extract and save the generated data
    all_examples = []
    for dataset_name in distiset:
        for entry in distiset[dataset_name]["train"]:
            if entry.get("label") in labels_fraud:
                all_examples.append({
                    "text": entry.get("input_text", ""),
                    "label": entry.get("label", ""),
                    "misleading_label": entry.get("misleading_label", "")
                })

    # Convert to DataFrame and save
    df = pd.DataFrame(all_examples)

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total examples: {len(df)}")
    print(f"Fraud examples: {len(df[df['label'] == 'fraud'])}")
    print(f"Not fraud examples: {len(df[df['label'] == 'not fraud'])}")

    # Save to CSV and JSON
    df.to_csv(output_csv, index=False)
    df.to_json(output_json, orient='records', indent=2)

    print(f"\nDataset saved to {os.path.abspath(output_csv)}")
    print(f"Dataset also saved to {os.path.abspath(output_json)}")

    # Display a few examples
    print("\nSample examples:")
    print(df.sample(min(3, len(df))).to_string())
    
    return df

# Example usage
if __name__ == "__main__":
    generate_insurance_fraud_dataset()