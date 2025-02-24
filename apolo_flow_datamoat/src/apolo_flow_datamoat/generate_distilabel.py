import os
import json
import pandas as pd
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import GenerateTextClassificationData
from distilabel.models import OpenAILLM
import random

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


# task = GenerateTextClassificationData(
#     language="English",
#     difficulty="college",
#     clarity="clear",
#     num_generations=1,
#     llm=OpenAILLM(
#                     model="Qwen/Qwen2.5-1.5B-Instruct",
#                     base_url="https://job-9145afaf-cc47-4d57-9b94-4dde8dd9720b.jobs.scottdc.org.apolo.scottdata.ai/v1/",
#                     api_key="any" 
#                 ),
#     input_batch_size=5,
# )
# task.load()
# result = next(
#     task.process([{"task": "Determine the insurance claim as fraud or not fraud"}])
# )

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
                num_generations=5,  # Generate 5 examples per task
                llm=OpenAILLM(
                    model="Qwen/Qwen2.5-1.5B-Instruct",
                    base_url="https://job-9145afaf-cc47-4d57-9b94-4dde8dd9720b.jobs.scottdc.org.apolo.scottdata.ai/v1/",
                    api_key="any" 
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
output_csv = "insurance_fraud_dataset.csv"
output_json = "insurance_fraud_dataset.json"

df.to_csv(output_csv, index=False)
df.to_json(output_json, orient='records', indent=2)

print(f"\nDataset saved to {os.path.abspath(output_csv)}")
print(f"Dataset also saved to {os.path.abspath(output_json)}")

# Display a few examples
print("\nSample examples:")
print(df.sample(3).to_string())