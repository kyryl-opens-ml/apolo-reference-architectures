import os
import pandas as pd
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import GenerateTextClassificationData
from distilabel.models import OpenAILLM
import random
from datasets import Dataset

def generate_insurance_fraud_dataset(
    output_hf_dataset="insurance_fraud_hf_dataset",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    base_url="https://job-45868b5e-356c-4d93-a58a-2f6b0202d6c2.jobs.scottdc.org.apolo.scottdata.ai/v1/",
    api_key="any",
    number_of_samples=100
):
    """
    Generate a synthetic dataset for insurance fraud classification using distilabel.
    
    Args:
        output_hf_dataset: Path to save the HuggingFace dataset
        model: The LLM model to use
        base_url: API base URL for the model
        api_key: API key for the model
        number_of_samples: Target number of total samples to generate
        
    Returns:
        datasets.Dataset: The generated HuggingFace dataset
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

    # Define different difficulties and clarity levels for diverse data
    difficulties = ["college", "high school", "PhD"]
    clarity = ["clear", "understandable with some effort", "ambiguous"]
    
    # Calculate how many examples per task we need
    # We'll use all task templates, all difficulties, and all clarity levels
    total_configs = len(task_templates) * len(difficulties) * len(clarity)
    examples_per_task = max(1, number_of_samples // total_configs)
    
    # Create classification tasks with different combinations of labels
    classification_tasks = [
        {"task": action.format(" or ".join(random.sample(labels_fraud, 2)))}
        for action in task_templates
    ]

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
        print(f"Target number of samples: {number_of_samples}")
        print(f"Generating approximately {examples_per_task} examples per task configuration")
        distiset = pipeline.run()

    # Extract the generated data and format for train_deepseek.py
    formatted_data = []
    for dataset_name in distiset:
        for entry in distiset[dataset_name]["train"]:
            if entry.get("label") in labels_fraud:
                # The claim text becomes the prompt
                prompt = entry.get("input_text", "")
                
                # Create a completion that analyzes the claim and determines if it's fraudulent
                if entry.get("label") == "fraud":
                    completion = f"After analyzing the claim details, I've identified several fraud indicators. This claim appears to be fraudulent for the following reasons:\n\n1. The claim contains suspicious elements typical of insurance fraud.\n2. There are inconsistencies in the reported information.\n3. The pattern matches known fraudulent claim characteristics.\n\nConclusion: This claim is likely FRAUDULENT and requires further investigation."
                else:  # not fraud
                    completion = f"After carefully reviewing the claim details, I don't see any significant fraud indicators. The claim appears to be legitimate based on:\n\n1. The information provided is consistent and reasonable.\n2. There are no suspicious patterns or red flags in the claim.\n3. The claim follows expected patterns for legitimate insurance claims.\n\nConclusion: This claim appears to be LEGITIMATE and can proceed through normal processing."
                
                formatted_data.append({
                    "prompt": prompt,
                    "completion": completion
                })

    # Create DataFrame for statistics and sample display
    df = pd.DataFrame(formatted_data)
    
    # Count fraud vs not fraud examples
    fraud_count = 0
    not_fraud_count = 0
    for entry in distiset:
        for item in distiset[entry]["train"]:
            if item.get("label") == "fraud":
                fraud_count += 1
            elif item.get("label") == "not fraud":
                not_fraud_count += 1

    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total examples: {len(df)}")
    print(f"Fraud examples: {fraud_count}")
    print(f"Not fraud examples: {not_fraud_count}")

    # Create and save as HuggingFace dataset
    hf_dataset = Dataset.from_pandas(df)
    os.makedirs(output_hf_dataset, exist_ok=True)
    hf_dataset.save_to_disk(output_hf_dataset)
    print(f"HuggingFace dataset saved to {os.path.abspath(output_hf_dataset)}")

    # Display a few examples
    print("\nSample examples:")
    sample_df = df.sample(min(3, len(df)))
    for _, row in sample_df.iterrows():
        print(f"\nPrompt: {row['prompt'][:100]}...")
        print(f"Completion: {row['completion'][:100]}...")
    
    return hf_dataset

if __name__ == "__main__":
    import typer
    
    app = typer.Typer()
    
    @app.command()
    def generate(
        output_hf_dataset: str = "insurance_fraud_hf_dataset",
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        base_url: str = "https://job-45868b5e-356c-4d93-a58a-2f6b0202d6c2.jobs.scottdc.org.apolo.scottdata.ai/v1/",
        api_key: str = "any",
        number_of_samples: int = 100
    ):
        """Generate a synthetic dataset for insurance fraud classification."""
        generate_insurance_fraud_dataset(
            output_hf_dataset=output_hf_dataset,
            model=model,
            base_url=base_url,
            api_key=api_key,
            number_of_samples=number_of_samples
        )
    
    app()