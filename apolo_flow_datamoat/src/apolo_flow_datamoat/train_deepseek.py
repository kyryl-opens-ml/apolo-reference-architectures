import typer
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported, FastLanguageModel
from typing import Optional

app = typer.Typer()

# Constants and configurations
DEFAULT_MODEL = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an insurance claim that provides context.
Write a response that appropriately analyzes the claim and determines if it's fraudulent.
Before answering, think carefully but concisely about the claim details and create a step-by-step analysis to ensure a logical and accurate fraud assessment.

### Instruction:
You are an expert insurance claims analyst with advanced knowledge in fraud detection. Your task is to analyze insurance claims and identify potential fraud indicators. Please review the following claim.

### Claim:
{}

### Analysis:
{}
"""

def format_dataset(examples, tokenizer):
    """Format the dataset according to the prompt template."""
    prompts = examples["prompt"]
    completions = examples["completion"]
    texts = [
        PROMPT_TEMPLATE.format(prompt, completion) + tokenizer.eos_token
        for prompt, completion in zip(prompts, completions)
    ]
    return {"text": texts}

def setup_model_and_tokenizer(model_name: str, max_seq_length: int = 2048):
    """Initialize the model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Add LORA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer

@app.command()
def train(
    model_name: str = DEFAULT_MODEL,
    repo_name: str = typer.Option(..., help="Your HuggingFace username"),
    output_model_name: str = typer.Option(..., help="Name for the fine-tuned model"),
    dataset_name: str = "sdiazlor/python-reasoning-dataset",
    num_epochs: int = 3,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    learning_rate: float = 2e-4,
):
    """Fine-tune DeepSeek model on a reasoning dataset."""
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Load and prepare dataset
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.map(
        lambda x: format_dataset(x, tokenizer),
        batched=True,
    )
    
    # Configure training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_steps=5,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    
    # Train the model
    trainer.train()
    
    # Save and push to Hub
    full_model_name = f"{repo_name}/{output_model_name}"
    model.save_pretrained(output_model_name)
    tokenizer.save_pretrained(output_model_name)
    model.save_pretrained_merged(output_model_name, tokenizer, save_method="merged_16bit")
    
    # Push to HuggingFace Hub
    model.push_to_hub(full_model_name, safe_serialization=None)
    tokenizer.push_to_hub(full_model_name, safe_serialization=None)
    model.push_to_hub_merged(full_model_name, tokenizer, save_method="merged_16bit")
    model.push_to_hub_gguf(
        f"{full_model_name}_q4_k_m",
        tokenizer,
        quantization_method="q4_k_m"
    )
    
    typer.echo(f"Training completed! Model saved as {full_model_name}")

@app.command()
def inference(
    question: str,
    model_path: str,
    max_new_tokens: int = 2048,
):
    """Run inference using the fine-tuned model."""
    model, tokenizer = setup_model_and_tokenizer(model_path)
    FastLanguageModel.for_inference(model)
    
    inputs = tokenizer(
        [PROMPT_TEMPLATE.format(question, "")],
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    
    response = tokenizer.batch_decode(outputs)
    print(response[0].split("### Analysis:")[1])

if __name__ == "__main__":
    app() 