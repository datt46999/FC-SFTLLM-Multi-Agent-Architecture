import json
import multiprocessing
from typing import Dict, Any, Optional

from datasets import load_dataset, Dataset

from transformers import AutoTokenizer


# Dataset prepare
def process_xlam_sample(row: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """
    Create format xlam dataset into training process
    format create:
    <user>[user query]</user>
    <tools>[tools definitions]</tools>
    <calls>[expected functions calls]</calls><EOS_TOKEN> 
    """

    formatted_query = f"<user>{row['query']}</user>\n\n"

    # Parse and format available tools
    try:
        parsed_tools = json.loads(row["tools"])
        tools_text = '\n'.join(str(tool) for tool in parsed_tools)
    except json.JSONDecodeError:
        tools_text = str(row["tools"])  # Fallback to raw string
    
    formatted_tools = f"<tools>{tools_text}</tools>\n\n"

    # Parse and format expected function calls
    try:
        parsed_answers = json.loads(row["answers"])
        answers_text = '\n'.join(str(answer) for answer in parsed_answers)
    except json.JSONDecodeError:
        answers_text = str(row["answers"])  # Fallback to raw string

    formatted_answers = f"<calls>{answers_text}</calls>"

    # Combine all parts with EOS token
    complete_text = formatted_query + formatted_tools + formatted_answers + tokenizer.eos_token

    # Update row with processed data
    row["query"] = formatted_query
    row["tools"] = formatted_tools
    row["answers"] = formatted_answers
    row["text"] = complete_text

    return row

def load_and_process_dataset(tokenizer: AutoTokenizer, sample_size: Optional[int] = None )->Dataset:
    """
    Load and process the complete xLam dataset for the function calling training

    args:
    tokenizer: configured dataset ready for training
    sample_size: Optional number of sample to use (None or full dataset)
    return:
    Dataset: Proccess dataset ready for training

    """

    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    
    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.select(range(sample_size))
        print(f"Using sample size: {sample_size:,} samples")
    
    print("Processing dataset sample into training format....")

    def process_batch(batch):
        processed_batch = []
        for i in range(len(batch["query"])):
            row = {
                'query': batch["query"][i],
                'tools': batch["tools"][i],
                'answers': batch["answers"][i]
            }
            process_row = process_xlam_sample(row, tokenizer)
            processed_batch.append(process_row)
        return {
            "text": [item["text"] for item in processed_batch],
            "query": [item["query"] for item in processed_batch],
            "tools": [item["tools"] for item in processed_batch],
            "answers": [item["answers"] for item in processed_batch],
        }

    processed_dataset = dataset.map(
        process_batch,
        batch_size = 1000,
        batched = True,
        num_proc = min(4, multiprocessing.cpu_count()),#use multiple cores
        desc = "Processing xLAM samples"
    )
    
    print(" Dataset processing complete!")
    print(f"Final dataset size: {len(processed_dataset):,} samples")
    print(f" Average text length: {sum(len(text) for text in processed_dataset['text']) / len(processed_dataset):,.0f} characters")
    
    return processed_dataset


def preview_dataset_sample(dataset: Dataset, index: int = 0) -> None:
    """
    Display a formatted preview of a dataset sample for inspection.
    
    Args:
        dataset: The processed dataset
        index: Index of the sample to preview (default: 0)
    """
    if index >= len(dataset):
        print(f"❌ Index {index} is out of range. Dataset has {len(dataset)} samples.")
        return
    
    sample = dataset[index]
    
    print(f" Dataset Sample Preview (Index: {index})")
    print("=" * 80)
    
    print(f"\n🔍 Raw Components:")
    print(f"Query: {sample['query'][:200]}{'...' if len(sample['query']) > 200 else ''}")
    print(f"Tools: {sample['tools'][:200]}{'...' if len(sample['tools']) > 200 else ''}")
    print(f"Answers: {sample['answers'][:200]}{'...' if len(sample['answers']) > 200 else ''}")
    
    print(f"\nComplete Training Text:")
    print("-" * 40)
    print(sample['text'])
    print("-" * 40)
    
    print(f"\nSample Statistics:")
    print(f"   • Text length: {len(sample['text']):,} characters")
    print(f"   • Estimated tokens: ~{len(sample['text']) // 4:,} tokens")
    print("\nPreview complete!")