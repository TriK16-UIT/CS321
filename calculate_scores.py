import json
from datasets import Dataset
from seqeval.metrics import f1_score

# ---------- Load dữ liệu JSONL ----------
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]
    
def convert_zero_to_o(label_list):
    return [[tag.replace("0", "O") for tag in sublist] for sublist in label_list]
    
def spans_to_iob(jsonl_data):
    all_iob_labels = []
    
    for entry in jsonl_data:
        text = entry["text"]
        labels = entry["label"]
        
        # Split text into tokens (space-separated)
        tokens = text.split()
        
        # Initialize IOB tags with "O" for all tokens
        iob_labels = ["O"] * len(tokens)
        
        # Calculate character positions for each token
        token_spans = []
        current_pos = 0
        for token in tokens:
            token_start = current_pos
            token_end = current_pos + len(token)
            token_spans.append((token_start, token_end))
            current_pos = token_end + 1  # +1 for the space
        
        # Process each labeled span
        for start, end, entity_type in labels:
            for i, (token_start, token_end) in enumerate(token_spans):
                if token_start >= start and token_end <= end:
                    if token_start == start:
                        iob_labels[i] = f"B-{entity_type}"
                    else:
                        iob_labels[i] = f"I-{entity_type}"
                elif token_start < start and token_end > start and token_end <= end:
                    iob_labels[i] = f"B-{entity_type}"
                elif token_start >= start and token_end > end and token_start < end:
                    iob_labels[i] = f"I-{entity_type}"
        
        # Append the IOB labels for this entry
        all_iob_labels.append(iob_labels)
    
    return all_iob_labels

if __name__ == "__main__":
    # Đường dẫn dữ liệu
    day_num = 11
    jsonl_quan = f"day12-2/quan.jsonl"
    jsonl_chau = f"day12-2/chau.jsonl" 
    jsonl_an = f"day12-2/an.jsonl"
    parquet_file = f"train-checkpoint-{8}.parquet"

    # Load dữ liệu
    data_quan = sorted(load_jsonl(jsonl_quan), key=lambda x: x["id"])
    data_chau = sorted(load_jsonl(jsonl_chau), key=lambda x: x["id"])
    data_an = sorted(load_jsonl(jsonl_an), key=lambda x: x["id"])

    if len(data_quan) != len(data_an):
        raise ValueError("Annotators JSONL không đồng bộ về số lượng dòng.")

    dataset = Dataset.from_parquet(parquet_file)
    dataset = dataset.select(range(len(data_quan)))  # Cắt theo số dòng annotator
    print(f"Loaded {len(dataset)} rows from the Parquet dataset.")
    print("\n")
    
    preprocessed_quan = spans_to_iob(data_quan)
    preprocessed_chau = spans_to_iob(data_chau)
    preprocessed_an = spans_to_iob(data_an)
    label = convert_zero_to_o(dataset["labels"])

    n = 1
    print("Text: ", data_quan[n - 1]["text"])
    print("Quan answer: ", preprocessed_quan[n - 1])
    print("Chau answer: ", preprocessed_chau[n - 1])
    print("An answer: ", preprocessed_an[n - 1])
    print("Label answer: ", label[n - 1])
    print("\n")
      
    print(" Annotator QUAN vs Dataset:")
    print(f1_score(label, preprocessed_quan))

    print("\n Annotator CHAU vs Dataset:")
    print(f1_score(label, preprocessed_chau))
    
    print("\n Annotator An vs Dataset:")
    print(f1_score(label, preprocessed_an))

    print("\n IAA (QUAN vs CHAU):")
    print(f1_score(preprocessed_quan, preprocessed_chau))
    
    print("\n IAA (QUAN vs AN):")
    print(f1_score(preprocessed_quan, preprocessed_an))
    
    print("\n IAA (AN vs CHAU):")
    print(f1_score(preprocessed_an, preprocessed_chau))


    
    