import json
from datasets import Dataset

# Step 1: Load JSONL files
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

n = 1  # Replace with your value
jsonl_file1 = load_jsonl(f"annotation_result/checkpoint-{n}/an.jsonl")
jsonl_file2 = load_jsonl(f"annotation_result/checkpoint-{n}/chau.jsonl")

# Step 2: Check lengths
if len(jsonl_file1) != len(jsonl_file2):
    raise ValueError(f"Length mismatch: File 1 has {len(jsonl_file1)} entries, File 2 has {len(jsonl_file2)} entries.")
else:
    print(f"Both JSONL files have the same length: {len(jsonl_file1)} entries.")

# Step 3: Load Parquet dataset
parquet_file = f"checkpoints/train-checkpoint-{n}.parquet"
dataset = Dataset.from_parquet(parquet_file)
dataset = dataset.select(range(len(jsonl_file1)))
print(f"Loaded {len(dataset)} rows from the Parquet dataset.")

# Step 4: Convert Parquet dataset from BIO to span-based format
def bio_to_spans(words, bio_labels):
    text = " ".join(words)
    char_positions = []
    current_pos = 0
    for i, word in enumerate(words):
        char_positions.append(current_pos)
        current_pos += len(word)
        if i < len(words) - 1:  # Add space only if not the last word
            current_pos += 1
    
    spans = []
    start_pos = None
    current_label = None
    
    for i, label in enumerate(bio_labels):
        if label == "0":
            if start_pos is not None:
                end_pos = char_positions[i-1] + len(words[i-1]) if i > 0 else char_positions[i]
                spans.append([start_pos, end_pos, current_label])
                start_pos = None
                current_label = None
        elif label.startswith("B-"):
            if start_pos is not None:
                end_pos = char_positions[i-1] + len(words[i-1]) if i > 0 else char_positions[i]
                spans.append([start_pos, end_pos, current_label])
            start_pos = char_positions[i]
            current_label = label[2:]
        elif label.startswith("I-"):
            if start_pos is None:
                start_pos = char_positions[i]
                current_label = label[2:]
    
    if start_pos is not None:
        end_pos = char_positions[-1] + len(words[-1])
        spans.append([start_pos, end_pos, current_label])
    
    return text, spans

def dataset_to_spans(dataset):
    span_data = []
    for entry in dataset:
        words = entry["words"]
        bio_labels = entry["labels"]
        text, spans = bio_to_spans(words, bio_labels)
        span_data.append({"text": text, "spans": spans})
    return span_data

dataset_spans = dataset_to_spans(dataset)

# Print for verification
print("Dataset entry 1:", dataset[2])
print("Dataset spans (entry 2):", dataset_spans[2]["spans"])
print("JSONL1 spans (entry 2):", jsonl_file1[2]["label"])
print("JSONL2 spans (entry 2):", jsonl_file2[2]["label"])

# Step 5: New scoring logic for comparing spans (exact match on position and label)
def calculate_span_score(spans1, spans2):
    # Special case: if both have no labels, count as a match (score = 100%)
    if len(spans1) == 0 and len(spans2) == 0:
        return 100.0
    
    # Sort spans to ensure consistent comparison
    spans1 = sorted(spans1, key=lambda x: (x[0], x[1]))
    spans2 = sorted(spans2, key=lambda x: (x[0], x[1]))
    
    # Convert spans to tuples for set operations
    spans1_set = set(tuple(span) for span in spans1)
    spans2_set = set(tuple(span) for span in spans2)
    
    # Find matching spans (exact match on start, end, and label)
    matching_spans = spans1_set.intersection(spans2_set)
    
    # Total unique spans (union)
    total_spans = len(spans1_set.union(spans2_set))
    
    # Score = number of exact matches / total unique spans
    return (len(matching_spans) / total_spans) * 100 if total_spans > 0 else 0

# Step 6: Compare dataset with JSONL files
def calculate_exact_match_score(dataset_spans, jsonl_data):
    total_entries = len(dataset_spans)
    total_score = 0
    
    for dataset_entry, jsonl_entry in zip(dataset_spans, jsonl_data):
        dataset_spans_list = dataset_entry["spans"]
        jsonl_spans_list = jsonl_entry["label"]
        
        # Calculate score for this pair
        score = calculate_span_score(dataset_spans_list, jsonl_spans_list)
        total_score += score
    
    # Average the scores across all entries
    return total_score / total_entries if total_entries > 0 else 0

score_dataset_vs_jsonl1 = calculate_exact_match_score(dataset_spans, jsonl_file1)
score_dataset_vs_jsonl2 = calculate_exact_match_score(dataset_spans, jsonl_file2)

# Step 7: Compare JSONL files
def compare_jsonl_files(jsonl1, jsonl2):
    total_entries = len(jsonl1)
    total_score = 0
    
    for entry1, entry2 in zip(jsonl1, jsonl2):
        spans1 = entry1["label"]
        spans2 = entry2["label"]
        
        # Calculate score for this pair
        score = calculate_span_score(spans1, spans2)
        total_score += score
    
    # Average the scores across all entries
    return total_score / total_entries if total_entries > 0 else 0

score_jsonl1_vs_jsonl2 = compare_jsonl_files(jsonl_file1, jsonl_file2)

# Step 8: Return scores
print(f"Score (Dataset vs JSONL1): {score_dataset_vs_jsonl1:.2f}%")
print(f"Score (Dataset vs JSONL2): {score_dataset_vs_jsonl2:.2f}%")
print(f"Score (JSONL1 vs JSONL2): {score_jsonl1_vs_jsonl2:.2f}%")