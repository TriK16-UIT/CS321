import json
import os
from typing import Optional, Union

from datasets import load_dataset, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from loguru import logger
import fire

class NERInferencer:
    def __init__(self,
                 model_path: str = "outputs/vinai/phobert-base/best_model", 
                 dataset_path: str = "Checkpoints/train-checkpoint-1.parquet",
                 output_dir: Optional[str] = None):
        
        self.dataset = Dataset.from_parquet(dataset_path)

        self.pipe, self.original_model = self._load_pipeline(model_path)

        self.output_dir = output_dir or f"results"
        os.makedirs(self.output_dir, exist_ok=True)

    def inference(self):
        all_word_sequence = [" ".join(self.dataset[i]["words"]) for i in range(len(self.dataset))]

        all_predictions = self.pipe(all_word_sequence)

      
        all_pred_tags = [self._merge_entities(predictions) for predictions in all_predictions]

        # Replace '0' with 'O' in all predicted tags
        all_pred_tags = [[tag if tag != '0' else 'O' for tag in tags] for tags in all_pred_tags]

        with open(os.path.join(self.output_dir, f"{os.path.basename(self.original_model)}.text"), "w", encoding="utf-8") as f:
            for words, tags in zip(self.dataset["words"], all_pred_tags):
                if len(words) != len(tags):
                    print(f"Warning: Mismatch in lengths - Words: {len(words)}, Tags: {len(tags)}")
                    continue  # Skip malformed sequences
                for word, tag in zip(words, tags):
                    f.write(f"{word}\t{tag}\n")
                f.write("\n")  # Blank line between sequences

        print("Saved predictions in CoNLL format!")

    def _load_pipeline(self, model_path):
        """Load the BERT-based NER model
        Args:
            model_path: the path to the model
        Return:
            pipe: the pipeline for NER
            original_model: the original model name"""
        logger.info("Loading pipeline")
        model = AutoModelForTokenClassification.from_pretrained(model_path).to("cuda")
        original_model = json.load(open(f"{model_path}/config.json"))["_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(original_model)
        pipe = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=None,
            batch_size=32,
            device="cuda"
        )
        return pipe, original_model
    
    def _merge_entities(self, entities):
        """Merge the entities with the same start or end
        Args:
            entities: the list of entities
        Return:
            merged_entities: the list of merged entities"""
        merged_entities = []
        current_entity = {}

        for entity in entities:
            # the next entity is the continuation of the current entity
            if current_entity and current_entity["end"] == entity["start"]:
                # Merge with the current entity
                current_entity["end"] = entity["end"]
                if entity["score"] > current_entity["score"]:
                    current_entity["entity"] = entity["entity"]
                    current_entity["score"] = entity["score"]
            # the next entity has the same start as the current entity
            elif current_entity and current_entity["start"] == entity["start"]:
                current_entity["start"] = entity["start"]
                if entity["score"] > current_entity["score"]:
                    current_entity["entity"] = entity["entity"]
                    current_entity["score"] = entity["score"]
            # the next entity has the same end as the current entity
            elif current_entity and current_entity["end"] == entity["end"]:
                current_entity["end"] = entity["end"]
                if entity["score"] > current_entity["score"]:
                    current_entity["entity"] = entity["entity"]
                    current_entity["score"] = entity["score"]
            elif entity["word"] == "▁":  # skip the special token
                continue
            else:
                # Add the current entity to the list and start a new one
                if current_entity:
                    merged_entities.append(current_entity["entity"])
                current_entity = entity

        # Add the last entity
        if current_entity:
            merged_entities.append(current_entity["entity"])

        return merged_entities
    
def main(
    model_path: str = "outputs/vinai/phobert-base/best_model", 
    dataset_path: str = "Checkpoints/train-checkpoint-1.parquet",
    output_dir: Optional[str] = None
):
    inferencer = NERInferencer(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir
    )

    inferencer.inference()

if __name__ == "__main__":
    fire.Fire(main)