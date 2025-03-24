import json
import os
from typing import Optional, Union

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from loguru import logger
import numpy as np
import fire

from slue import get_slue_format, get_ner_scores
from modified_seqeval import classification_report
from sklearn.metrics import accuracy_score


class NERTrainer:
    def __init__(self, 
                 model_name: str = "vinai/phobert-base", 
                 dataset: Union[str, Dataset] = "leduckhai/VietMed-NER",
                 output_dir: Optional[str] = None,
                 resume_from_checkpoint: Optional[str] = None,
                 train_fraction: float = 1.0):
        """
        Initialize NER Trainer with flexible configuration.
        
        Args:
            model_name (str): Pretrained model to use for training
            dataset (str): Name of the dataset to use or Dataset
            output_dir (str, optional): Directory to save outputs. 
                                        Defaults to f"outputs/{model_name}"
            resume_from_checkpoint (str, optional): Path to checkpoint to resume training
        """
        # Load dataset
        if isinstance(dataset, str):
            self.dataset = load_dataset(dataset)
        else:
            self.dataset = dataset

        if train_fraction < 1.0:
            self.dataset['train'] = self._sample_dataset(
                self.dataset['train'], 
                train_fraction
            )
            logger.info(f"Using {train_fraction*100}% of training data: {len(self.dataset['train'])} samples")   
        
        # Prepare label mappings
        self.id2label = self.dataset["train"].features["tags"].feature._int2str
        self.id2label = {int(k): v for k, v in enumerate(self.id2label)}
        self.label2id = self.dataset["train"].features["tags"].feature._str2int
        
        # Set output directory
        self.output_dir = output_dir or f"outputs/{model_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Logging setup
        self.log_file = os.path.join(self.output_dir, f"{model_name}.log")
        self.log_id = logger.add(self.log_file)
        
        # Model, tokenizer, and data preparation
        self.model_name = model_name
        self.model, self.tokenizer, self.data_collator, _ = self._prepare_model(
            model_name, resume_from_checkpoint=resume_from_checkpoint
        )
        
        # Compute metrics function
        self.compute_metrics = self._build_compute_metrics()
        
        # Tokenize dataset
        self.tokenized_dataset = self.dataset.map(
            self._tokenize_and_align_labels, 
            batched=True, 
            fn_kwargs={"tokenizer": self.tokenizer}
        )

    def _sample_dataset(self, dataset: Dataset, fraction: float) -> Dataset:
        """
        Sample a fraction of the dataset.
        
        Args:
            dataset (Dataset): Input dataset
            fraction (float): Fraction of data to keep
        
        Returns:
            Dataset: Sampled dataset
        """
        # Use shuffle and select to get a random subset
        return dataset.shuffle(seed=42).select(
            range(int(len(dataset) * fraction))
        )

    def _prepare_model(self, 
                       model_name: str, 
                       device: str = "cuda", 
                       resume_from_checkpoint: Optional[str] = None):
        """Prepare the model for training."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        data_collator = DataCollatorForTokenClassification(tokenizer)

        # Handle checkpoint resumption
        if resume_from_checkpoint:
            model = AutoModelForTokenClassification.from_pretrained(
                resume_from_checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
            ).to(device)
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                id2label=self.id2label,
                label2id=self.label2id,
            ).to(device)

        return model, tokenizer, data_collator, model_name

    def _tokenize_and_align_labels(self, examples, tokenizer):
        """Tokenize the input and align the labels with the tokens."""
        tokenized_inputs = tokenizer(
            examples["words"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def _build_compute_metrics(self):
        """Build the function to compute the metrics for evaluation."""
        def eval_compute_metrics(p):
            predictions, labels, inputs = p
            predictions = np.argmax(predictions, axis=2)
            inputs = np.where(inputs == -100, 0, inputs)

            original_words = [
                text.split(" ")
                for text in self.tokenizer.batch_decode(inputs, skip_special_tokens=True)
            ]

            true_predictions = [
                [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.id2label[l] for (_, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            flat_true_labels = [label for sublist in true_labels for label in sublist]
            flat_predictions = [pred for sublist in true_predictions for pred in sublist]

            all_gt = [
                get_slue_format(original_words[i], true_labels[i], False)
                for i in range(len(labels))
            ]
            all_pred = [
                get_slue_format(original_words[i], true_predictions[i], False)
                for i in range(len(predictions))
            ]
            all_gt_dummy = [
                get_slue_format(original_words[i], true_labels[i], True)
                for i in range(len(inputs))
            ]
            all_pred_dummy = [
                get_slue_format(original_words[i], true_predictions[i], True)
                for i in range(len(predictions))
            ]

            slue_scores = get_ner_scores(all_gt, all_pred)
            dummy_slue_scores = get_ner_scores(all_gt_dummy, all_pred_dummy)

            results = classification_report(
                true_predictions, true_labels, digits=4, output_dict=True
            )

            accuracy = accuracy_score(flat_true_labels, flat_predictions)

            return {
                "precision": results["macro avg"]["precision"],
                "recall": results["macro avg"]["recall"],
                "f1": results["macro avg"]["f1-score"],
                "accuracy": accuracy,
                "slue_scores": slue_scores,
                "dummy_slue_scores": dummy_slue_scores,
                "results": results,
            }

        return eval_compute_metrics

    def train(self, 
              learning_rate: float = 2e-5,
              batch_size: int = 8,
              num_epochs: int = 5,
              gradient_accumulation_steps: int = 2):
        """
        Train the model with configurable hyperparameters.
        
        Args:
            learning_rate (float): Learning rate for training
            batch_size (int): Batch size for training and evaluation
            num_epochs (int): Number of training epochs
            gradient_accumulation_steps (int): Number of gradient accumulation steps
        """
        logger.info(f"Training {self.model_name}")

        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_total_limit=1,
            logging_steps=20,
            push_to_hub=False,
            report_to="tensorboard",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            include_inputs_for_metrics=True,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        # Train the model
        logger.info("Start training")
        trainer.train()

        # Evaluate on test set
        logger.info("Start evaluation")
        result = trainer.predict(self.tokenized_dataset["test"])

        # Save results
        logger.info("Finish training, saving the results")
        with open(os.path.join(self.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
            json.dump(result.metrics, f)

        # Clean up logger
        logger.remove(self.log_id)

def main(
    model_name: str = "vinai/phobert-base",
    dataset: str = "leduckhai/VietMed-NER",
    output_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    num_epochs: int = 5,
    gradient_accumulation_steps: int = 2,
    train_fraction: float = 1.0
):
    """
    Main function to train NER model with flexible configuration.
    
    Args:
        model_name (str): Pretrained model to use for training
        dataset (str): Name of the dataset to use or Dataset
        output_dir (str, optional): Directory to save outputs
        resume_from_checkpoint (str, optional): Path to checkpoint to resume training
        learning_rate (float): Learning rate for training
        batch_size (int): Batch size for training and evaluation
        num_epochs (int): Number of training epochs
        gradient_accumulation_steps (int): Number of gradient accumulation steps
    """
    trainer = NERTrainer(
        model_name=model_name, 
        dataset=dataset, 
        output_dir=output_dir, 
        resume_from_checkpoint=resume_from_checkpoint,
        train_fraction=train_fraction
    )
    
    trainer.train(
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

if __name__ == "__main__":
    fire.Fire(main)