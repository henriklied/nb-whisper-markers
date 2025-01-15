import torch
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

from datasets import load_dataset, Audio, DatasetDict
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import re

# Load model and processor
model_id = "NbAiLab/nb-whisper-medium"
processor = WhisperProcessor.from_pretrained(model_id, language="no", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_id)
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)

# Freeze most of the model
def freeze_model_layers(model, num_unfrozen_layers=2):
    """Freeze most model layers, keeping only the last few decoder layers trainable"""
    
    # Freeze the entire model first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the last n decoder layers
    for i in range(len(model.model.decoder.layers) - num_unfrozen_layers, len(model.model.decoder.layers)):
        for param in model.model.decoder.layers[i].parameters():
            param.requires_grad = True
    
    # Unfreeze the output projection layer
    for param in model.proj_out.parameters():
        param.requires_grad = True
    
    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")

# Apply freezing
#freeze_model_layers(model, num_unfrozen_layers=32)

def unfreeze_all_layers(model):
    """Unfreeze all layers of the model"""
    for param in model.parameters():
        param.requires_grad = True
    
    # Print trainable parameter count
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")

# Apply unfreezing
unfreeze_all_layers(model)

# Load dataset (non-streaming)
dataset = load_dataset("scribe-project/nbtale3", streaming=False)


def clean_text(text):
    # Replace <vowel> with eee
    text = text.replace("<vowel>", "eee")
    
    # Replace <comma> with ,
    text = text.replace("<comma>", ",")
    text = re.sub(r';deadend=\d+', '', text)
    # Remove other special tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    cleaned = text.strip()
    return cleaned if cleaned else "empty"  # Return "empty" instead of an empty string


# Prepare the data preprocessing function
def prepare_dataset(batch):
    # Process audio
    audio = batch["utterance_audio_file"]
    
    # Compute input features
    batch["input_features"] = processor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features[0]
    
    # Encode targets
    cleaned_text = clean_text(batch["raw_text"])
    batch["labels"] = processor.tokenizer(cleaned_text).input_ids
    
    return batch

# Split the dataset
train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'eval': train_test_split['test']
})

# Resample audio and prepare features
dataset = dataset.cast_column("utterance_audio_file", Audio(sampling_rate=16000))
dataset = dataset.map(prepare_dataset, remove_columns=dataset["train"].column_names)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad input_features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch_input = processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch_labels = processor.tokenizer.pad(label_features, return_tensors="pt")
        
        batch = {
            "input_features": batch_input["input_features"],
            "labels": batch_labels["input_ids"],
        }
        return batch

# Create data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Memory-optimized training arguments for 14GB GPU
training_args = Seq2SeqTrainingArguments(
    output_dir="./nb-whisper-markers",
    per_device_train_batch_size=24,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=100,
    max_steps=8000,
    gradient_checkpointing=True,
    bf16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=4,    # Doubled from 2
    predict_with_generate=True,
    generation_max_length=225,
    max_grad_norm=1.0,
    save_steps=500,                  # More frequent saving
    eval_steps=500,                  # More frequent evaluation
    logging_steps=20,                # Much more frequent logging
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    # Memory optimization settings
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    torch_compile=True,
    optim="adamw_torch",
)


# Configure model for memory efficiency
model.config.use_cache = False

from evaluate import load
wer_metric = load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

# Then modify the Trainer creation to include compute_metrics:
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    data_collator=data_collator,
    processing_class=processor.feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback],
)

# Start training
trainer.train()