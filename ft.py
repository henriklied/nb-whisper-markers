import torch
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
    EvalPrediction,
)

from datasets import load_dataset, Audio, DatasetDict, concatenate_datasets
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import re
import soundfile as sf
from evaluate import load

import numpy as np

# Load model and processor
model_id = "NbAiLab/nb-whisper-medium"
processor = WhisperProcessor.from_pretrained(model_id, language="no", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(model_id)
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)

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

# Load datasets
dataset1 = load_dataset("scribe-project/nbtale3", streaming=False)
dataset2 = load_dataset("NbAiLab/annotated_distil_raw_ncc_speech_v7_compact1_large_v5", split="test_norwegian_fleurs", streaming=False)

def clean_text_nbtale3(text, sentence_lengths=None, length_threshold=None):
    # Replace <vowel> with eee
    text = text.replace("<vowel>", "eee")
    
    # Replace <comma> with ,
    text = text.replace(" <comma>", ",")
    
    # Remove other special tags, but keep eee and existing punctuation
    text = re.sub(r'<(?!eee)[^>]+>', '', text)
    
    # Ensure eee is surrounded by spaces
    text = re.sub(r'(?<!\s)eee', ' eee', text)
    text = re.sub(r'eee(?!\s)', 'eee ', text)
    
    # Remove any leading/trailing whitespace
    cleaned = text.strip()
    
    # Add a period if the sentence is among the 60% longest
    if sentence_lengths is not None and length_threshold is not None:
        if len(cleaned) >= length_threshold and not cleaned.endswith('.'):
            cleaned += '.'
    
    return cleaned if cleaned else "empty"

def calculate_sentence_stats(dataset):
    sentence_lengths = [len(clean_text_nbtale3(text)) for text in dataset['raw_text']]
    length_threshold = np.percentile(sentence_lengths, 40)  # 60% longest sentences
    return sentence_lengths, length_threshold

# Calculate sentence stats for nbtale3 dataset
sentence_lengths, length_threshold = calculate_sentence_stats(dataset1['train'])


def clean_text_ncc(text):
    # Keep only letters, numbers, spaces, and punctuation
    cleaned = re.sub(r'[^a-zA-Z0-9æøåÆØÅ:,.!?()\s]', '', text)
    return cleaned.strip()


def prepare_dataset(batch, dataset_name):
    audio = batch["utterance_audio_file" if dataset_name == "nbtale3" else "audio"]
    
    inputs = processor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    )
    batch["input_features"] = inputs.input_features[0]
    
    if dataset_name == "nbtale3":
        cleaned_text = clean_text_nbtale3(batch["raw_text"], sentence_lengths, length_threshold)
    else:
        cleaned_text = clean_text_ncc(batch["text"])
    
    batch["labels"] = processor.tokenizer(cleaned_text).input_ids
    
    return batch

# Prepare datasets (use the modified prepare_dataset function)
dataset1 = dataset1.map(lambda x: prepare_dataset(x, "nbtale3"), remove_columns=dataset1["train"].column_names)
dataset2 = dataset2.map(lambda x: prepare_dataset(x, "ncc"), remove_columns=dataset2.column_names)

# Merge datasets
merged_train = concatenate_datasets([dataset1["train"], dataset2])
merged_eval = concatenate_datasets([
    dataset1["train"].select(range(len(dataset1["train"]) // 10)),
    dataset2.select(range(len(dataset2) // 10))
])

dataset = DatasetDict({
    'train': merged_train,
    'eval': merged_eval
})

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch_input = processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch_labels = processor.tokenizer.pad(label_features, return_tensors="pt")
        
        batch = {
            "input_features": batch_input["input_features"],
            "labels": batch_labels["input_ids"],
        }
        return batch

# Create data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./nb-whisper-eee",
    per_device_train_batch_size=24,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=10,
    max_steps=100,
    gradient_checkpointing=False,
    bf16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=12,
    predict_with_generate=True,
    generation_max_length=225,
    max_grad_norm=1.0,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="punct_eee_acc",
    greater_is_better=True,
    push_to_hub=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    torch_compile=True,
    optim="adamw_torch",
)

# Configure model for memory efficiency
model.config.use_cache = True

wer_metric = load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute standard WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    # Compute punctuation and eee accuracy
    punct_eee_acc = compute_punct_eee_accuracy(pred_str, label_str)
    
    return {"wer": wer, "punct_eee_acc": punct_eee_acc}

def compute_punct_eee_accuracy(predictions, references):
    punct_eee_chars = set(",eee")
    total_chars = 0
    correct_chars = 0
    
    for pred, ref in zip(predictions, references):
        pred_chars = [c for c in pred if c in punct_eee_chars]
        ref_chars = [c for c in ref if c in punct_eee_chars]
        
        total_chars += len(ref_chars)
        correct_chars += sum(p == r for p, r in zip(pred_chars, ref_chars))
    
    return correct_chars / total_chars if total_chars > 0 else 1.0

class TranscriptionCallback(TrainerCallback):
    def __init__(self, audio_path, processor, device="cuda"):
        self.audio_path = audio_path
        self.processor = processor
        self.device = device
        
        audio_input, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            import librosa
            audio_input = librosa.resample(
                audio_input, 
                orig_sr=sample_rate, 
                target_sr=16000
            )
        inputs = processor(
            audio_input, 
            sampling_rate=16000,
            return_tensors="pt"
        )
        self.input_features = inputs.input_features.to(device)

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % 10 == 0:
            model.eval()
            with torch.no_grad():
                predicted_ids = model.generate(self.input_features)
                transcription = self.processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
                print(f"\nStep {state.global_step} transcription: {transcription}\n")
            model.train()

# Modify trainer creation to include the new callback
transcription_callback = TranscriptionCallback(
    "/root/13716864.wav",
    processor
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback, transcription_callback],
)

# Start training
trainer.train()
