import pandas as pd
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your dataset and drop rows with missing responses
df = pd.read_csv("./data/mrbench_v3_devset_train_data.csv").dropna(subset=["response"])

# Convert string labels to integer indices
# Create mapping dictionaries for encoding and later decoding
label_list = sorted(df["label"].unique())
label_to_id = {label: idx for idx, label in enumerate(label_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}

# Replace string labels with integer labels in the DataFrame
df["label"] = df["label"].map(label_to_id)

# Convert DataFrame columns to lists
conversations = df["conversation_history"].tolist()
responses = df["response"].tolist()
labels = df["label"].tolist()  # now integer labels: 0, 1, 2, ..., 8

# Prepare and split the dataset (stratify by the integer label)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    [f"Conversation: {c}\n\nResponse: {r}" for c, r in zip(conversations, responses)],
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=42,
)

# Now make Hugging Face Dataset objects manually
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

# Initialize tokenizer & model
model_name = "allenai/longformer-base-4096"
tokenizer = LongformerTokenizer.from_pretrained(model_name)
model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=len(label_list))
model.to(device)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=4096
    )

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # evaluates at the end of each epoch
    save_strategy="epoch",          # saves checkpoint at each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
predictions = trainer.predict(tokenized_test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

# Generate detailed classification report (using integer labels)
print("Detailed Classification Report (Integer Labels):")
print(classification_report(true_labels, preds, digits=4))

# Generate classification report with target names (map back to original string labels)
target_names = [id_to_label[i] for i in range(len(label_list))]
print("Detailed Classification Report (String Labels):")
print(classification_report(true_labels, preds, target_names=target_names, digits=4))

# --- Save the Fine-tuned Model ---
# Before saving, update the model configuration with label mappings so that id2label and label2id
# are stored in the saved config.json. This is necessary because categorical labels are not handled by default.
model.config.id2label = id_to_label
model.config.label2id = label_to_id

# Save the model and tokenizer; the config will be saved automatically
save_directory = "./saved_model"
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")
