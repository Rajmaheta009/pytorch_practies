from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import json

# === Load Dataset ===
with open("dataset.json") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list([
    {"input": d["input"], "output": json.dumps(d["output"])} for d in raw_data
])

# === Load Model and Tokenizer ===
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")

# === Tokenization Function ===
def tokenize(batch):
    return tokenizer(
        batch["input"],
        text_target=batch["output"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized = dataset.map(tokenize, batched=True)

# === Training Arguments ===
training_args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    logging_dir="./logs",
    save_strategy="epoch"
)

# === Trainer Setup ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer
)

# === Start Training ===
trainer.train()
