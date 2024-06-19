import os

import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, Trainer, TrainingArguments, get_linear_schedule_with_warmup

from bert.save_bert import save_bert_model
from configuration import DATASETS_PREPROCESED_PATH
from data_manager import load_dataset_from_disc


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits

        return logits


def model_train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        loss, outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        scheduler.step()


def model_evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(
        actual_labels, predictions, zero_division=0
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


def run_bert(model_type: str, percentage_dataset: float = 100):
    """Run BERT function."""

    """Set up parameters"""
    bert_model_name = "bert-base-uncased"
    num_classes = 2
    max_length = 128
    batch_size = 16
    num_epochs = 4
    learning_rate = 2e-5

    dataset = load_dataset_from_disc(os.path.join(DATASETS_PREPROCESED_PATH, "raw_review_All_Beauty"))
    df = dataset.to_pandas()

    texts = df["text"]
    labels = df["rating"]

    # Get % subset of dataset
    subset_size = round(len(df) * percentage_dataset / 100)
    print(f"Size of dataset: {subset_size}")
    subset_df = df.sample(n=subset_size, random_state=42)

    texts = subset_df["text"]
    labels = subset_df["rating"]

    (train_texts, val_texts, train_labels, val_labels) = train_test_split(
        texts,
        labels,
        random_state=42,
    )

    train_texts = train_texts.reset_index(drop=True)
    val_texts = val_texts.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True)
    val_labels = val_labels.reset_index(drop=True)

    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Use device to learn: ", device)
    model = BERTClassifier(bert_model_name, num_classes).to(device)

    if model_type == "default":
        # Default
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            model_train(model, train_dataloader, optimizer, scheduler, device)
            accuracy, report = model_evaluate(model, val_dataloader, device)
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(report)

            save_bert_model(model, "bert_classifier.pth")

    elif model_type == "best":
        # Best
        training_args = TrainingArguments("test", eval_strategy="steps", eval_steps=50, disable_tqdm=True)

        trainer = Trainer(
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            model_init=lambda: model,
            compute_metrics=compute_metrics,
        )

        best_run = trainer.hyperparameter_search(
            direction="maximize",
            backend="ray",
            n_trials=5,  # number of trials
        )
        print("Best run found:")
        print(best_run)

        save_bert_model(model, "bert_classifier_best.pth")
