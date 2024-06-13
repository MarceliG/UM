import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertModel, BertTokenizer, get_linear_schedule_with_warmup

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
        print(idx)
        print(self.texts)

        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label),
        }


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

    def model_train(self, model, data_loader, optimizer, scheduler, device):
        self.train()
        for batch in data_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

    def model_evaluate(self, model, data_loader, device):
        self.eval()
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
        return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)


def run_bert(percentage_dataset: float = 100):
    """Run BERT function."""

    """Set up parameters"""
    bert_model_name = "bert-base-uncased"
    num_classes = 2
    max_length = 128
    batch_size = 16
    num_epochs = 4
    learning_rate = 2e-5

    # dataset = load_dataset_from_disc(os.path.join(DATASETS_PREPROCESED_PATH, "raw_review_All_Beauty"))
    # df = dataset.to_pandas()

    data = {
        "text": [
            "I love this movie, it was fantastic!",
            "I hate this movie, it was terrible!",
            "This film was amazing, I enjoyed it a lot.",
            "What a bad movie, I did not like it.",
            "Great plot and excellent acting!",
            "Worst film ever, completely awful.",
            "It was an okay movie, nothing special.",
            "The storyline was very boring and dull.",
            "Loved the movie, it was wonderful!",
            "Terrible film, I disliked it a lot.",
            "Fantastic movie with great acting!",
            "Awful movie, not worth watching.",
            "One of the best movies I've seen.",
            "Really bad film, don't recommend it.",
            "Enjoyed every moment of the movie!",
            "The movie was very disappointing.",
        ],
        "rating": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    }
    df = pd.DataFrame(data)

    texts = df["text"]
    labels = df["rating"]

    # Get % subset of dataset
    subset_size = round(len(df) * percentage_dataset / 100)
    print(f"Size of dataset: {subset_size}")

    # (train_texts, val_texts, train_labels, val_labels) = train_test_split(
    #     texts, labels, train_size=subset_size, random_state=42
    # )
    (
        train_texts,
        val_texts,
        train_labels,
        val_labels,
    ) = train_test_split(
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

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.model_train(model, train_dataloader, optimizer, scheduler, device)
        accuracy, report = model.model_evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)

        save_bert_model(model, "bert_classifier.pth")
