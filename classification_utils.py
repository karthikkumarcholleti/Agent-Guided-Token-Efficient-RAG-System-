from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizerFast.from_pretrained("./distilbert-memory-classifier")
model = DistilBertForSequenceClassification.from_pretrained("./distilbert-memory-classifier")
model.eval()


def predict_use_memory(query):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    mapping = {0: "No", 1: "Yes"}
    return mapping[pred]
