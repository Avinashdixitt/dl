import torch
from transformers import AutoTokenizer
from model import CustomBERT

# Select device
device = torch.device("cpu")   # change to "cuda" if using GPU

# Update path if different
MODEL_PATH = "/home/iiti/Documents/dlproject/AmpleHate-main/save/ihc/seed_0/lambda_0.75/best_model.pth"
MODEL_TYPE = "bert-base-uncased"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)

# Load model
model = CustomBERT(MODEL_TYPE, hidden_dim=768, e=0.5)

checkpoint = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(checkpoint["model"])
threshold = checkpoint["threshold"]

model.to(device)
model.eval()


def predict(sentence):
    encoding = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Required by AmpleHate model architecture
    head_token_idx = torch.zeros((1, 1)).long().to(device)

    with torch.no_grad():
        outputs = model(input_ids, head_token_idx, attention_mask)
        prob = torch.softmax(outputs, dim=1)[0][1].item()

    label = "implicit_hate" if prob >= threshold else "not_hate"

    return label, prob


# Interactive input loop
while True:
    sentence = input("\nEnter sentence (or type exit): ")

    if sentence.lower() == "exit":
        break

    label, prob = predict(sentence)

    print(f"\nPrediction: {label}")
    print(f"Confidence: {prob:.4f}")