import json
import matplotlib.pyplot as plt

# file path (your exact path)
log_path = "/home/iiti/Documents/dlproject/AmpleHate-main/save/ihc/seed_0/lambda_1.5/log.json"

# load log file
with open(log_path, "r") as f:
    logs = json.load(f)

train_loss = logs["train"]["loss"]
train_f1 = logs["train"]["f1"]
thresholds = logs["train"]["threshold"]

valid_loss = logs["valid_loss"]
valid_f1 = logs["valid_f1_score"]

epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12,4))

# Loss curve
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, marker='o', label="Train Loss")
plt.axhline(y=valid_loss, linestyle='--', label="Valid Loss")
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# F1 curve
plt.subplot(1,2,2)
plt.plot(epochs, train_f1, marker='o', label="Train F1")
plt.axhline(y=valid_f1, linestyle='--', label="Valid F1")
plt.title("F1 Score vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()

plt.tight_layout()
plt.show()