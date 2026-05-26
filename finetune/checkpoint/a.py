import json
import matplotlib.pyplot as plt

# Load json
with open("train_llama.json", "r") as f:
    data = json.load(f)

# Lấy log history
logs = data["log_history"]

# Lấy step, loss, accuracy
steps = [x["step"] for x in logs if "loss" in x]
losses = [x["loss"] for x in logs if "loss" in x]
accs = [x["mean_token_accuracy"] for x in logs if "mean_token_accuracy" in x]

# ===== Loss Plot =====
plt.figure(figsize=(8,5))
plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()

plt.show()


# ===== Accuracy Plot =====
plt.figure(figsize=(8,5))
plt.plot(steps, accs)
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.grid()

plt.show()