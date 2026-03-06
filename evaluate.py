import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import Trainer, BertForSequenceClassification
from datasets import load_from_disk
import io

# 1. Hardware & Path Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r"C:\Users\ayush\PROJECTS\MINOR PERSONAL PROJECT\Fake News Detection using BERT\results\checkpoint-13470"

# 2. Load Data and Model
# Ensure 'processed_news_data/test' exists from your previous data-saving step
test_data = load_from_disk("processed_news_data/test")
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

# 3. Execution Logic: Prediction
trainer = Trainer(model=model)
raw_pred, labels, _ = trainer.predict(test_data)
predictions = np.argmax(raw_pred, axis=-1)

# --- EVALUATION LOGIC ---
report_text = classification_report(labels, predictions, target_names=['True', 'Fake'])
cm = confusion_matrix(labels, predictions)
acc = accuracy_score(labels, predictions)

# Print Report to Terminal
print("\n" + "="*45)
print(f"MODEL ACCURACY: {acc:.2%}")
print("="*45)
print(report_text)

# --- VISUALIZATION LOGIC (CV2) ---

def fig_to_img(fig):
    """Converts a Matplotlib figure to an OpenCV-friendly BGR image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    return cv2.imdecode(img_arr, 1)

# Generate Confusion Matrix via Matplotlib
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
       xticklabels=['True', 'Fake'], yticklabels=['True', 'Fake'],
       title='Confusion Matrix', ylabel='Actual Status', xlabel='Predicted Status')

# Annotate with Counts
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
matrix_img = fig_to_img(fig)
plt.close(fig)

# --- ERROR ANALYSIS CANVAS ---
# Create a black background for the visual report
canvas = np.zeros((750, 950, 3), dtype=np.uint8)
font = cv2.FONT_HERSHEY_SIMPLEX

# Draw Header
cv2.putText(canvas, "BERT EVALUATION SUMMARY", (50, 60), font, 1, (255, 255, 255), 2)
cv2.putText(canvas, f"Global Accuracy: {acc:.2%}", (50, 110), font, 0.7, (0, 255, 0), 2)

# Identify Errors
test_df = test_data.to_pandas()
test_df['predicted'] = predictions
test_df['actual'] = labels
errors = test_df[test_df['predicted'] != test_df['actual']]

# List Sample Misclassifications
cv2.putText(canvas, "SAMPLE MISCLASSIFIED ENTRIES:", (50, 170), font, 0.7, (0, 0, 255), 2)
y_pos = 220
for i, row in errors.head(5).iterrows():
    text_preview = f"ID {i}: {row['text'][:80]}..."
    cv2.putText(canvas, text_preview, (50, y_pos), font, 0.45, (180, 180, 180), 1)
    y_pos += 25
    cv2.putText(canvas, f"Actual: {row['actual']} | Predicted: {row['predicted']}", (70, y_pos), font, 0.45, (120, 255, 120), 1)
    y_pos += 45

# --- FINAL DISPLAY ---
cv2.imshow("Confusion Matrix (Inference Results)", matrix_img)
cv2.imshow("Evaluation Metrics & Errors", canvas)

# Save result
cv2.imwrite("final_confusion_matrix.png", matrix_img)
errors.to_csv("misclassified_samples.csv", index=False)

print("\nPress any key on the OpenCV window to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()