import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the log file into a DataFrame
log_file = "/Users/soumyashaw/Desktop/ACMLogFiles/Run1_metrics.txt"  # Replace with your actual log file path
metrics = pd.read_csv(log_file)

# Step 2: Compute Additional Metrics
metrics['TNR_Train'] = metrics['TN Train'] / (metrics['TN Train'] + metrics['FP Train'])
metrics['FNR_Train'] = metrics['FN Train'] / (metrics['FN Train'] + metrics['TP Train'])
metrics['TPR_Train'] = metrics['TP Train'] / (metrics['TP Train'] + metrics['FN Train'])
metrics['FPR_Train'] = metrics['FP Train'] / (metrics['FP Train'] + metrics['TN Train'])

metrics['TNR_Valid'] = metrics['TN Valid'] / (metrics['TN Valid'] + metrics['FP Valid'])
metrics['FNR_Valid'] = metrics['FN Valid'] / (metrics['FN Valid'] + metrics['TP Valid'])
metrics['TPR_Valid'] = metrics['TP Valid'] / (metrics['TP Valid'] + metrics['FN Valid'])
metrics['FPR_Valid'] = metrics['FP Valid'] / (metrics['FP Valid'] + metrics['TN Valid'])

# Step 3: Plot Metrics vs Epoch
epochs = metrics['Epoch']

fig, axes = plt.subplots(3, 2, figsize=(14, 18))
fig.suptitle("Metrics vs Epochs", fontsize=16)

# Define a padding value for titles
title_padding = 20

# Subplot 1: TNR
axes[0, 0].plot(epochs, metrics['TNR_Train'], label="Train TNR", marker="o")
axes[0, 0].plot(epochs, metrics['TNR_Valid'], label="Validation TNR", marker="o")
axes[0, 0].set_title("True Negative Rate (TNR)", pad=title_padding)
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("TNR")
axes[0, 0].set_ylim(-0.1, 1.1)  # Set y-axis range
axes[0, 0].legend()

# Subplot 2: FNR
axes[0, 1].plot(epochs, metrics['FNR_Train'], label="Train FNR", marker="o")
axes[0, 1].plot(epochs, metrics['FNR_Valid'], label="Validation FNR", marker="o")
axes[0, 1].set_title("False Negative Rate (FNR)", pad=title_padding)
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("FNR")
axes[0, 1].set_ylim(-0.1, 1.1)  # Set y-axis range
axes[0, 1].legend()

# Subplot 3: TPR
axes[1, 0].plot(epochs, metrics['TPR_Train'], label="Train TPR", marker="o")
axes[1, 0].plot(epochs, metrics['TPR_Valid'], label="Validation TPR", marker="o")
axes[1, 0].set_title("True Positive Rate (TPR)", pad=title_padding)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("TPR")
axes[1, 0].set_ylim(-0.1, 1.1)  # Set y-axis range
axes[1, 0].legend()

# Subplot 4: FPR
axes[1, 1].plot(epochs, metrics['FPR_Train'], label="Train FPR", marker="o")
axes[1, 1].plot(epochs, metrics['FPR_Valid'], label="Validation FPR", marker="o")
axes[1, 1].set_title("False Positive Rate (FPR)", pad=title_padding)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("FPR")
axes[1, 1].set_ylim(-0.1, 1.1)  # Set y-axis range
axes[1, 1].legend()

# Subplot 5: Accuracy
axes[2, 0].plot(epochs, metrics['Train Accuracy'], label="Train Accuracy", marker="o")
axes[2, 0].plot(epochs, metrics['Valid Accuracy'], label="Validation Accuracy", marker="o")
axes[2, 0].set_title("Accuracy", pad=title_padding)
axes[2, 0].set_xlabel("Epoch")
axes[2, 0].set_ylabel("Accuracy")
axes[2, 0].legend()

# Subplot 6: AUC
axes[2, 1].plot(epochs, metrics['AUC Train'], label="Train AUC", marker="o")
axes[2, 1].plot(epochs, metrics['AUC Valid'], label="Validation AUC", marker="o")
axes[2, 1].set_title("Area Under Curve (AUC)", pad=title_padding)
axes[2, 1].set_xlabel("Epoch")
axes[2, 1].set_ylabel("AUC")
axes[2, 1].set_ylim(-0.1, 1.1)  # Set y-axis range
axes[2, 1].legend()

# Adjust layout and show the plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("/Users/soumyashaw/Desktop/ACMLogFiles/Run_.png", dpi=300, bbox_inches="tight")
plt.show()