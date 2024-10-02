import matplotlib.pyplot as plt
import numpy as np
import sys

#file_path = sys.argv[1]
file_path = "/Users/soumyashaw/Downloads/score-ACM-OutofDist_packetLoss.txt"
#file_path = "/Users/soumyashaw/Downloads/score-ACM-ASVspoof2021_BG2_10dB.txt"

y_pred_proba = []
y_test = []

with open(file_path, "r") as file:
    for line in file:
        #score = (float(line.strip().split()[3]) + 1) / 2
        score = float(line.strip().split()[3])
        y_pred_proba.append(score)
        realclass = 0.0 if line.strip().split()[2] == "spoof" else 1.0
        y_test.append(realclass)

y_pred_proba = np.array(y_pred_proba)
y_test = np.array(y_test)

# Adding jitter to the x-axis for better visualization
x_jitter = np.random.normal(0, 0.03, size=len(y_pred_proba))  # Mean=0, small std dev for jitter

# Create a scatter plot in one dimension with jitter
plt.figure(figsize=(10, 6))
scatter = plt.scatter(y_pred_proba, x_jitter, c=y_test, cmap='coolwarm', s=10, alpha=0.2)

# Add a color bar
plt.colorbar(scatter, label='True Class (0: Spoof, 1: Non-spoof)')
plt.xlabel('Predicted Probability')
plt.ylabel('Jitter (for visualization)')
plt.title('Scatter Plot')
plt.show()