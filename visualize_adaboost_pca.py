import matplotlib.pyplot as plt
from load_data import load_wdbc, to_pca_2d
from adaboost_from_scratch import AdaBoost
import numpy as np

# Load data
X, y, _, _ = load_wdbc("dataset/wdbc.data")
X_2d, _, _, _ = to_pca_2d(X)

# Generate random predictions for initial visualization
np.random.seed(42)  # for reproducibility
random_pred = np.random.choice([-1, 1], size=len(y))

# Train AdaBoost
ada = AdaBoost(n_estimators=50)
ada.fit(X, y)

# Print accuracies
for i in range(1, 51, 5):
    pred = ada.predict_at_iteration(X, i)
    acc = np.mean(pred == y)
    print(f'Iteration {i}: Accuracy {acc:.2%}')

# Create subplots for selected iterations
iterations_to_plot = ['Initial', 'Ground Truth', 5, 10, 20, 30, 40, 50]
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

for idx, i in enumerate(iterations_to_plot):
    if i == 'Initial':
        c = 'gray'
        cmap = None
        title = 'Initial Data Distribution'
    elif i == 'Ground Truth':
        c = y
        cmap = 'bwr'
        title = 'Ground Truth'
    else:
        c = ada.predict_at_iteration(X, i)
        cmap = 'bwr'
        title = f'Iteration {i}'
    axes[idx].scatter(X_2d[:, 0], X_2d[:, 1], c=c, cmap=cmap, alpha=0.7)
    axes[idx].set_title(title)

plt.tight_layout()
plt.savefig('adaboost_progression.png')
plt.show()