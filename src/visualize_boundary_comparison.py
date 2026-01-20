import matplotlib.pyplot as plt
import numpy as np
from load_data import load_wdbc, to_pca_2d
from adaboost_from_scratch import AdaBoost

# Load data
X, y, _, _ = load_wdbc("./dataset/wdbc.data")
X_2d, _, _, _ = to_pca_2d(X)

# Train AdaBoost with MANY estimators
ada = AdaBoost(n_estimators=50)  # Train on 50, not 5!
ada.fit(X_2d, y)

# Create mesh grid
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                     np.linspace(y_min, y_max, 150))

# Plot comparison with VISIBLE progression
iterations = [1, 5, 10, 50]  # Show clear progression
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, n_iter in enumerate(iterations):
    # Predict on mesh
    Z = ada.predict_at_iteration(np.c_[xx.ravel(), yy.ravel()], n_iter)
    Z = Z.reshape(xx.shape)
    
    # Decision boundary with clear colors
    axes[idx].contourf(xx, yy, Z, alpha=0.25, levels=[-1, 0, 1], colors=['blue', 'red'])
    axes[idx].contour(xx, yy, Z, levels=[0], colors='black', linewidths=3)
    
    # Data points
    axes[idx].scatter(X_2d[y == -1, 0], X_2d[y == -1, 1], 
                     c='blue', label='Benign', edgecolors='black', s=40, alpha=0.7)
    axes[idx].scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], 
                     c='red', label='Malignant', edgecolors='black', s=40, alpha=0.7)
    
    # Accuracy
    pred = ada.predict_at_iteration(X_2d, n_iter)
    acc = np.mean(pred == y)
    errors = np.sum(pred != y)
    
    axes[idx].set_title(f'{n_iter} Learner{"s" if n_iter > 1 else ""}\n'
                       f'{acc:.1%} Accuracy | {errors} Errors', 
                       fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('PC1', fontsize=11)
    axes[idx].set_ylabel('PC2', fontsize=11)
    axes[idx].grid(alpha=0.3)
    if idx == 0:
        axes[idx].legend(loc='upper left', fontsize=9)

plt.suptitle('Decision Boundary Evolution: From Linear to Non-Linear', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../boundary_progression.png', dpi=300, bbox_inches='tight')
print("Saved: boundary_progression.png")
plt.show()
