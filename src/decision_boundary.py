import matplotlib.pyplot as plt
import numpy as np
from load_data import load_wdbc, to_pca_2d
from adaboost_from_scratch import AdaBoost

# Load and prepare data
X, y, _, _ = load_wdbc("../dataset/wdbc.data")
X_2d, explained_var, scaler, pca = to_pca_2d(X)

print(f"PCA 2D explains {explained_var:.2%} of variance")

# Train AdaBoost on 2D PCA data
ada = AdaBoost(n_estimators=50)
ada.fit(X_2d, y)

# Create mesh grid for decision boundary
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Plot evolution of decision boundaries
iterations_to_plot = [1, 5, 10, 20, 30, 50]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, n_iter in enumerate(iterations_to_plot):
    # Predict on mesh grid
    Z = ada.predict_at_iteration(np.c_[xx.ravel(), yy.ravel()], n_iter)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    axes[idx].contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], 
                       colors=['blue', 'red'], cmap='bwr')
    axes[idx].contour(xx, yy, Z, levels=[0], colors='black', 
                      linewidths=2, linestyles='--')
    
    # Plot data points
    scatter = axes[idx].scatter(X_2d[:, 0], X_2d[:, 1], c=y, 
                                cmap='bwr', edgecolors='black', 
                                s=50, alpha=0.8)
    
    # Calculate accuracy
    pred = ada.predict_at_iteration(X_2d, n_iter)
    acc = np.mean(pred == y)
    
    axes[idx].set_title(f'Iteration {n_iter} | Accuracy: {acc:.2%}', 
                        fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('PC1', fontsize=11)
    axes[idx].set_ylabel('PC2', fontsize=11)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('decision_boundary_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nSaved: decision_boundary_evolution.png")
