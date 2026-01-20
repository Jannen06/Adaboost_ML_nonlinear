import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from load_data import load_wdbc, to_pca_2d
from adaboost_from_scratch import AdaBoost

class AdaBoostLiveDemo:
    def __init__(self):
        # Load data
        print("Loading Breast Cancer Wisconsin dataset...")
        X, y, _, _ = load_wdbc("./dataset/wdbc.data")
        self.X_2d, explained, _, _ = to_pca_2d(X)
        self.y = y
        
        print(f"Dataset: {len(y)} samples, 2D PCA explains {explained:.2%} variance")
        print(f"Classes: {np.sum(y == 1)} Malignant, {np.sum(y == -1)} Benign\n")
        
        # Train AdaBoost
        print("Training AdaBoost with 50 weak learners...")
        self.ada = AdaBoost(n_estimators=50)
        self.ada.fit(self.X_2d, y)
        print("Training complete!\n")
        
        # Prepare mesh for decision boundary
        x_min, x_max = self.X_2d[:, 0].min() - 1, self.X_2d[:, 0].max() + 1
        y_min, y_max = self.X_2d[:, 1].min() - 1, self.X_2d[:, 1].max() + 1
        self.xx, self.yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                                        np.linspace(y_min, y_max, 150))
    
    def interactive_step_through(self):
        """Step through iterations with keyboard input"""
        iterations = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50]
        
        for i, n_iter in enumerate(iterations):
            plt.clf()
            
            # Compute decision boundary
            Z = self.ada.predict_at_iteration(np.c_[self.xx.ravel(), self.yy.ravel()], n_iter)
            Z = Z.reshape(self.xx.shape)
            
            # Plot
            plt.contourf(self.xx, self.yy, Z, alpha=0.2, levels=[-1, 0, 1], colors=['blue', 'red'])
            plt.contour(self.xx, self.yy, Z, levels=[0], colors='black', linewidths=3)
            plt.scatter(self.X_2d[:, 0], self.X_2d[:, 1], c=self.y, 
                       cmap='bwr', edgecolors='black', s=60, alpha=0.9)
            
            # Accuracy
            pred = self.ada.predict_at_iteration(self.X_2d, n_iter)
            acc = np.mean(pred == self.y)
            errors = np.sum(pred != self.y)
            
            plt.title(f'Iteration {n_iter} | Accuracy: {acc:.2%} | Errors: {errors}/{len(self.y)}', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Principal Component 1', fontsize=12)
            plt.ylabel('Principal Component 2', fontsize=12)
            plt.grid(alpha=0.3)
            
            # Print stats
            print(f"[Iteration {n_iter}] Accuracy: {acc:.2%} | Misclassified: {errors}")
            
            if i < len(iterations) - 1:
                plt.pause(2)  # Auto-advance after 2 seconds
                # Or use: input("Press Enter for next iteration...")
        
        plt.show()
    
    def animate(self, save=False):
        """Create animation of boundary evolution"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            n_iter = frame + 1
            
            # Decision boundary
            Z = self.ada.predict_at_iteration(np.c_[self.xx.ravel(), self.yy.ravel()], n_iter)
            Z = Z.reshape(self.xx.shape)
            
            ax.contourf(self.xx, self.yy, Z, alpha=0.2, levels=[-1, 0, 1], colors=['blue', 'red'])
            ax.contour(self.xx, self.yy, Z, levels=[0], colors='black', linewidths=3)
            ax.scatter(self.X_2d[:, 0], self.X_2d[:, 1], c=self.y, 
                      cmap='bwr', edgecolors='black', s=60, alpha=0.9)
            
            pred = self.ada.predict_at_iteration(self.X_2d, n_iter)
            acc = np.mean(pred == self.y)
            
            ax.set_title(f'AdaBoost Evolution | Iteration {n_iter}/50 | Accuracy: {acc:.2%}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('PC1', fontsize=11)
            ax.set_ylabel('PC2', fontsize=11)
            ax.grid(alpha=0.3)
        
        anim = FuncAnimation(fig, update, frames=50, interval=200, repeat=True)
        
        if save:
            print("Saving animation (this may take a minute)...")
            anim.save('adaboost_animation.gif', writer='pillow', fps=5)
            print("Saved: adaboost_animation.gif")
        
        plt.show()


if __name__ == "__main__":
    demo = AdaBoostLiveDemo()
    
    print("\n=== Choose Demo Mode ===")
    print("1. Interactive step-through (press Enter between iterations)")
    print("2. Animated evolution (auto-play)")
    print("3. Save animation as GIF")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\nStarting interactive demo...\n")
        demo.interactive_step_through()
    elif choice == '2':
        print("\nStarting animation...\n")
        demo.animate(save=False)
    elif choice == '3':
        demo.animate(save=True)
    else:
        print("Invalid choice, running interactive mode...")
        demo.interactive_step_through()
