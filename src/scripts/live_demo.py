import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from load_data import load_wdbc, to_pca_2d
from adaboost_from_scratch import AdaBoost
from sklearn.model_selection import train_test_split

class AdaBoostLiveDemo:
    def __init__(self, n_estimators=50):
        # Load data
        print("Loading Breast Cancer Wisconsin dataset...")
        X, y, _, _ = load_wdbc("../dataset/wdbc.data")
        self.X_2d, explained, _, _ = to_pca_2d(X)
        self.y = y
        
        # Print dataset info (YES, keep this!)
        print(f"Dataset: {len(y)} samples, 2D PCA explains {explained:.2%} variance")
        print(f"Classes: {np.sum(y == 1)} Malignant, {np.sum(y == -1)} Benign\n")
        
        # SPLIT DATA (NEW)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_2d, y, test_size=0.2, random_state=42
        )
        print(f"Train: {len(self.y_train)} samples | Test: {len(self.y_test)} samples\n")
        
        # Train AdaBoost 
        print(f"Training AdaBoost with {n_estimators} weak learners...")
        self.ada = AdaBoost(n_estimators=n_estimators)  # Use parameter, not hardcoded 50
        self.ada.fit(self.X_train, self.y_train)  # Train on training data, not all data
        print("Training complete!\n")
        
        # Prepare mesh for decision boundary (YES, keep this!)
        x_min, x_max = self.X_2d[:, 0].min() - 1, self.X_2d[:, 0].max() + 1
        y_min, y_max = self.X_2d[:, 1].min() - 1, self.X_2d[:, 1].max() + 1
        self.xx, self.yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                                        np.linspace(y_min, y_max, 150))

        
    
    def interactive_step_through(self):
        # iterations = [1, 2, 3, 5, 10, 15, 20, 30, 40, 50]

        n = self.ada.n_estimators
        if n <= 50:
            iterations = [1, 2, 3, 5, 10, 15, 20, 30, 40, n]
        else:
            # For larger n, show more spread out iterations
            iterations = [1, 5, 10, 20, n//4, n//2, 3*n//4, n]

        
        for i, n_iter in enumerate(iterations):
            plt.clf()
            
            Z = self.ada.predict_at_iteration(np.c_[self.xx.ravel(), self.yy.ravel()], n_iter)
            Z = Z.reshape(self.xx.shape)
            
            plt.contourf(self.xx, self.yy, Z, alpha=0.2, levels=[-1, 0, 1], colors=['blue', 'red'])
            plt.contour(self.xx, self.yy, Z, levels=[0], colors='black', linewidths=3)
            
            # Plot ALL data (train + test) for visualization
            plt.scatter(self.X_2d[:, 0], self.X_2d[:, 1], c=self.y, 
                    cmap='bwr', edgecolors='black', s=60, alpha=0.9)
            
            # Calculate accuracy on TEST SET
            pred_test = self.ada.predict_at_iteration(self.X_test, n_iter)
            acc_test = np.mean(pred_test == self.y_test)
            errors_test = np.sum(pred_test != self.y_test)
            
            plt.title(f'Iteration {n_iter} | Test Accuracy: {acc_test:.2%} | Errors: {errors_test}/{len(self.y_test)}', 
                    fontsize=16, fontweight='bold')
            
            print(f"[Iteration {n_iter}] Test Accuracy: {acc_test:.2%}")
            
            if i < len(iterations) - 1:
                plt.pause(2)
        
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
            
            ax.set_title(f'AdaBoost Evolution | Iteration {n_iter}/{self.ada.n_estimators} | Accuracy: {acc:.2%}', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('PC1', fontsize=11)
            ax.set_ylabel('PC2', fontsize=11)
            ax.grid(alpha=0.3)
        
        # anim = FuncAnimation(fig, update, frames=50, interval=200, repeat=True)
        anim = FuncAnimation(fig, update, frames=self.ada.n_estimators, 
                        interval=200, repeat=True)        
        if save:
            print("Saving animation (this may take a minute)...")
            anim.save('adaboost_animation.gif', writer='pillow', fps=5)
            print("Saved: adaboost_animation.gif")
        
        plt.show()


if __name__ == "__main__":
    print("\n=== AdaBoost Interactive Demo ===")
    
    while True:
        n_est = input("\nEnter number of estimators (or 'q' to quit): ").strip()
        
        if n_est.lower() == 'q':
            print("Exiting...")
            break
        
        try:
            n_est = int(n_est)
            if n_est < 1:
                print("Please enter a positive number")
                continue
                
            # Create new demo with specified estimators
            demo = AdaBoostLiveDemo(n_estimators=n_est)
            
            print("\n=== Choose Demo Mode ===")
            print("1. Interactive step-through")
            print("2. Animated evolution")
            
            choice = input("Enter choice (1/2): ").strip()
            
            if choice == '1':
                demo.interactive_step_through()
            elif choice == '2':
                demo.animate(save=False)
            else:
                print("Invalid choice")
                
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")