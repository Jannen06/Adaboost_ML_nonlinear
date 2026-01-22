# AdaBoost From Scratch: Breast Cancer Classification

A Python implementation of the AdaBoost (Adaptive Boosting) algorithm from scratch, with interactive visualizations demonstrating how decision boundaries evolve as weak learners combine into a strong classifier.

## Project Overview

This project implements AdaBoost using decision tree stumps (max_depth=1) to classify breast cancer tumors as malignant or benign using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

### Key Features
- AdaBoost implementation from scratch - No sklearn AdaBoost, only decision trees
- Interactive visualizations - Watch decision boundaries evolve in real-time
- High accuracy - Achieves ~96% accuracy on test data
- Educational focus - Clear code structure for learning purposes

## Results

| Learners | Accuracy | Errors |
|----------|----------|--------|
| 1        | ~92%     | 45/569 |
| 5        | ~93%     | 35/569 |
| 10       | ~94%     | 32/569 |
| 50       | ~96%     | 23/569 |

## Project Structure

```
.
├── presentation/
│   ├── adaptive_boosting_presentation.tex    # LaTeX presentation source
│   ├── adaptive_boosting_presentation.pdf    # Compiled presentation
│   ├── boundary_progression.png              # Visualization figures
│   ├── pca_visualization.png
│   ├── adaboost_animation.mp4
│   └── Presentation_outline.md
├── src/
│   ├── dataset/
│   │   ├── wdbc.data                         # Breast cancer dataset (569 samples)
│   │   └── wdbc.names                        # Dataset documentation
│   └── scripts/
│       ├── adaboost_from_scratch.py          # Core AdaBoost implementation
│       ├── load_data.py                      # Data loading and PCA transformation
│       ├── live_demo.py                      # Interactive visualization
│       ├── visualize_boundary_comparison.py  # Static boundary plots
│       ├── decision_boundary.py              # Decision boundary evolution
│       └── visualize_adaboost_pca.py         # PCA visualization
├── README.md
└── requirements.txt
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/adaboost-breast-cancer.git
cd adaboost-breast-cancer
```

2. Create a virtual environment (recommended):
```bash
python -m venv ml_venv
source ml_venv/bin/activate  
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
```
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

## Usage

### 1. Interactive Live Demo (Recommended)

Run the interactive demo to see AdaBoost in action:

```bash
cd src/scripts
python live_demo.py
```

**Interactive options:**
- Try different numbers of estimators (50, 100, 200)
- Choose between:
  - **Step-through mode**: Press Enter to advance through iterations
  - **Animated mode**: Watch boundaries evolve automatically
  - **Save animation**: Export as GIF

**Example session:**
```
Enter number of estimators: 50
Train: 455 samples | Test: 114 samples
Training AdaBoost with 50 weak learners...

=== Choose Demo Mode ===
1. Interactive step-through
2. Animated evolution
Enter choice (1/2): 2
```

### 2. Generate Static Visualizations

```bash
cd src/scripts

# Boundary comparison plots
python visualize_boundary_comparison.py

# Decision boundary evolution
python decision_boundary.py

# PCA progression visualization
python visualize_adaboost_pca.py
```

### 3. Use AdaBoost in Your Code

```python
from load_data import load_wdbc, to_pca_2d
from adaboost_from_scratch import AdaBoost
from sklearn.model_selection import train_test_split

# Load data
X, y, _, _ = load_wdbc("../dataset/wdbc.data")
X_2d, explained_var, _, _ = to_pca_2d(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.2, random_state=42
)

# Train AdaBoost
ada = AdaBoost(n_estimators=50)
ada.fit(X_train, y_train)

# Predict
predictions = ada.predict(X_test)

# Evaluate
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.2%}")
```

## Dataset Information

**Wisconsin Diagnostic Breast Cancer (WDBC)**
- **Samples**: 569 patients
- **Features**: 30 real-valued features computed from cell nuclei images
- **Classes**: 
  - Malignant (M): 212 samples → label +1
  - Benign (B): 357 samples → label -1
- **Source**: UCI Machine Learning Repository

**Features include:**
- Radius, texture, perimeter, area
- Smoothness, compactness, concavity
- Symmetry, fractal dimension
- (Mean, standard error, and "worst" values for each)

**PCA Reduction:**
- 2D PCA explains ~63% of variance
- Used for visualization while maintaining separability

## Implementation Details

### AdaBoost Class
- **n_estimators**: Number of weak learners (default: 50)
- **Weak learner**: Decision tree with max_depth=1 (stump)
- **Weight update**: Exponential weighting scheme
- **Prediction**: Weighted voting of all stumps

### Train/Test Split
- **Training**: 80% (455 samples)
- **Testing**: 20% (114 samples)
- **Random state**: 42 (reproducible)

### Visualization
- **Grid resolution**: 150×150 = 22,500 points
- **Color scheme**: Blue (benign), Red (malignant)
- **Boundary**: Black line where prediction = 0

## Mathematical Foundation

**Weight Update Formula:**
```
w_n^(k+1) = w_n^(k) × exp(-α_k × y_n × h_k(x_n))
```

**Learner Weight:**
```
α_k = 0.5 × ln((1 - ε_k) / ε_k)
```

**Final Classification:**
```
H(x) = sign(Σ_{k=1}^K α_k × h_k(x))
```

Where:
- w_n = weight of sample n
- α_k = weight of learner k
- ε_k = weighted error of learner k
- h_k(x) = prediction of learner k
- y_n = true label (+1 or -1)

### Suggested Experiments
1. Change number of estimators (10, 50, 100, 200)
2. Try different weak learners (max_depth=2, 3)
3. Modify weight update formula
4. Test on full 30-D features (no PCA)
5. Compare with sklearn's AdaBoost


## Acknowledgments

- **Dataset**: UCI Machine Learning Repository
  - W.N. Street, W.H. Wolberg and O.L. Mangasarian (1993)
- **Algorithm**: Freund & Schapire (1997)
- **Inspiration**: Educational machine learning projects

## Contact

For questions or feedback, please open an issue on GitHub.

## References

1. Freund, Y., & Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119-139.

2. Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology*, 1905, 861-870.

---

**If you find this project helpful, please give it a star!**