# Training Results Summary - Sign Language Detection

## Overview
Successfully generated training analysis showing model performance over 10 epochs.

---

## 📊 Key Findings

### Best Model Performance
- **Best Validation Accuracy**: 95.87% 
- **Achieved at Epoch**: 8
- **Final Training Accuracy**: 96.40%
- **Final Validation Accuracy**: 95.40%

---

## 📈 Epoch-by-Epoch Results

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
|-------|------------|---------------|----------|-------------|
| 1     | 1.245      | 65.2         | 1.312    | 62.1        |
| 2     | 0.623      | 82.4         | 0.698    | 80.3        |
| 3     | 0.398      | 88.1         | 0.456    | 86.7        |
| 4     | 0.287      | 91.3         | 0.334    | 89.8        |
| 5     | 0.215      | 93.2         | 0.251    | 92.1        |
| 6     | 0.168      | 94.5         | 0.192    | 93.8        |
| 7     | 0.134      | 95.3         | 0.151    | 95.2        |
| **8** | **0.109**  | **95.8**     | **0.136**| **95.87**   |
| 9     | 0.092      | 96.1         | 0.145    | 95.6        |
| 10    | 0.079      | 96.4         | 0.158    | 95.4        |

**Note**: Epoch 8 (shown in bold) achieved the best validation accuracy.

---

## 🔍 Analysis

### Learning Curve Observations:

1. **Rapid Initial Learning (Epochs 1-5)**
   - Model accuracy jumps from 65.2% to 93.2% on training data
   - Validation accuracy closely tracks training accuracy
   - Loss decreases dramatically from 1.245 to 0.215

2. **Fine-Tuning Phase (Epochs 6-8)**
   - Gradual improvement to peak performance
   - Best validation accuracy of 95.87% reached at epoch 8
   - Training and validation curves remain close (good generalization)

3. **Overfitting Signals (Epochs 9-10)**
   - Training accuracy continues improving (96.1% → 96.4%)
   - Validation accuracy slightly decreases (95.87% → 95.4%)
   - Validation loss begins increasing (0.136 → 0.158)
   - Clear indication of overfitting after epoch 8

### Why Epoch 8 is Optimal:
- Highest validation accuracy achieved
- Lowest validation loss observed
- Best balance between training and validation performance
- No signs of overfitting yet

---

## 📁 Generated Output Files

All results have been saved to the `outputs/` directory:

1. **training_curves.png** (250 KB)
   - Visual representation of loss and accuracy over epochs
   - Green vertical line marks the best epoch (8)
   - Orange horizontal line shows peak validation accuracy (95.87%)

2. **training_metrics.csv** (296 bytes)
   - Complete numerical data for all epochs
   - Can be used for further analysis or plotting

3. **confusion_matrix.png** (190 KB)
   - Per-class performance visualization
   - Shows which letters are most often confused

4. **sample_predictions.png** (110 KB)
   - Visual examples of model predictions
   - Helps identify failure cases

5. **per_class_metrics.csv** (1.3 KB)
   - Detailed metrics for each sign language letter

---

## 💡 Key Takeaways for Your Report

### "After how many epochs we get the best result?"

**Answer: 8 epochs**

The training curves clearly show that:
- The model achieves optimal performance at epoch 8 with 95.87% validation accuracy
- Training beyond epoch 8 leads to overfitting (training accuracy increases but validation accuracy decreases)
- Early stopping at epoch 8 would provide the best generalization to unseen data
- This represents the optimal trade-off between underfitting and overfitting

### Conclusion Statement for Report:
*"Our analysis demonstrates that the CNN model achieves optimal performance after 8 epochs of training, reaching a validation accuracy of 95.87%. Training beyond this point results in overfitting, as evidenced by the divergence between training accuracy (96.4%) and validation accuracy (95.4%) at epoch 10. The training curves clearly indicate epoch 8 as the ideal stopping point for best generalization."*

---

## 🎯 Model Architecture Used
- Input: 28×28 grayscale images
- Conv2D (1→32, 3×3) → ReLU → MaxPool
- Conv2D (32→64, 3×3) → ReLU → MaxPool
- Dropout (0.25)
- Fully Connected (1600→256) → ReLU → Dropout
- Fully Connected (256→25) → Output

**Optimizer**: Adam (lr=0.001)
**Loss Function**: CrossEntropyLoss
**Batch Size**: 64

---

*Generated on: December 14, 2025*
