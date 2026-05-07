# ICEEP (image classification exhaustive experiment project)

# Introduction 
This is a study of various combinations of deep learning pipeline choices.

It was done using CIFAR-10 Dataset consisting 60000 32x32 colour images in 10 classes.

We used tensor-flow for this process.

We are a team of 2: **Islam Waleed** (experiments 4->5) and **Hanif Aderolu** (experiments 1->3) 


## Data experiments 
### experiment 1.1: Normalization Comparison

This experiment compared the performance of the BaselineCNN model using different data normalization techniques: no normalization (raw pixels), Min-Max scaling to [0,1], and standardization per channel.

**Results:**

| Preprocessing           | Train Acc | Val Acc | Test Acc | Loss@Ep1 |
|:------------------------|----------:|--------:|---------:|---------:|
| No Norm (0–255)         |    0.8961 |  0.5767 |   0.5659 |   4.4016 |
| Min-Max [0,1]           |    0.9025 |  0.6646 |   0.6576 |   1.5660 |
| Standardized            |    0.9742 |  0.6782 |   0.6703 |   1.4101 |

**Conclusion:** Standardization per channel yielded the best test accuracy, indicating its effectiveness in preparing the data for the CNN model. Min-Max normalization also significantly improved performance compared to no normalization.

### experiment 1.2: Data Augmentation Comparison

This task compared training the baseline model with and without data augmentation using standardized data over 40 epochs.

**Results:**

- **No Augmentation:** Test Acc: 0.6793 | Test Loss: 3.0829 | Time: 575.4s
- **With Augmentation:** The model encountered a `ValueError` during training, preventing the completion of this comparison.

## network experiments 

### experiment 2.1: Filter Count Comparison

This experiment evaluated the impact of varying filter counts in a two-block CNN model (Medium filter configuration) on performance.

**Results:**

| Model      | Total Params | Train Acc | Val Acc | Test Acc | Time (s) |
|:-----------|-------------:|----------:|--------:|---------:|---------:|
| Small      |      269,266 |    0.9387 |  0.6131 |   0.6058 |     44.1 |
| Medium     |    1,116,970 |    0.9861 |  0.7227 |   0.7150 |     78.0 |
| Large      |    2,360,138 |    0.9890 |  0.7474 |   0.7384 |    148.6 |

**Conclusion:** Larger filter counts (Medium and Large models) generally led to better test accuracy, although at the cost of increased training time and number of parameters. The 'Large' model achieved the highest test accuracy.

### experiment 2.2: Network Depth Comparison

This task investigated the effect of network depth by comparing shallow (4 conv layers), medium (6 conv layers), and deep (8 conv layers) models, all using 32 filters per layer.

**Results:**

| Model      | Total Params | Train Acc | Val Acc | Test Acc | Time (s) |
|:-----------|-------------:|----------:|--------:|---------:|---------:|
| Shallow    |      555,754 |    0.9747 |  0.7018 |   0.6966 |     71.1 |
| Medium     |       58,154 |    0.7967 |  0.7380 |   0.7301 |     72.1 |
| Deep       |       76,650 |    0.8269 |  0.7210 |   0.7206 |     75.8 |

**Conclusion:** The medium-depth model achieved the highest test accuracy. The analysis suggests that for small image resolutions like CIFAR-10, deeper models can lead to loss of spatial information due to aggressive pooling or may overfit without sufficient regularization. The medium-depth model strikes a good balance.

## internal process experiments  

### experiment 3.1: Dropout Rate Comparison

This experiment explored the effect of different dropout rates (0.00, 0.25, 0.50) on the medium-sized CNN model.

**Results:**

- **D0 (rate=0.00):** The model encountered a `ValueError` during training, preventing the completion of this comparison.

### experiment 3.2: Early Stopping Comparison

This task evaluated the impact of early stopping with different patience values (5 and 10) compared to no early stopping.

**Results:**

| Experiment         | Stopped@Ep | Best Val Loss | Test Acc | Time (s) |
|:-------------------|-----------:|--------------:|---------:|---------:|
| ES0 (no ES)        |         50 |        0.7512 |   0.7183 |    176.7 |
| ES2 (pat=5)        |         19 |        0.6698 |   0.7296 |     66.7 |
| ES3 (pat=10)       |         29 |        Performance0.6723 |   0.7329 |     95.9 |

**Conclusion:** Early stopping significantly reduced training time while achieving comparable or better test accuracy. `ES3 (patience=10)` yielded the highest test accuracy and stopped at an earlier epoch compared to no early stopping, suggesting a good balance between convergence and preventing overfitting.

### experiment 4.1: Optimizer Comparison

This experiment compared the performance of different optimizers (SGD, Momentum, AdaGrad, RMSProp, Adam) using the 'Medium' filter model with standardized data.

**Results:**

| Optimizer  | Train Acc | Val Acc | Test Acc | Time (s) |
|:-----------|----------:|--------:|---------:|---------:|
| SGD        |    0.5417 |  0.5217 |   0.5236 |    121.8 |
| Momentum   |    0.9268 |  0.6517 |   0.6434 |    113.2 |
| AdaGrad    |    0.9998 |  0.6937 |   0.6857 |    113.1 |
| RMSProp    |    0.9908 |  0.7134 |   0.7083 |    112.3 |
| Adam       |    0.9934 |  0.7174 |   0.7102 |    113.7 |

**Conclusion:** Adaptive optimizers like Adam and RMSProp generally performed better than SGD and Momentum, achieving higher test accuracies. Adam slightly outperformed RMSProp in terms of validation and test accuracy.


### experiment 4.2: Learning Rate Comparison (with Adam)

This task evaluated the Adam optimizer with different learning rates (0.01, 0.001, 0.0001) using the 'Medium' filter model.

**Results:**

| Learning Rate | Train Acc | Val Acc | Test Acc | Time (s) |
|--------------:|----------:|--------:|---------:|---------:|
|          0.01 |    0.8035 |  0.4682 |   0.4739 |    111.7 |
|         0.001 |    0.9934 |  0.7174 |   0.7126 |    116.0 |
|        0.0001 |    0.9701 |  0.6625 |   0.6652 |    116.6 |

**Conclusion:** A learning rate of 0.001 yielded the best test accuracy. A higher learning rate (0.01) resulted in unstable training and lower performance, while a lower learning rate (0.0001) led to slower convergence and slightly lower final accuracy.

### Final Model Performance (no fine tuning) 

The final model configuration, combining insights from previous experiments (medium depth, Adam optimizer with learning rate 0.001, dropout rate 0.25, and early stopping with patience 7), achieved the following:

- **Test Accuracy:** 0.7627
- **Test Loss:** 0.6735
- **Training Time:** 177.4s

### experiment 5.1 : Transfer Learning with VGG16

This task explored the use of pre-trained VGG16 for CIFAR-10 classification, comparing a transfer learning approach (freezing VGG base and training a new head) with fine-tuning (unfreezing the last few VGG layers).

**Data Preprocessing:** Images were resized from 32x32 to 48x48 to match VGG16 input requirements and standardized.

**Results:**

| Model                  | Test Acc | Test Loss | Time (s) |
|:-----------------------|---------:|----------:|---------:|
| Medium CNN (48x48)     |   0.6843 |    2.9442 |    154.6 |
| VGG16 Transfer Learning|   0.8407 |    0.4632 |    189.6 |
| VGG16 Fine-Tuning      |   0.8643 |    0.3957 |    208.6 |

**Conclusion:** Both VGG16 transfer learning and fine-tuning significantly outperformed the custom Medium CNN model trained from scratch on 48x48 images. Fine-tuning, by unfreezing and training the last 4 layers of the VGG16 base with a lower learning rate, achieved the highest test accuracy (0.8643), demonstrating the benefits of leveraging pre-trained large models for feature extraction and adapting them to specific tasks.
