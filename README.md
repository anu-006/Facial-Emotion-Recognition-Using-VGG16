# Facial Emotion Detection using CNN & Transfer Learning (VGG16)

A deep learning project that classifies facial expressions into seven emotion categories using **VGG16 transfer learning** and a custom neural-network classification head built with TensorFlow/Keras.

The notebook covers dataset preparation, transfer learning, model training, learning-curve visualization, and model serialization.

## Project Highlights

- Uses **VGG16 pretrained on ImageNet** as a frozen feature extractor.
- Adds a custom classification head with Global Average Pooling, Dense layers, and Dropout.
- Performs seven-class emotion classification.
- Uses EarlyStopping with `restore_best_weights=True`.
- Continues training from epoch 21 after the initial 20-epoch run.
- Plots training/validation accuracy and loss.
- Saves the trained model in Keras `.keras` format.

## Dataset

The notebook downloads the dataset from Kaggle:

**[Balanced RAF-DB Dataset – 7,575 Grayscale (Kaggle)](https://www.kaggle.com/datasets/dollyprajapati182/balanced-raf-db-dataset-7575-grayscale)**

The notebook's directory loader reports the following number of images:

| Split | Images |
|---|---:|
| Training | 30,023 |
| Validation | 7,504 |
| Test | 4,165 |
| **Total** | **41,692** |

The seven class labels are inferred from the directory names and sorted alphabetically:

`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`


## Model Architecture

The model uses VGG16 with `include_top=False` and ImageNet weights. Its convolutional layers are frozen (`trainable=False`), so the network acts as a feature extractor.

The extracted features pass through this custom classification head:

1. GlobalAveragePooling2D
2. Dense(256, ReLU) + Dropout(0.3)
3. Dense(256, ReLU) + Dropout(0.5)
4. Dense(128, ReLU) + Dropout(0.3)
5. Dense(7, Softmax)

**Training configuration**

| Setting | Value |
|---|---|
| Input image size | 224 × 224 × 3 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Categorical cross-entropy |
| Initial training run | Up to 20 epochs |
| Continued training run | Epochs 21–30 (stopped early) |
| Early stopping | Monitor `val_loss`, patience 3 |
| Pretrained backbone | VGG16 / ImageNet |

The notebook reports 14,945,607 total parameters, of which 230,919 are trainable and 14,714,688 are non-trainable.

## Recorded Training Results

The notebook's saved output shows:

| Metric | Recorded result |
|---|---:|
| Training accuracy at epoch 29 | 74.31% |
| Best validation accuracy (epoch 26) | 72.93% |
| Validation loss at epoch 26 | 0.7761 |

These are training/validation metrics from the notebook output.

## Repository Structure

A suggested structure for publishing this project:

```text
emotion-detection-vgg16/
├── Emotion_Detection_using__CNN+TransferLearning.ipynb
├── README.md
├── requirements.txt
├── models/
   └── emotion_detection_vgg16.keras
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/anu-006/Facial-Emotion-Recognition-Using-VGG16
cd emotion-detection-vgg16
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Kaggle access

The notebook downloads the dataset using the Kaggle CLI. Configure your Kaggle API credentials according to the [Kaggle API documentation](https://github.com/Kaggle/kaggle-api).

Then run the notebook's dataset download and extraction cells. The notebook expects the extracted folders at:

```text
/content/dataset_emotion_detection/train
/content/dataset_emotion_detection/val
/content/dataset_emotion_detection/test
```

These paths are specific to Google Colab. If running locally, update them to the actual dataset location.

### 5. Run the notebook

Open the notebook in Jupyter or Google Colab and execute the cells in order. A GPU runtime is recommended for training; the uploaded notebook metadata indicates a Google Colab T4 GPU runtime.

The VGG16 ImageNet weights are downloaded automatically by Keras when they are not already cached.

## Saved Model

The notebook saves the trained model as:

```text
emotion_detection_vgg16.keras
```

Load it with:

```python
from tensorflow import keras

model = keras.models.load_model("emotion_detection_vgg16.keras")
```

For inference, apply the same image resizing and VGG16 `preprocess_input` preprocessing used during training, and provide a three-channel image tensor of shape `(batch_size, 224, 224, 3)`.



## Tech Stack

- Python
- TensorFlow / Keras
- VGG16 (ImageNet transfer learning)
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Kaggle API
- Jupyter Notebook / Google Colab

## Acknowledgements

- VGG16 architecture and ImageNet pretrained weights, provided through Keras Applications.
- Dataset: [Balanced RAF-DB Dataset – 7,575 Grayscale](https://www.kaggle.com/datasets/dollyprajapati182/balanced-raf-db-dataset-7575-grayscale).
