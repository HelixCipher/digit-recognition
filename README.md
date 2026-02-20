# Digit Recognition

<p align="center">
  <img src="https://github.com/HelixCipher/digit-recognition/blob/main/Project_Demonstration_GIF.gif" width="800" alt="Project_Demonstration_GIF.gif"/>
</p>

A Streamlit web application for recognizing handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Features

- **Draw Mode**: Draw a digit directly on the canvas.

- **Upload Mode**: Upload an image of a handwritten digit.

- **Real-time Prediction**: See prediction results instantly.

- **GPU Support**: Automatically detects and uses GPU if available.

## Installation

1. Clone the repository.

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## GPU Support (Optional)

For GPU acceleration, install the following:

- **CUDA Toolkit 12.x** - [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

- **cuDNN 8.x** - [NVIDIA cuDNN Downloads](https://developer.nvidia.com/cudnn)

After installation, restart the app. The app will display "GPU: X detected" in the header if successful.

### Verifying GPU Installation

To verify CUDA is installed, run:

```bash
nvidia-smi
```

This should display GPU information if CUDA is properly configured.

### Troubleshooting

If GPU is not detected after installing CUDA and cuDNN:

1. Ensure CUDA binaries are in your system PATH.

2. Verify cuDNN files are in the CUDA installation directory.

3. Restart your terminal or computer.

4. Reinstall TensorFlow: `pip install --upgrade tensorflow`.

If GPU is unavailable, the app will automatically run on CPU.

## Running the App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Usage

### Draw Tab
1. Use your mouse to draw a digit (0-9) on the canvas.

2. The model will automatically recognize the digit and display prediction probabilities.

### Upload Image Tab
1. Upload an image file (PNG, JPG, or JPEG).

2. The model will process the image and display prediction results.

## Model Details

- **Architecture**: CNN with 3 convolutional layers, max pooling, and 2 dense layers.

- **Training Data**: MNIST dataset (60,000 training images, 10,000 test images).

- **Data Augmentation**: Rotation, width/height shift, zoom.

- **Accuracy**: ~99% on test set

## Files

- `app.py` - Main Streamlit application.

- `training.py` - Model training script.

- `model.keras` - Pre-trained Keras model.

- `requirements.txt` - Python dependencies.

---

## License & Attribution

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

You are free to **use, share, copy, modify, and redistribute** this material for any purpose (including commercial use), **provided that proper attribution is given**.

### Attribution requirements

Any reuse, redistribution, or derivative work **must** include:

1. **The creator’s name**: `HelixCipher`
2. **A link to the original repository**:  
   https://github.com/HelixCipher/digit-recognition
3. **An indication of whether changes were made**
4. **A reference to the license (CC BY 4.0)**

#### Example Attribution

> This work is based on *Digit Recognition* by `HelixCipher`.  
> Original source: https://github.com/HelixCipher/digit-recognition
> Licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0).

You may place this attribution in a README, documentation, credits section, or other visible location appropriate to the medium.

Full license text: https://creativecommons.org/licenses/by/4.0/


---

## Disclaimer

This project is provided **“as—is”**. The author accepts no responsibility for how this material is used. There is **no warranty** or guarantee that the scripts are safe, secure, or appropriate for any particular purpose. Use at your own risk.

see [DISCLAIMER.md](./DISCLAIMER.md) for full terms. Use at your own risk.
