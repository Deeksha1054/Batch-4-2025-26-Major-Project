# Batch-4-2025-26-Major-Project
 Link to all the Trained Models used in the project - ( https://drive.google.com/drive/folders/1itVy2f_hNvaREKriNFawW4tID-GUDBvR?usp=sharing)
 
# Explainable Multi-Model Diabetic Retinopathy Detection System 🩺

##  Project Overview
Diabetic Retinopathy (DR) is a leading cause of blindness among diabetic patients, driven by retinal blood vessel damage. Early detection is critical for effective treatment. This project introduces an **explainable multi-model deep learning system** for detecting DR from fundus images, achieving **96% accuracy** while providing interpretable results through Grad-CAM visualizations. The system leverages an ensemble of **ResNet50**, **EfficientNet**, and **MobileNetV4**, trained on over 10,000 fundus images, and includes a **Flask-based web interface** for real-time clinical screening.

##  Objectives
- Build a robust multi-model system for accurate DR detection from fundus images.
- Enhance clinical trust with explainable AI (Grad-CAM) for visual prediction insights.
- Achieve ≥95% accuracy across five DR severity levels: No DR, Mild, Moderate, Severe, Proliferative.
- Provide a user-friendly web interface for real-time DR screening.
- Overcome limitations like black-box models, overfitting, and high computational costs.

##  Methodology
###  Dataset
- **Source**: Combined APTOS, DDR, IDRiD, and Messidor datasets (>10,000 fundus images).
- **Split**: 70% training, 20% validation, 10% testing.

###  Preprocessing
- Resize images to 224x224 pixels.
- Normalize pixel values for consistency.
- Apply data augmentation (rotation, flip, brightness adjustment) to enhance model robustness.

###  Model Architecture
- **Framework**: PyTorch.
- **Models**: Ensemble of ResNet50, EfficientNet, and MobileNetV4.
- **Feature Fusion**: Concatenation of model features, followed by a fully connected layer for 5-class DR classification.
- **Explainability**: Grad-CAM generates heatmaps highlighting critical regions (e.g., microaneurysms, hemorrhages).

###  Training
- **Optimizer**: Adam.
- **Loss Function**: Cross-entropy.
- **Hyperparameters**: Batch size 32, 50 epochs, GPU-accelerated.
- **Performance**: 96% accuracy, 94% sensitivity, 95% specificity on the test set.

###  Web Interface
- **Framework**: Flask.
- **Features**: Upload fundus images, view DR severity predictions, and display Grad-CAM heatmaps for interpretability.

###  Block Diagram
```
[Fundus Image] → [Preprocessing: Resize, Normalize, Augment] → [Multi-Model Ensemble: ResNet50 + EfficientNet + MobileNetV4] → [Feature Fusion] → [Classification: DR Severity] → [Grad-CAM: Visual Explanations] → [Web Interface: Display Results]
```

##  Literature Review
| Study | Methodology | Dataset | Performance | Gaps & Limitations |
|-------|-------------|---------|-------------|--------------------|
| DeepDR Plus (2024) | CNNs for time-to-DR progression | 717,308 fundus images | Concordance: 0.754–0.846 | Focuses on progression, not grading; no explainability |
| ACCESS Trial (2024) | AI/LLMs for screening timing | 20 case scenarios | Identifies risk factors | No image-based detection; no explainability |
| RSG-Net (2025) | CNN for DR classification | Messidor-1 | 98.50% (binary), 89.60% (multi-class) | Single-model; no explainability |
| Ghosh Aronno et al. (2023) | CNN + DCGAN | EyePACS | F1-score: 0.9430 | Synthetic data biases; no explainability |

##  Installation
### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Flask 2.0+
- Dependencies: OpenCV, NumPy, Pandas, Pillow
- GPU (recommended for training)

### Setup Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/explainable-dr-detection.git
   cd explainable-dr-detection
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare the Dataset**:
   - Download APTOS, DDR, IDRiD, and Messidor datasets.
   - Place them in the `data/` directory.
   - Run preprocessing:
     ```bash
     python scripts/preprocess.py --data-dir data/
     ```

### Troubleshooting
- Ensure GPU drivers and CUDA are installed for PyTorch.
- Verify dataset paths in `config.py` match your directory structure.

##  Usage
### Training the Model
1. Train the ensemble model:
   ```bash
   python scripts/train.py --data-dir data/ --epochs 50 --batch-size 32 --output-dir models/
   ```
2. Model weights are saved to `models/ensemble.pth`.

### Running the Web Interface
1. Launch the Flask app:
   ```bash
   python app.py
   ```
2. Open `http://localhost:5000` in a browser to upload images and view results.

### Evaluating Performance
1. Evaluate the model on the test set:
   ```bash
   python scripts/evaluate.py --model-path models/ensemble.pth --data-dir data/test/
   ```
2. Outputs accuracy, sensitivity, and specificity metrics.

##  Results
- **Accuracy**: 96%
- **Sensitivity**: 94%
- **Specificity**: 95%
- **Highlights**:
  - Multi-model ensemble reduces overfitting and enhances robustness.
  - Grad-CAM visualizations improve clinical interpretability.
  - Flask interface enables seamless real-time DR screening.

##  Future Work
- Integrate Optical Coherence Tomography (OCT) images for multi-modal analysis.
- Deploy on cloud platforms (e.g., AWS, Azure) for scalability.
- Explore additional explainability methods (e.g., SHAP, LIME).

##  Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

##  Contributors
- Interal Guide: Dr. Agughasi Victor I
- Deeksha R G           4MH22CA014
- Abdul Z Baig          4MH22CA003
- Abu Hurer             4MH22CA025
- Hemanth               4MH23CA401


##  License
This project is licensed under the MIT License. 

##  Project Status
- **Current**: Model training and web interface fully functional.
- **Next Steps**: Cloud deployment and OCT integration.
