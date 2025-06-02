# AI vs Real Image Detection - Streamlit Application

## Overview
This Streamlit application allows users to detect whether an image is AI-generated or real using two different deep learning models: ResNet18 and Vision Transformer (ViT). Users can upload their own images, select which model(s) to use, and view detailed analysis results with confidence scores and visual comparisons.

## Features
- Image upload functionality with preview
- Support for both ResNet18 and Vision Transformer models
- Side-by-side model comparison option
- Detailed visualization of prediction results
- Confidence score display for both classes
- Performance metrics (processing time)
- Responsive design for different screen sizes

## Project Structure
```
streamlit_app/
├── app.py              # Main Streamlit application
├── models.py           # Model implementation classes
├── ui_design.md        # UI design documentation
└── samples/            # Sample images directory
    └── sample_real.jpg # Default sample image
```

## Installation and Setup

### Prerequisites
- Python 3.8+
- PyTorch
- Streamlit
- Transformers library

### Installation Steps
1. Clone the repository or extract the provided files
2. Install the required dependencies:
   ```
   pip install streamlit torch torchvision transformers pillow matplotlib seaborn
   ```
3. Navigate to the project directory

## Running the Application
Run the Streamlit application with:
```
streamlit run app.py
```

## Usage Instructions
1. Upload an image using the file uploader or use the provided sample image
2. Select which model(s) to use for analysis:
   - ResNet18: Faster but potentially less accurate
   - Vision Transformer: More accurate but potentially slower
   - Compare Both: Run both models and compare results
3. Click the "Analyze" button to process the image
4. View the results, including:
   - Prediction (Real or AI-generated)
   - Confidence scores
   - Processing time
   - Comparison charts (when using both models)

## Model Information
- **ResNet18**: A residual network with 18 layers, pre-trained on ImageNet and fine-tuned for AI vs real image detection
- **Vision Transformer**: A transformer-based model that divides images into patches and processes them with self-attention mechanisms

## Technical Notes
- The application uses PyTorch for model inference
- Images are preprocessed to 224x224 pixels and normalized
- Models are cached to improve performance for subsequent analyses
- GPU acceleration is used when available

## Limitations
- The models are based on the training data provided and may not generalize to all types of AI-generated images
- Performance may vary depending on image quality and characteristics
- First-time model loading may take a few seconds

## Future Improvements
- Add more model options
- Implement batch processing for multiple images
- Add detailed explanations of model decisions
- Improve performance on edge cases
