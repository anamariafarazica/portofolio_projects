# Emotion Classification Pipeline for Video Content Analysis

## Overview

This pipeline performs end-to-end emotion analysis on video content by extracting audio, transcribing speech, round-translating it to English, and classifying emotions in the transcript. Built for the Content Intelligence Agency to help media makers analyze emotional content in their videos at scale.

**Pipeline Flow:**
```
Video/Audio Input → Audio Extraction → Speech-to-Text → Translation → Emotion Classification
```

The system identifies 7 emotions: **happiness, sadness, anger, neutral, disgust, fear, and surprise**.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)

---

## Features

- **Modular Design**: Each pipeline component (audio extraction, transcription, translation, classification) can run independently
- **DeBERTa-based Classification**: Fine-tuned microsoft/deberta-base model (125M parameters) achieving 73.6% accuracy
- **Batch Processing**: Efficiently processes multiple videos
- **Configurable**: Centralized configuration via `config.py`

**Technologies Used:**
- Python 3.10
- HuggingFace Transformers
- PyTorch
- Speech recognition libraries
- Translation APIs

---
## Customization

### Modifying Training Parameters

To adjust training hyperparameters, edit the training script or notebook directly:

**Key parameters in `DeBERTa_training.ipynb`:**
- `learning_rate`: 2e-5
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 10
- `max_length`: 128 tokens

**To change:**
1. Open the notebook: `jupyter notebook notebooks/DeBERTa.ipynb`
2. Navigate to the "Training Arguments" cell
3. Modify the `TrainingArguments` parameters
4. Re-run training

### Modifying Pipeline Settings

Each module accepts command-line arguments. Use `--help` to see available options:
```bash
python audio_extraction.py --help
python speech_to_text.py --help
python emotion_classifier.py --help
```

**Common parameters:**
- `--input`: Input file path
- `--output`: Output directory
- `--language`: Language code (default: 'en')
- `--batch_size`: Processing batch size
---
## Requirements

### Prerequisites
- **Python**: 3.10 or higher
- **Hardware**: 
  - CPU: Minimum 4 cores recommended
  - GPU: Optional but recommended (NVIDIA with CUDA support for faster inference)
  - RAM: Minimum 8GB, 16GB+ recommended for large videos
- **OS**: Windows, macOS, or Linux

---

## Installation

### Step 1: Clone or Download the Project
```bash
git clone <your-repository-url>
cd emotion-classification-pipeline
```

### Step 2: Create a Virtual Environment

**Using Conda (Recommended):**
```bash
conda create -n emotion_pipeline python=3.10
conda activate emotion_pipeline
```

**Using venv:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "from transformers import AutoTokenizer; print('Installation successful!')"
```

## Dataset

### Training Data
- **Source**: Unscripted entertainment TV show transcripts
- **Size**: 23,696 labeled English text samples
- **Format**: CSV with columns: `Sentence`, `Emotion`, `Label`

### Test Data
- **Source**: "Kitchen Nightmares" TV show transcripts
- **Size**: 1,210 labeled English text samples

### Emotion Labels (7 classes)
Based on Paul Ekman's six core emotions plus neutral:
```
0: Happiness
1: Sadness
2: Anger
3: Neutral
4: Disgust
5: Fear
6: Surprise
```

### Class Distribution
⚠️ **Note**: Class imbalance exists - Happiness (23%) and Neutral (32%) are over-represented, while Fear (5%) and Disgust (6%) are under-represented.

### Data Preprocessing
1. Removed missing values and duplicates
2. Lowercased all text for consistency
3. Tokenized with DeBERTa tokenizer (max length: 128 tokens)
4. Label encoding to integer classes (0-6)

**Data Location:**
```
data/
├── cleaned_train_data.csv
└── cleaned_test_data.csv
```

---

## Usage

### Running the Full Pipeline

Process a video file from start to finish:
```bash
python pipeline.py --input path/to/video.mp4 --language en --output output/
```

**Available options:**
- `--input`: Path to input video/audio file (required)
- `--language`: Source language code (default: 'en')
- `--output`: Output directory (default: 'output/')
- `--model_path`: Path to trained model (default: 'models/deberta-finetuned/')
- `--max_length`: Maximum token length (default: 128)

### Running Individual Components

#### 1. Audio Extraction
```bash
python audio_extraction.py --input video.mp4 --output audio_files/
```

#### 2. Speech-to-Text Transcription
```bash
python speech_to_text.py --input audio_files/audio.wav --output transcriptions/
```

#### 3. Translation
```bash
python translator.py --input transcriptions/transcript.txt --target_lang en
```

#### 4. Emotion Classification
```bash
python emotion_classifier.py --input translations/translated.txt --output predictions/
```

## Project Structure
```
emotion-classification-pipeline/
│
├── data/                          # Training and test datasets
│   ├── cleaned_train_data.csv
│   └── cleaned_test_data.csv
│
├── models/                        # Trained models
│   └── checkpoint-14810/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
│
├── notebooks/                     # Jupyter notebooks
│   └── DeBERTa.ipynb
│
├── output/                        # Pipeline output results - !auto-generated once you run the pipeline
├── predictions/                   # Emotion predictions - !auto-generated once you run the pipeline
├── transcriptions/                # Speech-to-text outputs
├── translations/                  # Translated texts
├── audio_files/                   # Extracted audio
├── temp_audio/                    # Temporary audio processing
│
├── audio_extraction.py            # Audio extraction module
├── speech_to_text.py             # Transcription module
├── translator.py                 # Translation module
├── emotion_classifier.py         # Emotion classification module
├── pipeline.py                   # Main pipeline orchestrator
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
```

---

## Model Performance

### Overall Metrics
- **Accuracy**: 73.6%
- **Weighted F1-Score**: 0.734
- **Error Rate**: 26.4%

### Per-Class Performance

| Emotion    | F1-Score | Support | Performance |
|------------|----------|---------|-------------|
| Neutral    | 0.789    | 391     | ✅ Strong   |
| Happiness  | 0.789    | 278     | ✅ Strong   |
| Surprise   | 0.763    | 126     | ✅ Strong   |
| Anger      | 0.725    | 155     | ⚠️ Moderate |
| Sadness    | 0.635    | 129     | ⚠️ Moderate |
| Disgust    | 0.604    | 72      | ❌ Weak     |
| Fear       | 0.429    | 59      | ❌ Weak     |

### Key Limitations
- Fear detection has only 35.6% recall (misses 2 out of 3 instances)
- Happiness frequently confused with Neutral
- Disgust predictions are sensitive to paraphrasing
- Cannot detect sarcasm, irony, or mixed emotions
- English-only (no cross-lingual capability)

**⚠️ Important**: ~27% error rate requires human oversight for high-stakes applications.

For detailed performance analysis, see [model_card.md](model_card.md).

---

### Performance Tips
- Use GPU for faster processing (5-10x speedup)
- Batch similar-length videos together
- Cache predictions for previously processed content
- Use confidence thresholds to flag uncertain predictions

---

## Recommendations for Deployment

Based on model performance analysis:

1. **Confidence Thresholds:**
   - Surprise, Anger: Use standard monitoring (>0.5)
   - Happiness: Manual review recommended (often confused with Neutral)
   - Disgust: Flag predictions <0.6 for verification

2. **Human-in-the-Loop Workflow:**
   - Automated classification for high-confidence predictions
   - Manual review for low-confidence or critical content
   - Collect reviewed cases to improve future training

3. **Not Recommended For:**
   - High-stakes decisions (employment, medical, legal)
   - Real-time critical applications
   - Detection of sarcasm or irony
   - Very short text (<5 words)

---

## Acknowledgments

- Based on microsoft/deberta-base model
- Emotion categories based on Paul Ekman's research (1971)
- Dataset sourced from unscripted entertainment TV show transcripts
