# Model Card: Emotion Classification Model

## Model Overview
### Architecture
DeBERTa-base fine-tuned for emotion classification with 7 emotion categories.

    - Base: DeBERTa-base (125M parameters, 12 layers, 768 hidden dimensions)
    - Classification Head: Dense layer + output projection for 7-class emotion prediction
    - Input: Text sequences up to 128 tokens
    - Output: Probability distribution over 7 emotion classes

### Key Design Choices
    - Learning rate: 2e-5 with 500 warmup steps
    - Batch size: 16 (train), 32 (eval)
    - 8 epochs with early stopping based on F1-score
    - Weight decay (0.01) and gradient clipping (max norm 1.0)
    - Mixed precision (FP16) for efficient training

### Purpose
    This model classifies emotions in text transcripts. It takes transcript text as input and predicts one of 7 emotions: happiness, sadness, fear, disgust, neutral, anger, or surprise.

    The model enables automated emotion analysis for media companies analyzing video transcripts, customer feedback, or other text-based content at scale.

### Development Context
#### Key Assumptions:
    - Emotions can be reliably inferred from text alone (without audio/visual cues)
    - 7 emotion categories adequately capture the emotional range in target content
    - Pre-trained DeBERTa knowledge transfers effectively to emotion tasks

#### Constraints:
    - Limited to English language text
    - Training data constrained by available labeled emotion datasets
    - Class imbalance across emotion categories
    - GPU resources required for practical training time

#### Development Conditions:
    - Trained on GPU with CUDA support (~9 minutes total)
    - Framework: HuggingFace Transformers with PyTorch


## Intended Use
### Intended Applications
This application have been built & alligned for use for the Content Intelligence Agency, which is involved in building software for media makers to analyse their content. The final form of this model is to be integrated in a full-built pipeline that takes a video as input, transcribes the video, translates the text to English and tag the emotions expressed per sentence.
Automated emotion detection in English text transcripts, including:

    -> Video transcript analysis: Identifying emotional moments in interviews, podcasts, or video content
    -> Social media monitoring: Analyzing emotions in user comments, reviews, and feedback
    -> Content classification: Organizing media by emotional themes
    -> Audience research: Understanding emotional patterns in viewer responses

**Example**: A media company analyzes 1,000 customer call transcripts to identify frustration vs. satisfaction, or content creators review podcasts to find emotionally engaging segments.

### Not Intended Use:
! Do NOT use this model for:

    - High-stakes decisions: Employment, medical, legal, or any context with serious consequences
    - Non-English text: Trained only on English; unreliable for other languages
    - Multimodal contexts: Cannot incorporate tone, facial expressions, or body language
    - Very short text: Performance degrades on fragments < 5 words
    - Sarcasm/irony: May misclassify when literal meaning differs from intent

**Key limitations**: 73% accuracy means ~1 in 4 predictions may be wrong; cannot detect mixed emotions; struggles with rare emotion classes (disgust, fear).

### Value for Media Analysis
Enables media analysis at scale by:

    -> Automating emotion labeling
    -> Processing thousands of transcripts in minutes vs. days
    -> Allowing human reviewers to focus on nuanced cases flagged by the model

## Dataset Details
### Data Sources and overview
Training data: 23,696 labeled English text samples from unscripted entertainment TV show transcripts, sourced from the company pipeline.
Evaluation data: 1,210 labeled English text samples from unscripted TV Show "Kitchen Nightmares", sourced from the company pipeline

The dataset consists of short English sentences or phrases labeled with one of 7 emotions, alligned with the six core emotions studied by Paul Ekman in 1971 (happiness, sadness, fear, disgust, anger, surprise), and neutral.

### Preprocessing steps
1. Data cleaning:
Removed missing values and NaN rows
Dropped duplicate entries
Lowercased all sentences for consistency

2. Tokenization: DeBERTa tokenizer with:
Maximum sequence length: 128 tokens
Truncation for longer sequences
Padding to uniform length

3. Label encoding: Emotion labels converted to integer classes (0-6)

### Class Distribution
![Test_Data](test%20data%20distribution.png)

    -> Class imbalance: Happiness and neutral emotions are over-represented, while disgust and sadness are under-represented. This affects model performance, with better accuracy on majority classes.

### Language and Cultural Representativeness
#### Language scope: English only
1. Cultural limitations:
    - Dataset reflects Western/English-speaking cultural expressions of emotion
    - Emotion expressions vary significantly across cultures (e.g., directness of anger, display rules for sadness)
    - Model may not generalize to non-Western emotional expressions or culturally-specific language patterns
    - Idioms, metaphors, and emotion descriptors may be culture-specific

2. Multilingual challenges:
    - No language diversity: Model trained exclusively on English text
    - Zero cross-lingual capability: Cannot process or accurately classify emotions in other languages
    - Translation limitations: Translated text may lose emotional nuance or context

## Performance Metrics & Evaluation
### Overall Performance:

- **Accuracy**: 73.6%
- **F1-Score**: 0.734
- **Error Rate**: 26.4% (320 misclassifications out of 1,210 test examples)

| Emotion | F1-Score | Test Examples | Performance Tier |
|---------|----------|---------------|------------------|
| Neutral | 0.789 | 391 (32.3%) | Strong |
| Happiness | 0.789 | 278 (23.0%) | Strong |
| Surprise | 0.763 | 126 (10.4%) | Strong |
| Anger | 0.725 | 155 (12.8%) | Moderate |
| Sadness | 0.635 | 129 (10.7%) | Moderate |
| Disgust | 0.604 | 72 (5.9%) | Weak |
| Fear | 0.429 | 59 (4.9%) | Weak |

![Error_Analysis](confusion_matrix.png)

### Key Limitations:

    - Fear detection has only 35.6% recall, missing almost 2 out of 3 fear instances
    - Strong class imbalance effects: rare emotions (fear, disgust) significantly underperform
    - Medium-length sentences (6-10 words) show lowest accuracy at 64% compared to 79% for short sentences (0-5 words)

Full Error Analysis documentation [here](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-11-1/blob/a4d7d13affa813fd7eed960d26f5191e50f620a3/Task%209/Group%2011%20-%20Error%20Analysis.pdf).

## Explainability and Transparency
- Token-level Analysis: The model correctly identifies emotional keywords in 72% of sentences (e.g., "trouble" for fear, "disgusting" for disgust). However, special tokens ([CLS], [SEP]) often dominate attribution scores, limiting interpretation clarity.

- Emotion-Specific Performance:
    - Strong: Surprise, sadness, and disgust consistently highlight correct emotional words
    - Weak: Happiness predictions frequently misclassified as neutral, focusing on irrelevant proper nouns instead of emotional terms

- Key Finding: 55.6% of predictions depend heavily on just 1-3 tokens. Disgust is particularly fragile—"That's disgusting" loses 33% confidence per token removed

**Example:** For the surprise sentence "I never thought I'd see Pete change," the model correctly assigned highest relevance to "never" (16.50%), "change" (15.65%), and "see" (12.66%), demonstrating strong alignment with human-identified emotional content.

Full XAI analysis [here](https://github.com/BredaUniversityADSAI/fae2-nlpr-group-group-11-1/blob/59e0205b180e3005e290b95328b00ab6f9d1440e/Task%2010/Group%2011%20-%20Explainable%20AI%20for%20Emotion%20Classification.pdf)

## Recommendation for Use
For optimal deployment, implement emotion-specific confidence thresholds to manage operational risks. Predictions for surprise, fear, and anger are  production-ready and can be used with standard monitoring. However, happiness predictions should be manually reviewed due to high misclassification rates   neutral, and disgust predictions with confidence below 60% should be flagged for human verification due to keyword-dependency. Use a phased workflow: run automated classification, flag low-confidence predictions for review, use high-confidence results for immediate tagging and search indexing, and collect reviewed cases to improve future training.

### Key Operational Risks:

- ~27% error rate requires human oversight for sensitive applications
- Disgust predictions are brittle and sensitive to paraphrasing
- Cannot detect sarcasm, irony, or cultural nuances
- Happiness frequently confused with neutral emotion

## Sustainibility Considerations

### Environmental Impact:

- Training: Fine-tuned pre-trained twitter-deberta-base model (125M parameters) on 23,689 examples using NVIDIA RTX 4050
- Estimated Energy: 0.5-2 kWh for fine-tuning; ~0.2-0.8 kg CO₂eq emissions (varies by grid)
- Hardware: NVIDIA RTX 4050 (TDP: 140W) - energy-efficient mobile GPU
- Inference: Lightweight enough for CPU deployment; batch processing recommended for efficiency

### Sustainability Strategies:

- Transfer learning reduced training cost by ~99% compared to training from scratch
- Used base model rather than larger variants, balancing performance with efficiency
- Enables automated analysis of thousands of hours, replacing energy-intensive manual annotation

### Recommendations:

- Deploy on renewable energy cloud infrastructure
- Use batch inference to maximize hardware utilization
- Cache predictions for previously analyzed content
