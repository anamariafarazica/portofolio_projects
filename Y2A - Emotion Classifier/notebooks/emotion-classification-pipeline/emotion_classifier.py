"""
Emotion Classification Module
Classifies emotions from text using fine-tuned DeBERTa model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionClassifier:
    """
    Handles emotion classification using fine-tuned DeBERTa model.
    """
    
    def __init__(self, model_path="./deberta-finetuned", device=None):
        """
        Initialize the Emotion Classifier.
        
        Args:
            model_path (str): Path to fine-tuned DeBERTa model
            device (str): Device to use ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Emotion label mapping (update these with your actual emotion names!)
        # Based on your training data, these should be the 7 emotions
        self.id2label = {
            0: "anger",      # Update these with your actual labels
            1: "disgust",
            2: "fear",
            3: "happiness",
            4: "neutral",
            5: "sadness",
            6: "surprise"
        }
        
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Max length from training
        self.max_length = 128
        
        # Convert to absolute path
        model_path = os.path.abspath(model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model directory not found at: {model_path}\n"
                f"Please ensure your fine-tuned model is saved at this location."
            )
        
        # Load model and tokenizer
        try:
            logger.info(f"Loading model from: {model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            
            self.model.eval()  # Set to evaluation mode
            logger.info("Emotion classification model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load emotion model: {str(e)}")
    
    
    def predict(self, text):
        """
        Predict emotion for a single text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Predicted emotion label
        """
        if not text or text.strip() == "":
            logger.warning("Empty text provided, returning neutral")
            return "neutral"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_label_id = torch.argmax(logits, dim=-1).item()
            
            # Convert to emotion label
            emotion = self.id2label[predicted_label_id]
            
            return emotion
            
        except Exception as e:
            logger.error(f"Prediction failed for text: {text[:50]}... Error: {str(e)}")
            return "neutral"  # Default fallback
    
    
    def predict_with_confidence(self, text):
        """
        Predict emotion with confidence scores.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: {
                'emotion': predicted emotion label,
                'confidence': confidence score (0-1),
                'all_scores': dict of all emotion scores
            }
        """
        if not text or text.strip() == "":
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_scores': {}
            }
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence, predicted_label_id = torch.max(probs, dim=-1)
                
                predicted_label_id = predicted_label_id.item()
                confidence = confidence.item()
                
                # Get all scores
                all_probs = probs[0].cpu().numpy()
                all_scores = {
                    self.id2label[i]: float(prob) 
                    for i, prob in enumerate(all_probs)
                }
            
            emotion = self.id2label[predicted_label_id]
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'all_scores': all_scores
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'emotion': 'neutral',
                'confidence': 0.0,
                'all_scores': {}
            }
    
    
    def predict_batch(self, texts, batch_size=32):
        """
        Predict emotions for multiple texts efficiently.
        
        Args:
            texts (list): List of input texts
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of predicted emotion labels
        """
        logger.info(f"Predicting emotions for {len(texts)} texts")
        
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    predicted_ids = torch.argmax(logits, dim=-1).cpu().numpy()
                
                # Convert to labels
                batch_emotions = [self.id2label[pred_id] for pred_id in predicted_ids]
                predictions.extend(batch_emotions)
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch)}/{len(texts)} texts")
                
            except Exception as e:
                logger.error(f"Batch prediction failed at index {i}: {str(e)}")
                # Return neutral for failed batch
                predictions.extend(["neutral"] * len(batch))
        
        logger.info("Emotion prediction completed")
        return predictions
    
    
    def save_predictions_to_csv(self, texts, predictions, output_path="predictions/emotion_predictions.csv"):
        """
        Save emotion predictions to CSV for testing/inspection.
        
        Args:
            texts (list): List of input texts
            predictions (list): List of predicted emotions
            output_path (str): Path where to save the CSV
            
        Returns:
            str: Path to saved CSV file
        """
        import pandas as pd
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'Text': texts,
            'Predicted Emotion': predictions
        })
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Predictions saved to: {output_path}")
        
        return output_path
