"""
Translation Module
Handles round-trip translation: English → Dutch → English
This is used for data augmentation before emotion classification.
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Translator:
    """
    Handles round-trip translation (EN→NL→EN) using MarianMT models.
    """
    
    def __init__(self, 
                 model_en_nl="Helsinki-NLP/opus-mt-en-nl",
                 model_nl_en="Helsinki-NLP/opus-mt-nl-en",
                 device=None):
        """Initialize the Translator with both direction models."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        try:
            logger.info(f"Loading EN→NL model: {model_en_nl}")
            self.model_en_nl = MarianMTModel.from_pretrained(model_en_nl).to(self.device)
            self.tokenizer_en_nl = MarianTokenizer.from_pretrained(model_en_nl)
            
            logger.info(f"Loading NL→EN model: {model_nl_en}")
            self.model_nl_en = MarianMTModel.from_pretrained(model_nl_en).to(self.device)
            self.tokenizer_nl_en = MarianTokenizer.from_pretrained(model_nl_en)
            
            logger.info("Both translation models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise RuntimeError(f"Failed to load translation models: {str(e)}")
    
    
    def translate_en_to_nl(self, text):
        """Translate single text from English to Dutch."""
        if not text or text.strip() == "":
            return text
        
        try:
            inputs = self.tokenizer_en_nl(
                text, return_tensors="pt", padding=True, 
                truncation=True, max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                translated = self.model_en_nl.generate(**inputs, max_length=128)
            
            return self.tokenizer_en_nl.decode(translated[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"EN→NL translation failed: {str(e)}")
            return text
    
    
    def translate_nl_to_en(self, text):
        """Translate single text from Dutch to English."""
        if not text or text.strip() == "":
            return text
        
        try:
            inputs = self.tokenizer_nl_en(
                text, return_tensors="pt", padding=True,
                truncation=True, max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                translated = self.model_nl_en.generate(**inputs, max_length=128)
            
            return self.tokenizer_nl_en.decode(translated[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"NL→EN translation failed: {str(e)}")
            return text
    
    
    def round_trip_translate(self, text):
        """Perform round-trip translation: EN → NL → EN"""
        dutch = self.translate_en_to_nl(text)
        back_translated = self.translate_nl_to_en(dutch)
        
        return {
            'original': text,
            'dutch': dutch,
            'back_translated': back_translated
        }
    
    
    def round_trip_translate_batch(self, texts, batch_size=16):
        """Perform round-trip translation on multiple texts efficiently."""
        logger.info(f"Round-trip translating {len(texts)} texts")
        
        results = []
        
        # Step 1: EN → NL
        logger.info("Step 1/2: Translating EN → NL")
        dutch_translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            try:
                inputs = self.tokenizer_en_nl(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=128
                ).to(self.device)
                
                with torch.no_grad():
                    translated = self.model_en_nl.generate(**inputs, max_length=128)
                
                batch_translations = self.tokenizer_en_nl.batch_decode(
                    translated, skip_special_tokens=True
                )
                dutch_translations.extend(batch_translations)
                
            except Exception as e:
                logger.error(f"EN→NL batch failed: {str(e)}")
                dutch_translations.extend(batch)
        
        # Step 2: NL → EN
        logger.info("Step 2/2: Translating NL → EN")
        english_translations = []
        
        for i in range(0, len(dutch_translations), batch_size):
            batch = dutch_translations[i:i+batch_size]
            
            try:
                inputs = self.tokenizer_nl_en(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=128
                ).to(self.device)
                
                with torch.no_grad():
                    translated = self.model_nl_en.generate(**inputs, max_length=128)
                
                batch_translations = self.tokenizer_nl_en.batch_decode(
                    translated, skip_special_tokens=True
                )
                english_translations.extend(batch_translations)
                
            except Exception as e:
                logger.error(f"NL→EN batch failed: {str(e)}")
                english_translations.extend(batch)
        
        # Combine results
        for original, dutch, back_translated in zip(texts, dutch_translations, english_translations):
            results.append({
                'original': original,
                'dutch': dutch,
                'back_translated': back_translated
            })
        
        logger.info("Round-trip translation completed")
        return results
    
    
    def save_translations_to_csv(self, results, output_path="translations/round_trip_translations.csv"):
        """
        Save round-trip translation results to CSV for testing/inspection.
        
        Args:
            results (list): List of translation result dictionaries
            output_path (str): Path where to save the CSV
            
        Returns:
            str: Path to saved CSV file
        """
        import pandas as pd
        import os
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Rename columns to be clearer
        df = df.rename(columns={
            'original': 'Original English',
            'dutch': 'Dutch Translation',
            'back_translated': 'Back-Translated English'
        })
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Translations saved to: {output_path}")
        
        return output_path