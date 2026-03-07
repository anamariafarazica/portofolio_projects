"""
Complete Emotion Classification Pipeline
Processes video input through transcription, translation, and emotion classification.

Usage:
    python pipeline.py --input <video_path_or_url> --output <output_csv_path>
"""

import argparse
import logging
import os
from datetime import datetime

from audio_extraction import AudioExtractor
from speech_to_text import SpeechToText
from translator import Translator
from emotion_classifier import EmotionClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmotionPipeline:
    """
    Complete pipeline for emotion classification from video.
    """
    
    def __init__(self, 
                 assemblyai_api_key,
                 emotion_model_path="deberta-finetuned/checkpoint-14810",
                 temp_audio_dir="temp_audio"):
        """
        Initialize the pipeline with all components.
        
        Args:
            assemblyai_api_key (str): AssemblyAI API key for transcription
            emotion_model_path (str): Path to fine-tuned emotion model
            temp_audio_dir (str): Directory for temporary audio files
        """
        logger.info("Initializing Emotion Classification Pipeline...")
        
        # Create temp directory if it doesn't exist
        self.temp_audio_dir = temp_audio_dir
        if not os.path.exists(temp_audio_dir):
            os.makedirs(temp_audio_dir)
            logger.info(f"Created temporary audio directory: {temp_audio_dir}")
        
        # Initialize all components
        try:
            logger.info("Loading Audio Extractor...")
            self.audio_extractor = AudioExtractor(output_dir=temp_audio_dir)
            
            logger.info("Loading Speech-to-Text module...")
            self.speech_to_text = SpeechToText(api_key=assemblyai_api_key)
            
            logger.info("Loading Translator...")
            self.translator = Translator()
            
            logger.info("Loading Emotion Classifier...")
            self.emotion_classifier = EmotionClassifier(model_path=emotion_model_path)
            
            logger.info("✓ All modules loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise
    
    
    def process_video(self, input_source, output_csv_path, cleanup=True):
        """
        Process a video through the complete pipeline.
        
        Args:
            input_source (str): Path to video file or YouTube URL
            output_csv_path (str): Path where to save the output CSV
            cleanup (bool): Whether to delete temporary audio files
            
        Returns:
            str: Path to the output CSV file
        """
        logger.info("=" * 70)
        logger.info("STARTING EMOTION CLASSIFICATION PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Input: {input_source}")
        logger.info(f"Output: {output_csv_path}")
        logger.info("")
        
        try:
            # Step 1: Extract/Get Audio
            logger.info("[STEP 1/5] Extracting audio from video...")
            audio_path = self.audio_extractor.get_audio(input_source)
            logger.info(f"✓ Audio extracted: {audio_path}")
            logger.info("")
            
            # Step 2: Transcribe Audio
            logger.info("[STEP 2/5] Transcribing audio to text...")
            segments = self.speech_to_text.transcribe(audio_path)
            logger.info(f"✓ Transcription complete: {len(segments)} segments")
            logger.info("")
            
            # Step 3: Round-Trip Translation (EN → NL → EN)
            logger.info("[STEP 3/5] Performing round-trip translation...")
            texts_to_translate = [seg['text'] for seg in segments]
            translation_results = self.translator.round_trip_translate_batch(
                texts_to_translate,
                batch_size=16
            )
            logger.info(f"✓ Translation complete: {len(translation_results)} texts")
            logger.info("")
            
            # Step 4: Classify Emotions
            logger.info("[STEP 4/5] Classifying emotions...")
            back_translated_texts = [res['back_translated'] for res in translation_results]
            emotions = self.emotion_classifier.predict_batch(
                back_translated_texts,
                batch_size=32
            )
            logger.info(f"✓ Emotion classification complete: {len(emotions)} predictions")
            logger.info("")
            
            # Step 5: Combine Results and Save to CSV
            logger.info("[STEP 5/5] Creating output CSV...")
            self._save_results_to_csv(
                segments=segments,
                translation_results=translation_results,
                emotions=emotions,
                output_path=output_csv_path
            )
            logger.info(f"✓ Results saved to: {output_csv_path}")
            logger.info("")
            
            # Cleanup temporary files
            if cleanup and os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"✓ Cleaned up temporary audio file")
            
            logger.info("=" * 70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 70)
            
            return output_csv_path
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    
    def _save_results_to_csv(self, segments, translation_results, emotions, output_path):
        """
        Combine all results and save to CSV in the required format.
        
        Args:
            segments (list): Transcription segments with timestamps
            translation_results (list): Translation results
            emotions (list): Emotion predictions
            output_path (str): Path to save CSV
        """
        import pandas as pd
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Combine all data
        results = []
        for segment, translation, emotion in zip(segments, translation_results, emotions):
            results.append({
                'Start Time': segment['start_time'],
                'End Time': segment['end_time'],
                'Sentence': segment['text'],
                'Translation': translation['back_translated'],
                'Emotion': emotion
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Created CSV with {len(df)} rows")


def main():
    """
    Main function to run the pipeline from command line.
    """
    parser = argparse.ArgumentParser(
        description='Emotion Classification Pipeline - Process videos for emotion analysis'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input video file path or YouTube URL'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default="6ae9bf56e3f94a828775fed718ee69df",
        help='AssemblyAI API key (default: from code)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default="deberta-finetuned/checkpoint-14810",
        help='Path to emotion classification model'
    )
    
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Keep temporary audio files (default: delete them)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = EmotionPipeline(
        assemblyai_api_key=args.api_key,
        emotion_model_path=args.model_path
    )
    
    # Process video
    pipeline.process_video(
        input_source=args.input,
        output_csv_path=args.output,
        cleanup=not args.no_cleanup
    )


# -------------------------------
# Testing Section
# -------------------------------

if __name__ == "__main__":
    """
    Run the pipeline.
    
    Examples:
        # Process a YouTube video
        python pipeline.py --input "https://www.youtube.com/watch?v=3IY0EiM1TPo" --output "results.csv"
        
        # Process a local video file
        python pipeline.py --input "video.mp4" --output "results.csv"
        
        # Process with custom model path
        python pipeline.py --input "video.mp4" --output "results.csv" --model-path "path/to/model"
    """
    
    # Check if running with command line arguments
    import sys
    
    if len(sys.argv) > 1:
        # Run with command line arguments
        main()
    else:
        # Run test example
        print("=" * 70)
        print("PIPELINE TEST MODE")
        print("=" * 70)
        print("\nNo arguments provided. Running test example...")
        print("\nTo use the pipeline, run:")
        print("  python pipeline.py --input <video_url_or_path> --output <output.csv>")
        print("\nExample:")
        print('  python pipeline.py --input "https://www.youtube.com/watch?v=3IY0EiM1TPo" --output "results.csv"')
        print("\n" + "=" * 70)
        
        # Ask if user wants to run a test
        print("\nWould you like to run a test with a sample video? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            # Initialize pipeline with defaults
            pipeline = EmotionPipeline(
                assemblyai_api_key="#########################################",
                emotion_model_path="emotion-classification-pipeline/checkpoint-14810"
            )
            
            # Test with a short video (update this URL or path)
            test_input = "https://www.youtube.com/watch?v=3IY0EiM1TPo"
            test_output = f"output/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            print(f"\nProcessing test video...")
            print(f"Input: {test_input}")
            print(f"Output: {test_output}\n")
            
            try:
                pipeline.process_video(test_input, test_output)
                print("\n✓ Test completed successfully!")
                print(f"✓ Check the results at: {test_output}")
            except Exception as e:
                print(f"\n✗ Test failed: {e}")
        else:
            print("\nExiting. Run with --help for usage information.")