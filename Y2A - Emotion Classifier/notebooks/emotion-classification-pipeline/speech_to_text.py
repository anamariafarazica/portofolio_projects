"""
Speech-to-Text Module
Handles transcription using AssemblyAI and sentence segmentation.
Returns sentences with timestamps in the required format.
"""

import assemblyai as aai
import nltk
import pandas as pd
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechToText:
    """
    Handles speech-to-text conversion with sentence-level segmentation.
    """
    
    def __init__(self, api_key):
        """
        Initialize the Speech-to-Text module.
        
        Args:
            api_key (str): AssemblyAI API key
        """
        aai.settings.api_key = api_key
        self.config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.universal
        )
        self.transcriber = aai.Transcriber(config=self.config)
        
        # Download NLTK data if needed (for sentence tokenization)
        self._setup_nltk()
        
        logger.info("SpeechToText module initialized")
    
    
    def _setup_nltk(self):
        """
        Download required NLTK data for sentence tokenization.
        """
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("Downloading NLTK tokenizers...")
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
    
    
    def transcribe(self, audio_path):
        """
        Transcribe audio file and return sentences with timestamps.
        
        Args:
            audio_path (str): Path to audio file (.mp3, .wav, etc.)
            
        Returns:
            list: List of dictionaries with format:
                  [
                      {
                          'start_time': '00:00:00,000',
                          'end_time': '00:00:02,000',
                          'text': 'Hello world.'
                      },
                      ...
                  ]
                  
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If transcription fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            logger.info(f"Starting transcription of: {audio_path}")
            
            # Run AssemblyAI transcription
            transcript = self.transcriber.transcribe(audio_path)
            
            # Check for errors
            if transcript.status == "error":
                raise RuntimeError(f"Transcription failed: {transcript.error}")
            
            logger.info("Transcription completed successfully")
            
            # Extract sentences with timestamps
            segments = self._create_segments_from_transcript(transcript)
            
            logger.info(f"Created {len(segments)} sentence segments")
            return segments
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise RuntimeError(f"Transcription error: {str(e)}")
    
    
    def save_segments_to_csv(self, segments, output_path="transcriptions/transcription.csv"):
        """
        Save transcribed segments to CSV file for testing/inspection.
        
        Args:
            segments (list): List of segment dictionaries
            output_path (str): Path where to save the CSV
            
        Returns:
            str: Path to saved CSV file
        """
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Convert segments to DataFrame
        df = pd.DataFrame(segments)
        
        # Rename columns to match your requirements
        df = df.rename(columns={
            'start_time': 'Start Time',
            'end_time': 'End Time',
            'text': 'Sentence'
        })
        
        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Segments saved to: {output_path}")
        
        return output_path
    
    
    def _create_segments_from_transcript(self, transcript):
        """
        Create sentence segments with timestamps from AssemblyAI transcript.
        
        Uses AssemblyAI's word-level timestamps and NLTK sentence tokenization
        to create accurate sentence boundaries with timing information.
        
        Args:
            transcript: AssemblyAI transcript object
            
        Returns:
            list: List of segment dictionaries with timestamps
        """
        segments = []
        
        # Get the full transcript text
        full_text = transcript.text
        
        # Split into sentences using NLTK
        sentences = nltk.sent_tokenize(full_text)
        
        # If we have word-level timestamps, use them for accurate timing
        if hasattr(transcript, 'words') and transcript.words:
            segments = self._align_sentences_with_words(sentences, transcript.words)
        else:
            # Fallback: use utterances if available
            if hasattr(transcript, 'utterances') and transcript.utterances:
                logger.warning("Using utterances as fallback (no word-level timestamps)")
                for utterance in transcript.utterances:
                    segment = {
                        'start_time': self._milliseconds_to_timestamp(utterance.start),
                        'end_time': self._milliseconds_to_timestamp(utterance.end),
                        'text': utterance.text
                    }
                    segments.append(segment)
            else:
                # Last resort: return full transcript without precise timestamps
                logger.warning("No word-level data available, using full transcript")
                segments = [{
                    'start_time': '00:00:00,000',
                    'end_time': '00:00:00,000',
                    'text': full_text
                }]
        
        return segments
    
    
    def _align_sentences_with_words(self, sentences, words):
        """
        Align NLTK sentences with AssemblyAI word timestamps.
        
        Args:
            sentences (list): List of sentences from NLTK
            words (list): List of word objects from AssemblyAI with timestamps
            
        Returns:
            list: List of segments with accurate timestamps
        """
        segments = []
        word_index = 0
        
        for sentence in sentences:
            # Clean the sentence for comparison
            sentence_words = sentence.split()
            num_words = len(sentence_words)
            
            if word_index >= len(words):
                logger.warning(f"Ran out of words at sentence: {sentence[:50]}...")
                break
            
            # Get start time from first word
            start_time = words[word_index].start
            
            # Get end time from last word of this sentence
            end_word_index = min(word_index + num_words - 1, len(words) - 1)
            end_time = words[end_word_index].end
            
            segment = {
                'start_time': self._milliseconds_to_timestamp(start_time),
                'end_time': self._milliseconds_to_timestamp(end_time),
                'text': sentence.strip()
            }
            segments.append(segment)
            
            # Move to next sentence's starting word
            word_index += num_words
        
        return segments
    
    
    def _milliseconds_to_timestamp(self, milliseconds):
        """
        Convert milliseconds to timestamp format HH:MM:SS,mmm
        (Required format for your final CSV output)
        
        Args:
            milliseconds (int): Time in milliseconds
            
        Returns:
            str: Formatted timestamp (e.g., '00:00:02,500')
        """
        hours = int(milliseconds // 3600000)
        minutes = int((milliseconds % 3600000) // 60000)
        seconds = int((milliseconds % 60000) // 1000)
        ms = int(milliseconds % 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"
