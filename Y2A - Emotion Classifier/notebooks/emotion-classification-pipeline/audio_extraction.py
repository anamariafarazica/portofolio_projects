"""
Audio Extraction Module
Handles extracting/downloading audio from various sources:
- YouTube URLs
- Local video files (MP4, AVI, etc.)
- Returns path to audio file (MP3)
"""

import os
from pytubefix import YouTube
from moviepy.editor import VideoFileClip
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioExtractor:
    """
    Extracts audio from various sources (YouTube, local video files).
    """
    
    def __init__(self, output_dir="audio_files"):
        """
        Initialize AudioExtractor.
        
        Args:
            output_dir (str): Directory to save extracted audio files
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
    
    
    def is_youtube_url(self, input_path):
        """
        Check if input is a YouTube URL.
        
        Args:
            input_path (str): Input string to check
            
        Returns:
            bool: True if it's a YouTube URL
        """
        youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com']
        return any(domain in input_path.lower() for domain in youtube_domains)
    
    
    def is_audio_file(self, input_path):
        """
        Check if input is already an audio file.
        
        Args:
            input_path (str): Path to check
            
        Returns:
            bool: True if it's an audio file (mp3, wav, etc.)
        """
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.aac']
        return any(input_path.lower().endswith(ext) for ext in audio_extensions)
    
    
    def is_video_file(self, input_path):
        """
        Check if input is a video file.
        
        Args:
            input_path (str): Path to check
            
        Returns:
            bool: True if it's a video file
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        return any(input_path.lower().endswith(ext) for ext in video_extensions)
    
    
    def extract_from_youtube(self, youtube_url, output_filename=None):
        """
        Download audio from YouTube URL.
        
        Args:
            youtube_url (str): YouTube video URL
            output_filename (str): Optional custom filename (without extension)
            
        Returns:
            str: Path to downloaded audio file (.mp3)
            
        Raises:
            RuntimeError: If download fails
        """
        try:
            logger.info(f"Downloading audio from YouTube: {youtube_url}")
            
            # Create YouTube object
            yt = YouTube(youtube_url)
            
            # Extract only audio
            audio_stream = yt.streams.filter(only_audio=True).first()
            
            if not audio_stream:
                raise RuntimeError("No audio stream found for this video")
            
            # Download the file
            out_file = audio_stream.download(output_path=self.output_dir)
            
            # Convert to .mp3
            base, ext = os.path.splitext(out_file)
            
            # Use custom filename if provided, otherwise use original
            if output_filename:
                new_file = os.path.join(self.output_dir, f"{output_filename}.mp3")
            else:
                new_file = base + '.mp3'
            
            os.rename(out_file, new_file)
            
            logger.info(f"Audio downloaded successfully: {new_file}")
            return new_file
            
        except Exception as e:
            raise RuntimeError(f"Failed to download from YouTube: {str(e)}")
    
    
    def extract_from_video(self, video_path, output_filename=None):
        """
        Extract audio from local video file.
        
        Args:
            video_path (str): Path to video file
            output_filename (str): Optional custom filename (without extension)
            
        Returns:
            str: Path to extracted audio file (.mp3)
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            RuntimeError: If extraction fails
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            logger.info(f"Extracting audio from video: {video_path}")
            
            # Determine output filename
            if output_filename:
                audio_path = os.path.join(self.output_dir, f"{output_filename}.mp3")
            else:
                # Use original filename with .mp3 extension
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                audio_path = os.path.join(self.output_dir, f"{base_name}.mp3")
            
            # Extract audio using moviepy
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()
            
            logger.info(f"Audio extracted successfully: {audio_path}")
            return audio_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract audio from video: {str(e)}")
    
    
    def get_audio(self, input_source, output_filename=None):
        """
        Universal method to get audio from any source.
        Automatically detects the input type and processes accordingly.
        
        Args:
            input_source (str): Can be:
                - YouTube URL
                - Path to video file
                - Path to audio file (returns as-is)
            output_filename (str): Optional custom output filename
            
        Returns:
            str: Path to audio file (.mp3)
            
        Raises:
            ValueError: If input type cannot be determined
            FileNotFoundError: If local file doesn't exist
        """
        logger.info(f"Processing input: {input_source}")
        
        # Case 1: YouTube URL
        if self.is_youtube_url(input_source):
            return self.extract_from_youtube(input_source, output_filename)
        
        # Case 2: Already an audio file - return path as-is
        elif self.is_audio_file(input_source):
            if not os.path.exists(input_source):
                raise FileNotFoundError(f"Audio file not found: {input_source}")
            logger.info(f"Input is already an audio file: {input_source}")
            return input_source
        
        # Case 3: Video file - extract audio
        elif self.is_video_file(input_source):
            return self.extract_from_video(input_source, output_filename)
        
        # Case 4: Unknown format
        else:
            raise ValueError(
                f"Unknown input format: {input_source}\n"
                "Please provide a YouTube URL, video file, or audio file."
            )
