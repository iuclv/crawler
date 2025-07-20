"""Image downloader module for downloading and validating images from URLs."""

import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urlparse
import hashlib
import time
from PIL import Image
import io

from config.settings import settings

logger = logging.getLogger(__name__)

class ImageDownloader:
    """Handles downloading and validation of images from URLs."""
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self.session = session
        self.timeout = aiohttp.ClientTimeout(total=settings.TIMEOUT_SECONDS)
        self.max_retries = settings.MAX_RETRIES
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        
    async def __aenter__(self):
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_file_extension(self, url: str, content_type: str = None) -> str:
        """Extract file extension from URL or content type."""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Try to get extension from URL
        for ext in self.supported_formats:
            if path.endswith(ext):
                return ext
        
        # Try to get extension from content type
        if content_type:
            content_type = content_type.lower()
            if 'jpeg' in content_type or 'jpg' in content_type:
                return '.jpg'
            elif 'png' in content_type:
                return '.png'
            elif 'webp' in content_type:
                return '.webp'
            elif 'gif' in content_type:
                return '.gif'
        
        # Default to .jpg
        return '.jpg'
    
    def _generate_filename(self, url: str, content_type: str = None) -> str:
        """Generate a unique filename based on URL hash."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        extension = self._get_file_extension(url, content_type)
        return f"{url_hash}{extension}"
    
    async def download_image(self, url: str, output_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """
        Download a single image from URL.
        
        Args:
            url: Image URL to download
            output_dir: Directory to save the image (optional)
            
        Returns:
            Dictionary with download result information or None if failed
        """
        if not self.session:
            raise RuntimeError("ImageDownloader must be used as async context manager")
        
        output_dir = output_dir or settings.CACHE_DIRECTORY
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Downloading image from {url} (attempt {attempt + 1})")
                
                async with self.session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        content_type = response.headers.get('content-type', '')
                        
                        # Validate image content
                        validation_result = self.validate_image_content(content)
                        if not validation_result['valid']:
                            logger.warning(f"Invalid image content from {url}: {validation_result['error']}")
                            continue
                        
                        # Generate filename and save
                        filename = self._generate_filename(url, content_type)
                        file_path = output_dir / filename
                        
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        
                        logger.info(f"Successfully downloaded image: {filename}")
                        
                        return {
                            'url': url,
                            'file_path': str(file_path),
                            'filename': filename,
                            'size_bytes': len(content),
                            'content_type': content_type,
                            'dimensions': validation_result.get('dimensions'),
                            'format': validation_result.get('format'),
                            'download_time': time.time()
                        }
                    else:
                        logger.warning(f"HTTP {response.status} for {url}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout downloading {url} (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Error downloading {url} (attempt {attempt + 1}): {str(e)}")
            
            if attempt < self.max_retries - 1:
                # Exponential backoff
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
        
        logger.error(f"Failed to download image after {self.max_retries} attempts: {url}")
        return None
    
    def validate_image_content(self, content: bytes) -> Dict[str, Any]:
        """
        Validate image content and extract metadata.
        
        Args:
            content: Raw image bytes
            
        Returns:
            Dictionary with validation result and metadata
        """
        try:
            # Try to open with PIL
            image = Image.open(io.BytesIO(content))
            
            # Basic validation
            if image.size[0] < 50 or image.size[1] < 50:
                return {
                    'valid': False,
                    'error': 'Image too small (minimum 50x50 pixels)'
                }
            
            if len(content) > 50 * 1024 * 1024:  # 50MB limit
                return {
                    'valid': False,
                    'error': 'Image too large (maximum 50MB)'
                }
            
            return {
                'valid': True,
                'dimensions': image.size,
                'format': image.format,
                'mode': image.mode
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Invalid image format: {str(e)}'
            }
    
    def validate_image_file(self, file_path: Path) -> bool:
        """
        Validate an image file on disk.
        
        Args:
            file_path: Path to image file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not file_path.exists():
                return False
            
            with open(file_path, 'rb') as f:
                content = f.read()
            
            result = self.validate_image_content(content)
            return result['valid']
            
        except Exception as e:
            logger.error(f"Error validating image file {file_path}: {str(e)}")
            return False
    
    async def batch_download(self, urls: List[str], output_dir: Optional[Path] = None, 
                           concurrent_limit: Optional[int] = None) -> List[Optional[Dict[str, Any]]]:
        """
        Download multiple images concurrently.
        
        Args:
            urls: List of image URLs to download
            output_dir: Directory to save images
            concurrent_limit: Maximum concurrent downloads
            
        Returns:
            List of download results (same order as input URLs)
        """
        concurrent_limit = concurrent_limit or settings.CONCURRENT_DOWNLOADS
        output_dir = output_dir or settings.CACHE_DIRECTORY
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(concurrent_limit)
        
        async def download_with_semaphore(url: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.download_image(url, output_dir)
        
        logger.info(f"Starting batch download of {len(urls)} images with {concurrent_limit} concurrent downloads")
        
        # Execute downloads concurrently
        tasks = [download_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception downloading {urls[i]}: {str(result)}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        successful_downloads = sum(1 for r in processed_results if r is not None)
        logger.info(f"Batch download completed: {successful_downloads}/{len(urls)} successful")
        
        return processed_results
    
    def resize_image(self, image_path: Path, target_size: Tuple[int, int], 
                    output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Resize an image to target dimensions.
        
        Args:
            image_path: Path to source image
            target_size: Target (width, height)
            output_path: Output path (optional, defaults to overwriting source)
            
        Returns:
            Path to resized image or None if failed
        """
        try:
            with Image.open(image_path) as image:
                # Resize with maintaining aspect ratio
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                output_path = output_path or image_path
                image.save(output_path, optimize=True, quality=85)
                
                logger.debug(f"Resized image {image_path} to {image.size}")
                return output_path
                
        except Exception as e:
            logger.error(f"Error resizing image {image_path}: {str(e)}")
            return None
