"""Tests for the image downloader module."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import aiohttp

from src.image_downloader import ImageDownloader

class TestImageDownloader:
    """Test cases for ImageDownloader class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def downloader(self):
        """Create an ImageDownloader instance."""
        return ImageDownloader()
    
    def test_get_file_extension_from_url(self, downloader):
        """Test file extension extraction from URL."""
        assert downloader._get_file_extension("https://example.com/image.jpg") == ".jpg"
        assert downloader._get_file_extension("https://example.com/image.png") == ".png"
        assert downloader._get_file_extension("https://example.com/image.webp") == ".webp"
        assert downloader._get_file_extension("https://example.com/image.gif") == ".gif"
        assert downloader._get_file_extension("https://example.com/image") == ".jpg"  # default
    
    def test_get_file_extension_from_content_type(self, downloader):
        """Test file extension extraction from content type."""
        assert downloader._get_file_extension("https://example.com/image", "image/jpeg") == ".jpg"
        assert downloader._get_file_extension("https://example.com/image", "image/png") == ".png"
        assert downloader._get_file_extension("https://example.com/image", "image/webp") == ".webp"
        assert downloader._get_file_extension("https://example.com/image", "image/gif") == ".gif"
    
    def test_generate_filename(self, downloader):
        """Test filename generation."""
        url = "https://example.com/image.jpg"
        filename = downloader._generate_filename(url)
        
        assert filename.endswith(".jpg")
        assert len(filename) == 36  # 32 char hash + 4 char extension
        
        # Same URL should generate same filename
        filename2 = downloader._generate_filename(url)
        assert filename == filename2
    
    def test_validate_image_content_valid(self, downloader):
        """Test validation of valid image content."""
        # Create a simple 1x1 PNG image
        import io
        from PIL import Image
        
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        content = img_bytes.getvalue()
        
        result = downloader.validate_image_content(content)
        
        assert result['valid'] is True
        assert result['dimensions'] == (100, 100)
        assert result['format'] == 'PNG'
    
    def test_validate_image_content_too_small(self, downloader):
        """Test validation of image that's too small."""
        import io
        from PIL import Image
        
        img = Image.new('RGB', (10, 10), color='red')  # Too small
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        content = img_bytes.getvalue()
        
        result = downloader.validate_image_content(content)
        
        assert result['valid'] is False
        assert 'too small' in result['error']
    
    def test_validate_image_content_invalid(self, downloader):
        """Test validation of invalid image content."""
        content = b"This is not an image"
        
        result = downloader.validate_image_content(content)
        
        assert result['valid'] is False
        assert 'Invalid image format' in result['error']
    
    @pytest.mark.asyncio
    async def test_download_image_success(self, downloader, temp_dir):
        """Test successful image download."""
        url = "https://example.com/image.jpg"
        
        # Mock image content
        import io
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        mock_content = img_bytes.getvalue()
        
        # Mock aiohttp response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.read = AsyncMock(return_value=mock_content)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        downloader.session = mock_session
        
        result = await downloader.download_image(url, temp_dir)
        
        assert result is not None
        assert result['url'] == url
        assert result['size_bytes'] == len(mock_content)
        assert result['content_type'] == 'image/jpeg'
        assert Path(result['file_path']).exists()
    
    @pytest.mark.asyncio
    async def test_download_image_http_error(self, downloader, temp_dir):
        """Test download with HTTP error."""
        url = "https://example.com/image.jpg"
        
        # Mock aiohttp response with error
        mock_response = AsyncMock()
        mock_response.status = 404
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        downloader.session = mock_session
        
        result = await downloader.download_image(url, temp_dir)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_batch_download(self, downloader, temp_dir):
        """Test batch download functionality."""
        urls = [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            "https://example.com/image3.jpg"
        ]
        
        # Mock successful downloads
        import io
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        mock_content = img_bytes.getvalue()
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_response.read = AsyncMock(return_value=mock_content)
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        async with ImageDownloader(mock_session) as downloader:
            results = await downloader.batch_download(urls, temp_dir, concurrent_limit=2)
        
        assert len(results) == 3
        assert all(result is not None for result in results)
        assert all(result['url'] in urls for result in results)
    
    def test_validate_image_file_exists(self, downloader, temp_dir):
        """Test validation of existing image file."""
        # Create a test image file
        import io
        from PIL import Image
        
        img = Image.new('RGB', (100, 100), color='red')
        test_file = temp_dir / "test_image.jpg"
        img.save(test_file, format='JPEG')
        
        result = downloader.validate_image_file(test_file)
        assert result is True
    
    def test_validate_image_file_not_exists(self, downloader, temp_dir):
        """Test validation of non-existent image file."""
        non_existent_file = temp_dir / "non_existent.jpg"
        
        result = downloader.validate_image_file(non_existent_file)
        assert result is False
    
    def test_resize_image(self, downloader, temp_dir):
        """Test image resizing functionality."""
        # Create a test image
        from PIL import Image
        
        img = Image.new('RGB', (200, 200), color='red')
        test_file = temp_dir / "test_image.jpg"
        img.save(test_file, format='JPEG')
        
        # Resize image
        result_path = downloader.resize_image(test_file, (100, 100))
        
        assert result_path == test_file
        
        # Check if image was resized
        with Image.open(test_file) as resized_img:
            assert resized_img.size[0] <= 100
            assert resized_img.size[1] <= 100
