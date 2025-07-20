"""Tests for the main crawler functionality."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.main import ImageCrawler

class TestImageCrawler:
    """Test cases for ImageCrawler class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return [
            {
                "name": "Test Category 1",
                "image_urls": [
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg"
                ]
            },
            {
                "name": "Test Category 2",
                "image_urls": [
                    "https://example.com/image3.jpg"
                ]
            }
        ]
    
    @pytest.fixture
    def crawler(self):
        """Create an ImageCrawler instance."""
        return ImageCrawler()
    
    def test_load_data_json_valid(self, crawler, temp_dir, sample_data):
        """Test loading valid JSON data."""
        data_file = temp_dir / "test_data.json"
        
        with open(data_file, 'w') as f:
            json.dump(sample_data, f)
        
        loaded_data = crawler.load_data_json(data_file)
        
        assert loaded_data == sample_data
        assert len(loaded_data) == 2
        assert loaded_data[0]['name'] == "Test Category 1"
        assert len(loaded_data[0]['image_urls']) == 2
    
    def test_load_data_json_file_not_found(self, crawler, temp_dir):
        """Test loading non-existent JSON file."""
        non_existent_file = temp_dir / "non_existent.json"
        
        with pytest.raises(FileNotFoundError):
            crawler.load_data_json(non_existent_file)
    
    def test_load_data_json_invalid_structure(self, crawler, temp_dir):
        """Test loading JSON with invalid structure."""
        data_file = temp_dir / "invalid_data.json"
        
        # Invalid data - not a list
        invalid_data = {"not": "a list"}
        
        with open(data_file, 'w') as f:
            json.dump(invalid_data, f)
        
        with pytest.raises(ValueError, match="Data must be a list"):
            crawler.load_data_json(data_file)
    
    def test_load_data_json_missing_name(self, crawler, temp_dir):
        """Test loading JSON with missing name field."""
        data_file = temp_dir / "invalid_data.json"
        
        # Missing 'name' field
        invalid_data = [{"image_urls": ["url1", "url2"]}]
        
        with open(data_file, 'w') as f:
            json.dump(invalid_data, f)
        
        with pytest.raises(ValueError, match="missing 'name' field"):
            crawler.load_data_json(data_file)
    
    def test_load_data_json_missing_image_urls(self, crawler, temp_dir):
        """Test loading JSON with missing image_urls field."""
        data_file = temp_dir / "invalid_data.json"
        
        # Missing 'image_urls' field
        invalid_data = [{"name": "Test Category"}]
        
        with open(data_file, 'w') as f:
            json.dump(invalid_data, f)
        
        with pytest.raises(ValueError, match="missing 'image_urls' field"):
            crawler.load_data_json(data_file)
    
    def test_load_data_json_invalid_image_urls_type(self, crawler, temp_dir):
        """Test loading JSON with invalid image_urls type."""
        data_file = temp_dir / "invalid_data.json"
        
        # image_urls is not a list
        invalid_data = [{"name": "Test Category", "image_urls": "not a list"}]
        
        with open(data_file, 'w') as f:
            json.dump(invalid_data, f)
        
        with pytest.raises(ValueError, match="'image_urls' must be a list"):
            crawler.load_data_json(data_file)
    
    @pytest.mark.asyncio
    async def test_process_single_image_success(self, crawler):
        """Test successful processing of a single image."""
        image_url = "https://example.com/test.jpg"
        category_name = "Test Category"
        
        # Mock downloader
        mock_download_result = {
            'file_path': '/path/to/image.jpg',
            'size_bytes': 12345,
            'dimensions': (100, 100)
        }
        
        mock_downloader = AsyncMock()
        mock_downloader.download_image.return_value = mock_download_result
        mock_downloader.batch_download.return_value = [mock_download_result]
        
        # Mock searcher
        mock_search_results = {
            'similar_images': [
                {
                    'url': 'https://example.com/similar1.jpg',
                    'title': 'Similar Image 1'
                }
            ]
        }
        
        mock_similar_images = [
            {
                'url': 'https://example.com/similar1.jpg',
                'title': 'Similar Image 1',
                'source_page': 'https://example.com/page1'
            }
        ]
        
        crawler.downloader = mock_downloader
        crawler.searcher = Mock()
        crawler.searcher.reverse_image_search.return_value = mock_search_results
        crawler.searcher.extract_similar_images.return_value = mock_similar_images
        
        result = await crawler.process_single_image(image_url, category_name)
        
        assert result is not None
        assert result['input_image'] == image_url
        assert result['category'] == category_name
        assert len(result['similar_images']) > 0
        assert result['total_found'] == 1
    
    @pytest.mark.asyncio
    async def test_process_single_image_download_failure(self, crawler):
        """Test processing when image download fails."""
        image_url = "https://example.com/test.jpg"
        category_name = "Test Category"
        
        # Mock downloader failure
        mock_downloader = AsyncMock()
        mock_downloader.download_image.return_value = None
        
        crawler.downloader = mock_downloader
        
        result = await crawler.process_single_image(image_url, category_name)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_process_single_image_search_failure(self, crawler):
        """Test processing when search fails."""
        image_url = "https://example.com/test.jpg"
        category_name = "Test Category"
        
        # Mock successful download but failed search
        mock_download_result = {
            'file_path': '/path/to/image.jpg',
            'size_bytes': 12345,
            'dimensions': (100, 100)
        }
        
        mock_downloader = AsyncMock()
        mock_downloader.download_image.return_value = mock_download_result
        
        crawler.downloader = mock_downloader
        crawler.searcher = Mock()
        crawler.searcher.reverse_image_search.return_value = None
        
        result = await crawler.process_single_image(image_url, category_name)
        
        assert result is not None
        assert result['input_image'] == image_url
        assert result['similar_images'] == []
        assert result['total_found'] == 0
    
    @pytest.mark.asyncio
    async def test_process_category(self, crawler, sample_data):
        """Test processing a category."""
        category_data = sample_data[0]  # First category
        
        # Mock process_single_image to return success
        mock_image_result = {
            'input_image': 'https://example.com/image1.jpg',
            'similar_images': [{'url': 'similar1.jpg'}],
            'total_found': 1,
            'category': 'Test Category 1'
        }
        
        with patch.object(crawler, 'process_single_image', return_value=mock_image_result):
            result = await crawler.process_category(category_data)
        
        assert result['category'] == "Test Category 1"
        assert len(result['results']) == 2  # Two images in category
        assert result['total_similar_found'] == 2  # One similar per image
        assert result['processing_time'] > 0
    
    def test_save_results_json(self, crawler, temp_dir, sample_data):
        """Test saving results in JSON format."""
        # Mock results
        results = [
            {
                'category': 'Test Category',
                'results': [
                    {
                        'input_image': 'https://example.com/test.jpg',
                        'similar_images': []
                    }
                ]
            }
        ]
        
        with patch('src.main.settings') as mock_settings:
            mock_settings.OUTPUT_DIRECTORY = temp_dir
            
            output_file = crawler.save_results(results, 'json')
        
        assert output_file.exists()
        assert output_file.suffix == '.json'
        
        # Verify content
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data == results
    
    def test_generate_summary_report(self, crawler):
        """Test generating summary report."""
        results = [
            {
                'category': 'Category 1',
                'input_images': ['url1', 'url2'],
                'total_similar_found': 5,
                'processing_time': 10.5
            },
            {
                'category': 'Category 2',
                'input_images': ['url3'],
                'total_similar_found': 3,
                'processing_time': 7.2,
                'error': 'Some error'
            }
        ]
        
        summary = crawler.generate_summary_report(results)
        
        assert summary['summary']['total_categories'] == 2
        assert summary['summary']['successful_categories'] == 1
        assert summary['summary']['failed_categories'] == 1
        assert summary['summary']['total_input_images'] == 3
        assert summary['summary']['total_similar_images_found'] == 8
        assert summary['summary']['total_processing_time_seconds'] == 17.7
        
        assert len(summary['category_breakdown']) == 2
        assert summary['category_breakdown'][0]['success'] is True
        assert summary['category_breakdown'][1]['success'] is False
