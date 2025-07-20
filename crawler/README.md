# Image Similarity Crawler

A Python application for finding similar images from the internet based on structured JSON input. The system processes image URLs from a JSON file and uses SerpAPI's reverse image search to find visually similar images across the web.

## Features

- **Structured Input**: Process categories of images from `data.json`
- **Reverse Image Search**: Uses SerpAPI's Google Images for finding similar images
- **Concurrent Processing**: Async downloads and processing for better performance
- **Multiple Output Formats**: JSON and CSV output support
- **Comprehensive Logging**: Detailed logging with configurable levels
- **Error Handling**: Robust error handling with retry logic
- **Image Validation**: Validates downloaded images for integrity
- **Rate Limiting**: Respects API rate limits automatically

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd image-crawler
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp config/.env.example .env
   # Edit .env file with your SerpAPI key and other settings
   ```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# API Keys
SERPAPI_KEY=your_serpapi_key_here

# Search Settings
MAX_RESULTS_PER_IMAGE=50
SIMILARITY_THRESHOLD=0.8
CONCURRENT_DOWNLOADS=10

# Storage
OUTPUT_DIRECTORY=./data/output
CACHE_DIRECTORY=./data/cache
MAX_CACHE_SIZE_GB=5

# API Rate Limiting
SERPAPI_REQUESTS_PER_HOUR=100
MAX_RETRIES=3
TIMEOUT_SECONDS=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/crawler.log
```

### SerpAPI Setup

1. Sign up for a SerpAPI account at [serpapi.com](https://serpapi.com)
2. Get your API key from the dashboard
3. Add the key to your `.env` file

## Usage

### Input Format

Create a `data.json` file with the following structure:

```json
[
    {
        "name": "Plant Disease Category 1",
        "image_urls": [
            "https://example.com/disease1-image1.jpg",
            "https://example.com/disease1-image2.jpg"
        ]
    },
    {
        "name": "Plant Disease Category 2",
        "image_urls": [
            "https://example.com/disease2-image1.jpg"
        ]
    }
]
```

### Command Line Usage

```bash
# Basic usage
python -m src.main

# Specify custom data file
python -m src.main --data my_data.json

# Generate CSV output
python -m src.main --output-format csv

# Generate summary report
python -m src.main --summary

# Full example
python -m src.main --data data.json --output-format json --summary
```

### Programmatic Usage

```python
import asyncio
from pathlib import Path
from src.main import ImageCrawler

async def main():
    # Initialize crawler
    crawler = ImageCrawler()
    
    # Load data
    data = crawler.load_data_json(Path('data.json'))
    
    # Process all categories
    results = await crawler.process_all_categories(data)
    
    # Save results
    output_file = crawler.save_results(results, 'json')
    print(f"Results saved to: {output_file}")
    
    # Generate summary
    summary = crawler.generate_summary_report(results)
    print(f"Found {summary['summary']['total_similar_images_found']} similar images")

if __name__ == "__main__":
    asyncio.run(main())
```

## Output

### JSON Output

```json
{
  "category": "Plant Disease Category 1",
  "results": [
    {
      "input_image": "https://example.com/disease1-image1.jpg",
      "similar_images": [
        {
          "url": "https://similar-site.com/similar-image.jpg",
          "title": "Similar Disease Image",
          "source_page": "https://similar-site.com/page",
          "source_domain": "similar-site.com",
          "thumbnail_url": "https://similar-site.com/thumb.jpg",
          "dimensions": {
            "width": 800,
            "height": 600
          },
          "downloaded": true,
          "local_path": "./data/output/Plant_Disease_Category_1/abc123.jpg"
        }
      ],
      "processing_time": 5.23,
      "total_found": 15
    }
  ]
}
```

### CSV Output

The CSV format flattens the results for easy analysis:

```csv
category,input_url,similar_url,title,source_page,source_domain,downloaded,local_path,dimensions_width,dimensions_height
Plant Disease Category 1,https://example.com/disease1-image1.jpg,https://similar-site.com/similar-image.jpg,Similar Disease Image,https://similar-site.com/page,similar-site.com,true,./data/output/Plant_Disease_Category_1/abc123.jpg,800,600
```

## Project Structure

```
image-crawler/
├── src/
│   ├── __init__.py
│   ├── image_downloader.py      # Download and validate images
│   ├── similarity_detector.py   # Image similarity algorithms
│   ├── serpapi_search.py        # SerpAPI integration
│   └── main.py                 # Main orchestrator
├── config/
│   ├── __init__.py
│   ├── settings.py             # Configuration management
│   └── .env.example           # Environment variables template
├── data/
│   ├── data.json              # Input JSON file
│   ├── output/                # Downloaded similar images
│   └── cache/                 # Cached data
├── tests/
│   ├── __init__.py
│   ├── test_downloader.py
│   └── test_crawler.py
├── logs/                      # Log files
├── requirements.txt
├── README.md
└── PLAN.md
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src

# Run specific test file
python -m pytest tests/test_downloader.py

# Run with verbose output
python -m pytest -v
```

## Logging

The application provides comprehensive logging:

- **DEBUG**: Detailed processing information
- **INFO**: General operation status
- **WARNING**: Non-critical issues
- **ERROR**: Processing failures
- **CRITICAL**: System failures

Logs are written to both console and file (configurable via `LOG_FILE`).

## Error Handling

The system includes robust error handling:

- **Network errors**: Automatic retry with exponential backoff
- **API errors**: Graceful handling of SerpAPI errors and rate limits
- **Image validation**: Validates image format and size
- **File system errors**: Handles missing directories and permissions

## Performance Considerations

- **Concurrent Downloads**: Configurable concurrent download limit
- **Rate Limiting**: Automatic SerpAPI rate limiting
- **Memory Management**: Streams large images to avoid memory issues
- **Caching**: Caches downloaded images to avoid re-downloading

## Limitations

- **SerpAPI Dependency**: Requires SerpAPI subscription for search functionality
- **Rate Limits**: Subject to SerpAPI rate limits (configurable)
- **Image Formats**: Supports JPEG, PNG, WebP, GIF
- **File Size**: Maximum 50MB per image

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

1. Check the logs for detailed error information
2. Verify your SerpAPI key and quota
3. Ensure all dependencies are installed correctly
4. Check the GitHub issues for similar problems

## Changelog

### Version 1.0.0
- Initial release
- Basic image similarity search functionality
- SerpAPI integration
- JSON and CSV output formats
- Comprehensive test suite
