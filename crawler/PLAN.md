# Image Similarity Crawler - Project Plan

## Overview
A Python project for finding similar images from the internet based on a structured JSON input file (`data.json`). The system processes image URLs from the JSON file and uses SerpAPI's reverse image search to find visually similar images across the web.

## Project Structure
```
image-crawler/
├── src/
│   ├── __init__.py
│   ├── image_downloader.py      # Download and validate images
│   ├── similarity_detector.py   # Image similarity algorithms
│   ├── serpapi_search.py        # SerpAPI integration for image search
│   └── main.py                 # Main orchestrator
├── config/
│   ├── settings.py             # Configuration management
│   └── .env.example           # Environment variables template
├── data/
│   ├── data.json              # Input JSON file with image URLs
│   ├── output/                # Downloaded similar images
│   └── cache/                 # Cached data and models
├── tests/
│   ├── test_downloader.py
│   ├── test_similarity.py
│   └── test_crawler.py
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── PLAN.md                   # This file
└── .gitignore               # Git ignore rules
```

## Core Components

### 1. Image Downloader Module (`image_downloader.py`)
**Purpose**: Download and validate images from URLs
- Download images from provided URLs
- Support multiple formats (JPEG, PNG, WebP, GIF)
- Implement retry logic with exponential backoff
- Validate image integrity and size constraints
- Handle HTTP errors and timeouts gracefully
- Resize/normalize images for processing

**Key Functions**:
- `download_image(url, output_path)`
- `validate_image(image_path)`
- `resize_image(image, target_size)`
- `batch_download(urls, concurrent_limit)`

### 2. Similarity Detection Module (`similarity_detector.py`)
**Purpose**: Detect visual similarity between images
- **Perceptual Hashing**: pHash, dHash, aHash for basic similarity
- **Feature Extraction**: Use pre-trained CNN models (ResNet, VGG)
- **Semantic Similarity**: CLIP embeddings for content understanding
- **Custom Metrics**: Combine multiple similarity scores

**Key Functions**:
- `calculate_perceptual_hash(image)`
- `extract_features(image, model_name)`
- `calculate_similarity(image1, image2, method)`
- `find_similar_images(target_image, candidate_images, threshold)`

### 3. SerpAPI Search Module (`serpapi_search.py`)
**Purpose**: Search for similar images using SerpAPI's Google Images reverse search
- **Reverse Image Search**: Google Images via SerpAPI
- **API Integration**: Clean JSON responses without scraping
- **Rate Limiting**: Built-in API rate limiting
- **Reliable Results**: No anti-bot measures to handle

**Key Functions**:
- `reverse_image_search(image_url)`
- `parse_serpapi_response(response)`
- `extract_similar_images(results)`
- `handle_api_errors(response)`

### 4. Main Orchestrator (`main.py`)
**Purpose**: Coordinate all modules and manage workflow
- Process data.json input file
- Extract image URLs from JSON structure
- Manage concurrent operations for similar image search
- Generate reports organized by original categories
- Handle configuration and logging

**Key Functions**:
- `ImageCrawler` class with main interface
- `load_data_json(file_path)`
- `process_category_images(category_data)`
- `generate_category_report(results)`
- `save_results(data, format)`

## Implementation Phases

### Phase 1: Foundation (Week 1)
**Goals**: Set up project infrastructure and basic functionality

**Tasks**:
- [ ] Create project structure and virtual environment
- [ ] Set up configuration system with environment variables
- [ ] Implement JSON data loader for data.json format
- [ ] Implement basic image downloader with error handling
- [ ] Add logging system with different levels
- [ ] Create unit tests for data loading and downloader modules

**Deliverables**:
- JSON data loader for structured input
- Working image downloader
- Configuration management
- Basic test suite
- Project documentation

### Phase 2: Similarity Detection (Week 2)
**Goals**: Implement core similarity detection algorithms

**Tasks**:
- [ ] Implement perceptual hashing algorithms
- [ ] Add feature extraction using pre-trained models
- [ ] Integrate CLIP for semantic similarity
- [ ] Create similarity scoring and ranking system
- [ ] Add image preprocessing pipeline
- [ ] Optimize performance for batch processing categories

**Deliverables**:
- Multiple similarity detection methods
- Configurable similarity thresholds
- Performance benchmarks
- Similarity visualization tools

### Phase 3: SerpAPI Integration (Week 3)
**Goals**: Integrate SerpAPI for reliable image search

**Tasks**:
- [ ] Set up SerpAPI account and API key management
- [ ] Implement Google Images reverse search via SerpAPI
- [ ] Add response parsing and error handling
- [ ] Create result filtering and validation
- [ ] Implement API rate limiting and quota management
- [ ] Add result caching to minimize API calls
- [ ] Handle batch processing for multiple categories

**Deliverables**:
- SerpAPI integration module
- Reliable reverse image search
- API error handling and recovery
- Cost-effective caching system

### Phase 4: Integration & Optimization (Week 4)
**Goals**: Combine modules and optimize performance

**Tasks**:
- [ ] Integrate all modules into main orchestrator
- [ ] Add concurrent processing with asyncio for categories
- [ ] Implement result filtering and deduplication
- [ ] Create comprehensive reporting system organized by categories
- [ ] Add command-line interface with data.json input
- [ ] Performance optimization and memory management

**Deliverables**:
- Complete working system with JSON input support
- CLI interface for data.json processing
- Category-organized output and reporting
- Performance optimizations
- Comprehensive documentation

## Key Technologies & Dependencies

### Core Libraries
```python
# Image Processing
Pillow>=9.0.0              # Image manipulation
opencv-python>=4.5.0       # Computer vision
imagehash>=4.3.0          # Perceptual hashing

# Machine Learning
torch>=1.12.0             # Deep learning framework
torchvision>=0.13.0       # Pre-trained vision models
transformers>=4.20.0      # CLIP and other transformers
scikit-learn>=1.1.0       # ML utilities

# API Integration
requests>=2.28.0          # HTTP requests
aiohttp>=3.8.0           # Async HTTP
google-search-results>=2.4.2  # SerpAPI Python client

# Utilities
numpy>=1.21.0            # Numerical computing
pandas>=1.4.0            # Data manipulation
tqdm>=4.64.0             # Progress bars
python-dotenv>=0.20.0    # Environment management
loguru>=0.6.0            # Enhanced logging
```

### Optional Dependencies
```python
# Advanced Features
faiss-cpu>=1.7.0         # Efficient similarity search
redis>=4.3.0             # Caching and job queues
celery>=5.2.0            # Distributed task processing
```

## Configuration Options

### Environment Variables (.env)
```bash
# API Keys
SERPAPI_KEY=your_serpapi_key

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
```

### Configuration File (config/settings.py)
```python
# Image Processing
IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'webp', 'gif']
MAX_IMAGE_SIZE_MB = 10
TARGET_IMAGE_SIZE = (224, 224)

# Similarity Detection
SIMILARITY_METHODS = ['phash', 'features', 'clip']
FEATURE_MODEL = 'resnet50'
CLIP_MODEL = 'ViT-B/32'

# SerpAPI Configuration
SERPAPI_ENGINE = 'google_reverse_image'
SERPAPI_LOCATION = 'United States'
SERPAPI_LANGUAGE = 'en'
```

## Usage Examples

### Basic Usage
```python
from src.main import ImageCrawler

# Initialize crawler
crawler = ImageCrawler(
    similarity_threshold=0.8,
    max_results_per_image=10,
    serpapi_key='your_api_key'
)

# Process data.json file
results = crawler.process_data_file('data/data.json')
crawler.save_results(results, format='json')
```

### Command Line Interface
```bash
# Basic search
python -m src.main --input data/data.json --output results/ --threshold 0.8

# Advanced options
python -m src.main \
    --input data/data.json \
    --output results/ \
    --threshold 0.8 \
    --max-results 20 \
    --serpapi-key your_key \
    --concurrent 5 \
    --format json
```

### Batch Processing
```python
from src.main import ImageCrawler

crawler = ImageCrawler()

# Process categories from data.json
data = crawler.load_data_json('data/data.json')
for category in data:
    results = crawler.process_category(category)
    crawler.save_category_results(results, category['name'])
```

## Output Formats

### Input Data Format (data.json)
```json
[
    {
        "name": "Pytophthora Fruit Rot",
        "image_urls": [
            "https://plantwiseplusknowledgebank.org/cms/10.1079/pwkb.20187800447/asset/adf588d2-f754-4626-b863-a5cab63e39ad/assets/graphic/phytophthora-on-durian-global-1.jpg"
        ]
    },
    {
        "name": "Apple Scab",
        "image_urls": [
            "https://example.com/apple-scab1.jpg",
            "https://example.com/apple-scab2.jpg"
        ]
    }
]
```

### Output Results Format
```json
{
    "category": "Pytophthora Fruit Rot",
    "results": [
        {
            "input_image": "https://plantwiseplusknowledgebank.org/.../phytophthora-on-durian-global-1.jpg",
            "similar_images": [
                {
                    "url": "https://site.com/similar1.jpg",
                    "similarity_score": 0.95,
                    "source": "serpapi_google_images",
                    "metadata": {
                        "size": "1024x768",
                        "format": "JPEG",
                        "source_page": "https://site.com/page"
                    }
                }
            ],
            "processing_time": 2.34,
            "total_found": 15
        }
    ]
}
```

### CSV Report
```csv
category,input_url,similar_url,similarity_score,source,size,format
Pytophthora Fruit Rot,https://plantwiseplusknowledgebank.org/.../phytophthora-on-durian-global-1.jpg,https://site.com/sim1.jpg,0.95,serpapi_google,1024x768,JPEG
```

## Performance Considerations

### Optimization Strategies
- **Async Processing**: Use asyncio for concurrent downloads
- **Caching**: Cache downloaded images and computed features
- **Batch Processing**: Process multiple images simultaneously
- **Memory Management**: Stream large images, cleanup resources
- **Model Optimization**: Use quantized models for faster inference

### Scalability
- **Distributed Processing**: Use Celery for large-scale operations
- **Database Integration**: Store results in PostgreSQL/MongoDB
- **Cloud Storage**: Use AWS S3/Google Cloud for image storage
- **Load Balancing**: Distribute requests across multiple proxies

## Legal & Ethical Considerations

### Best Practices
- **API Usage**: Follow SerpAPI terms of service and rate limits
- **Cost Management**: Implement caching to minimize API calls
- **Attribution**: Provide source attribution for found images
- **Copyright**: Respect intellectual property rights
- **Privacy**: Handle personal images responsibly

### Compliance
- **GDPR**: Handle personal data according to regulations
- **SerpAPI ToS**: Respect SerpAPI terms of service
- **Fair Use**: Ensure usage falls under fair use guidelines

## Testing Strategy

### Unit Tests
- Image download and validation
- Similarity calculation accuracy
- SerpAPI integration functionality
- Configuration management

### Integration Tests
- End-to-end workflow testing
- SerpAPI response handling
- Error handling and recovery
- Performance benchmarking

### Test Data
- Curated image datasets for similarity testing
- Mock SerpAPI responses for testing
- Performance test suites with large datasets

## Monitoring & Logging

### Logging Levels
- **DEBUG**: Detailed processing information
- **INFO**: General operation status
- **WARNING**: Non-critical issues
- **ERROR**: Processing failures
- **CRITICAL**: System failures

### Metrics to Track
- Download success/failure rates
- Similarity detection accuracy
- Processing time per image
- Memory and CPU usage
- API rate limit status

## Future Enhancements

### Advanced Features
- **Video Similarity**: Extend to video content
- **Audio Similarity**: Add audio fingerprinting
- **Real-time Processing**: Stream processing capabilities
- **Machine Learning**: Custom similarity models
- **Mobile App**: React Native mobile interface

### Integrations
- **Cloud APIs**: Google Vision, AWS Rekognition
- **Social Media**: Instagram, Pinterest APIs
- **E-commerce**: Product image search
- **Content Management**: WordPress, Drupal plugins

## Success Metrics

### Technical Metrics
- **Accuracy**: >90% relevant similar images found
- **Performance**: <5 seconds average processing time per image
- **Reliability**: <1% failure rate for valid URLs
- **Scalability**: Handle 1000+ images per hour

### User Experience
- **Ease of Use**: Simple CLI and Python API
- **Documentation**: Comprehensive guides and examples
- **Error Handling**: Clear error messages and recovery
- **Flexibility**: Configurable for different use cases

---

## Getting Started

1. **Clone Repository**: `git clone <repo-url>`
2. **Setup Environment**: `python -m venv venv && source venv/bin/activate`
3. **Install Dependencies**: `pip install -r requirements.txt`
4. **Configure Settings**: Copy `.env.example` to `.env` and update
5. **Run Tests**: `python -m pytest tests/`
6. **Start Crawling**: `python -m src.main --help`

For detailed setup instructions, see [README.md](README.md).
