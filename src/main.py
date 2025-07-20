"""Main orchestrator for the Image Similarity Crawler."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import sys

from config.settings import settings
from src.image_downloader import ImageDownloader
from src.serpapi_search import SerpAPISearcher
from src.similarity_detector import SimilarityDetector

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ImageCrawler:
    """Main class for orchestrating image similarity crawling."""
    
    def __init__(self):
        self.downloader = None
        self.searcher = SerpAPISearcher()
        self.similarity_detector = SimilarityDetector()
        self.results = []
        
        # Ensure output directories exist
        settings.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        settings.CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    def load_data_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load and validate data from JSON file.
        
        Args:
            file_path: Path to data.json file
            
        Returns:
            List of category data
        """
        try:
            logger.info(f"Loading data from {file_path}")
            
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate data structure
            if not isinstance(data, list):
                raise ValueError("Data must be a list of category objects")
            
            for i, category in enumerate(data):
                if not isinstance(category, dict):
                    raise ValueError(f"Category {i} must be a dictionary")
                
                if 'name' not in category:
                    raise ValueError(f"Category {i} missing 'name' field")
                
                if 'image_urls' not in category:
                    raise ValueError(f"Category {i} missing 'image_urls' field")
                
                if not isinstance(category['image_urls'], list):
                    raise ValueError(f"Category {i} 'image_urls' must be a list")
            
            logger.info(f"Successfully loaded {len(data)} categories")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    async def process_category(self, category_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single category of images.
        
        Args:
            category_data: Category data with name and image_urls
            
        Returns:
            Processing results for the category
        """
        category_name = category_data['name']
        image_urls = category_data['image_urls']
        
        logger.info(f"Processing category: {category_name} ({len(image_urls)} images)")
        
        start_time = time.time()
        category_results = {
            'category': category_name,
            'input_images': image_urls,
            'results': [],
            'processing_time': 0,
            'total_similar_found': 0,
            'errors': []
        }
        
        async with ImageDownloader() as downloader:
            self.downloader = downloader
            
            # Process each image in the category
            for i, image_url in enumerate(image_urls):
                logger.info(f"Processing image {i+1}/{len(image_urls)} in {category_name}: {image_url}")
                
                try:
                    image_result = await self.process_single_image(image_url, category_name)
                    if image_result:
                        category_results['results'].append(image_result)
                        category_results['total_similar_found'] += len(image_result.get('similar_images', []))
                    else:
                        category_results['errors'].append(f"Failed to process: {image_url}")
                        
                except Exception as e:
                    error_msg = f"Error processing {image_url}: {str(e)}"
                    logger.error(error_msg)
                    category_results['errors'].append(error_msg)
        
        category_results['processing_time'] = time.time() - start_time
        
        logger.info(f"Completed category {category_name}: "
                   f"{len(category_results['results'])} processed, "
                   f"{category_results['total_similar_found']} similar images found")
        
        return category_results
    
    async def process_single_image(self, image_url: str, category_name: str) -> Optional[Dict[str, Any]]:
        """
        Process a single image: download and find similar images.
        
        Args:
            image_url: URL of the image to process
            category_name: Name of the category this image belongs to
            
        Returns:
            Processing result for the image
        """
        try:
            start_time = time.time()
            
            # Download the original image
            logger.debug(f"Downloading original image: {image_url}")
            download_result = await self.downloader.download_image(image_url)
            
            if not download_result:
                logger.error(f"Failed to download image: {image_url}")
                return None
            
            # Perform reverse image search
            logger.debug(f"Performing reverse image search: {image_url}")
            search_results = self.searcher.reverse_image_search(image_url)
            
            if not search_results:
                logger.warning(f"No search results for: {image_url}")
                return {
                    'input_image': image_url,
                    'similar_images': [],
                    'processing_time': time.time() - start_time,
                    'total_found': 0,
                    'category': category_name
                }
            
            # Extract similar images
            similar_images = self.searcher.extract_similar_images(search_results)
            
            # Download similar images for local similarity analysis (optional)
            downloaded_similar = []
            if similar_images:
                logger.debug(f"Downloading {len(similar_images)} similar images")
                
                # Limit the number of similar images to download
                max_download = min(len(similar_images), 10)
                similar_urls = [img['url'] for img in similar_images[:max_download]]
                
                download_results = await self.downloader.batch_download(
                    similar_urls, 
                    settings.OUTPUT_DIRECTORY / category_name.replace(' ', '_')
                )
                
                # Combine download results with similar image metadata
                for i, (similar_img, download_result) in enumerate(zip(similar_images[:max_download], download_results)):
                    if download_result:
                        similar_img.update({
                            'local_path': download_result['file_path'],
                            'file_size': download_result['size_bytes'],
                            'downloaded': True
                        })
                    else:
                        similar_img['downloaded'] = False
                    
                    downloaded_similar.append(similar_img)
            
            result = {
                'input_image': image_url,
                'input_image_local_path': download_result['file_path'],
                'similar_images': downloaded_similar,
                'processing_time': time.time() - start_time,
                'total_found': len(similar_images),
                'category': category_name,
                'metadata': {
                    'search_engine': 'google_images_serpapi',
                    'original_image_size': download_result.get('size_bytes'),
                    'original_image_dimensions': download_result.get('dimensions')
                }
            }
            
            logger.info(f"Successfully processed {image_url}: {len(similar_images)} similar images found")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_url}: {str(e)}")
            return None
    
    async def process_all_categories(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all categories from the data file.
        
        Args:
            data: List of category data
            
        Returns:
            List of processing results for all categories
        """
        logger.info(f"Starting processing of {len(data)} categories")
        start_time = time.time()
        
        all_results = []
        
        for i, category_data in enumerate(data):
            logger.info(f"Processing category {i+1}/{len(data)}: {category_data['name']}")
            
            try:
                category_result = await self.process_category(category_data)
                all_results.append(category_result)
                
            except Exception as e:
                error_msg = f"Error processing category {category_data['name']}: {str(e)}"
                logger.error(error_msg)
                all_results.append({
                    'category': category_data['name'],
                    'error': error_msg,
                    'results': [],
                    'processing_time': 0,
                    'total_similar_found': 0
                })
        
        total_time = time.time() - start_time
        total_similar = sum(result.get('total_similar_found', 0) for result in all_results)
        
        logger.info(f"Completed processing all categories in {total_time:.2f}s. "
                   f"Total similar images found: {total_similar}")
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]], output_format: str = 'json') -> Path:
        """
        Save processing results to file.
        
        Args:
            results: Processing results
            output_format: Output format ('json' or 'csv')
            
        Returns:
            Path to saved file
        """
        timestamp = int(time.time())
        
        if output_format.lower() == 'json':
            output_file = settings.OUTPUT_DIRECTORY / f"results_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        elif output_format.lower() == 'csv':
            import pandas as pd
            
            # Flatten results for CSV
            csv_data = []
            for category_result in results:
                category_name = category_result['category']
                
                for image_result in category_result.get('results', []):
                    input_url = image_result['input_image']
                    
                    for similar_img in image_result.get('similar_images', []):
                        csv_data.append({
                            'category': category_name,
                            'input_url': input_url,
                            'similar_url': similar_img['url'],
                            'title': similar_img.get('title', ''),
                            'source_page': similar_img.get('source_page', ''),
                            'source_domain': similar_img.get('source_domain', ''),
                            'downloaded': similar_img.get('downloaded', False),
                            'local_path': similar_img.get('local_path', ''),
                            'dimensions_width': similar_img.get('dimensions', {}).get('width'),
                            'dimensions_height': similar_img.get('dimensions', {}).get('height')
                        })
            
            df = pd.DataFrame(csv_data)
            output_file = settings.OUTPUT_DIRECTORY / f"results_{timestamp}.csv"
            df.to_csv(output_file, index=False)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Results saved to: {output_file}")
        return output_file
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary report of the processing results.
        
        Args:
            results: Processing results
            
        Returns:
            Summary report
        """
        total_categories = len(results)
        total_input_images = sum(len(r.get('input_images', [])) for r in results)
        total_similar_found = sum(r.get('total_similar_found', 0) for r in results)
        total_processing_time = sum(r.get('processing_time', 0) for r in results)
        
        successful_categories = sum(1 for r in results if not r.get('error'))
        failed_categories = total_categories - successful_categories
        
        summary = {
            'summary': {
                'total_categories': total_categories,
                'successful_categories': successful_categories,
                'failed_categories': failed_categories,
                'total_input_images': total_input_images,
                'total_similar_images_found': total_similar_found,
                'total_processing_time_seconds': round(total_processing_time, 2),
                'average_similar_per_input': round(total_similar_found / max(total_input_images, 1), 2)
            },
            'category_breakdown': []
        }
        
        for result in results:
            category_summary = {
                'category': result['category'],
                'input_images_count': len(result.get('input_images', [])),
                'similar_images_found': result.get('total_similar_found', 0),
                'processing_time_seconds': round(result.get('processing_time', 0), 2),
                'success': not result.get('error'),
                'error': result.get('error')
            }
            summary['category_breakdown'].append(category_summary)
        
        return summary

async def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Image Similarity Crawler')
    parser.add_argument('--data', '-d', type=str, default='data.json',
                       help='Path to data.json file (default: data.json)')
    parser.add_argument('--output-format', '-f', choices=['json', 'csv'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--summary', '-s', action='store_true',
                       help='Generate summary report')
    
    args = parser.parse_args()
    
    try:
        # Validate settings
        if not settings.validate():
            logger.error("Configuration validation failed. Please check your settings.")
            sys.exit(1)
        
        # Initialize crawler
        crawler = ImageCrawler()
        
        # Load data
        data_path = Path(args.data)
        data = crawler.load_data_json(data_path)
        
        # Process all categories
        results = await crawler.process_all_categories(data)
        
        # Save results
        output_file = crawler.save_results(results, args.output_format)
        
        # Generate summary if requested
        if args.summary:
            summary = crawler.generate_summary_report(results)
            summary_file = settings.OUTPUT_DIRECTORY / f"summary_{int(time.time())}.json"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Summary report saved to: {summary_file}")
            
            # Print summary to console
            print("\n" + "="*50)
            print("PROCESSING SUMMARY")
            print("="*50)
            print(f"Categories processed: {summary['summary']['total_categories']}")
            print(f"Input images: {summary['summary']['total_input_images']}")
            print(f"Similar images found: {summary['summary']['total_similar_images_found']}")
            print(f"Processing time: {summary['summary']['total_processing_time_seconds']}s")
            print(f"Average similar per input: {summary['summary']['average_similar_per_input']}")
        
        logger.info("Processing completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
