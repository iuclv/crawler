"""SerpAPI integration for reverse image search using Google Images."""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from serpapi import GoogleSearch
import aiohttp
from urllib.parse import quote

from config.settings import settings

logger = logging.getLogger(__name__)

class SerpAPISearcher:
    """Handles reverse image search using SerpAPI's Google Images."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.SERPAPI_KEY
        self.max_results = settings.MAX_RESULTS_PER_IMAGE
        self.requests_per_hour = settings.SERPAPI_REQUESTS_PER_HOUR
        self.last_request_time = 0
        self.request_count = 0
        self.request_times = []
        
        if not self.api_key:
            logger.warning("SerpAPI key not provided. Search functionality will be disabled.")
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Remove requests older than 1 hour
        self.request_times = [t for t in self.request_times if current_time - t < 3600]
        
        # Check if we're at the rate limit
        if len(self.request_times) >= self.requests_per_hour:
            oldest_request = min(self.request_times)
            wait_time = 3600 - (current_time - oldest_request)
            if wait_time > 0:
                logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
                time.sleep(wait_time)
        
        # Add current request time
        self.request_times.append(current_time)
    
    def reverse_image_search(self, image_url: str) -> Optional[Dict[str, Any]]:
        """
        Perform reverse image search using SerpAPI.
        
        Args:
            image_url: URL of the image to search for
            
        Returns:
            Dictionary with search results or None if failed
        """
        if not self.api_key:
            logger.error("SerpAPI key not available")
            return None
        
        try:
            # Enforce rate limiting
            self._check_rate_limit()
            
            logger.debug(f"Performing reverse image search for: {image_url}")
            
            # Set up search parameters
            params = {
                "engine": "google_reverse_image",
                "image_url": image_url,
                "api_key": self.api_key,
                "num": min(self.max_results, 100)  # SerpAPI limit is 100
            }
            
            # Perform search
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "error" in results:
                logger.error(f"SerpAPI error: {results['error']}")
                return None
            
            logger.info(f"Reverse image search completed for {image_url}")
            return self.parse_serpapi_response(results)
            
        except Exception as e:
            logger.error(f"Error performing reverse image search: {str(e)}")
            return None
    
    def parse_serpapi_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse SerpAPI response and extract relevant information.
        
        Args:
            response: Raw SerpAPI response
            
        Returns:
            Parsed and structured response
        """
        try:
            parsed_results = {
                "search_metadata": response.get("search_metadata", {}),
                "search_parameters": response.get("search_parameters", {}),
                "similar_images": [],
                "image_results": [],
                "total_results": 0
            }
            
            # Extract similar images from image_results
            image_results = response.get("image_results", [])
            for result in image_results:
                similar_image = {
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "displayed_link": result.get("displayed_link", ""),
                    "thumbnail": result.get("thumbnail", ""),
                    "source": result.get("source", ""),
                    "original": {
                        "link": result.get("original", {}).get("link", ""),
                        "width": result.get("original", {}).get("width"),
                        "height": result.get("original", {}).get("height")
                    }
                }
                parsed_results["similar_images"].append(similar_image)
            
            # Extract additional image results if available
            inline_images = response.get("inline_images", [])
            for image in inline_images:
                image_result = {
                    "title": image.get("title", ""),
                    "link": image.get("link", ""),
                    "thumbnail": image.get("thumbnail", ""),
                    "source": image.get("source", ""),
                    "original": {
                        "link": image.get("original", {}).get("link", ""),
                        "width": image.get("original", {}).get("width"),
                        "height": image.get("original", {}).get("height")
                    }
                }
                parsed_results["image_results"].append(image_result)
            
            parsed_results["total_results"] = len(parsed_results["similar_images"]) + len(parsed_results["image_results"])
            
            logger.debug(f"Parsed {parsed_results['total_results']} results from SerpAPI response")
            return parsed_results
            
        except Exception as e:
            logger.error(f"Error parsing SerpAPI response: {str(e)}")
            return {
                "similar_images": [],
                "image_results": [],
                "total_results": 0,
                "error": str(e)
            }
    
    def extract_similar_images(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and format similar images from search results.
        
        Args:
            search_results: Parsed search results
            
        Returns:
            List of similar images with metadata
        """
        similar_images = []
        
        # Combine similar_images and image_results
        all_images = (search_results.get("similar_images", []) + 
                     search_results.get("image_results", []))
        
        for image in all_images:
            # Skip images without valid links
            original_link = image.get("original", {}).get("link")
            if not original_link:
                continue
            
            similar_image = {
                "url": original_link,
                "title": image.get("title", ""),
                "source_page": image.get("link", ""),
                "source_domain": image.get("displayed_link", ""),
                "thumbnail_url": image.get("thumbnail", ""),
                "dimensions": {
                    "width": image.get("original", {}).get("width"),
                    "height": image.get("original", {}).get("height")
                },
                "metadata": {
                    "source": "serpapi_google",
                    "search_engine": "google_images"
                }
            }
            
            similar_images.append(similar_image)
        
        logger.info(f"Extracted {len(similar_images)} similar images")
        return similar_images
    
    async def batch_reverse_search(self, image_urls: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        Perform reverse image search for multiple images.
        
        Args:
            image_urls: List of image URLs to search
            
        Returns:
            List of search results (same order as input)
        """
        if not self.api_key:
            logger.error("SerpAPI key not available for batch search")
            return [None] * len(image_urls)
        
        logger.info(f"Starting batch reverse image search for {len(image_urls)} images")
        
        results = []
        for i, url in enumerate(image_urls):
            logger.info(f"Processing image {i+1}/{len(image_urls)}: {url}")
            
            result = self.reverse_image_search(url)
            results.append(result)
            
            # Add delay between requests to respect rate limits
            if i < len(image_urls) - 1:  # Don't sleep after the last request
                sleep_time = 3600 / self.requests_per_hour  # Spread requests evenly
                await asyncio.sleep(sleep_time)
        
        successful_searches = sum(1 for r in results if r is not None)
        logger.info(f"Batch reverse search completed: {successful_searches}/{len(image_urls)} successful")
        
        return results
    
    def handle_api_errors(self, response: Dict[str, Any]) -> bool:
        """
        Handle API errors and return whether the request was successful.
        
        Args:
            response: SerpAPI response
            
        Returns:
            True if successful, False if error
        """
        if "error" in response:
            error_msg = response["error"]
            logger.error(f"SerpAPI error: {error_msg}")
            
            # Handle specific error types
            if "Invalid API key" in error_msg:
                logger.error("Invalid SerpAPI key. Please check your configuration.")
            elif "rate limit" in error_msg.lower():
                logger.warning("SerpAPI rate limit exceeded. Consider upgrading your plan.")
            elif "quota" in error_msg.lower():
                logger.warning("SerpAPI quota exceeded. Please check your usage.")
            
            return False
        
        return True
    
    def get_usage_info(self) -> Dict[str, Any]:
        """
        Get current usage information.
        
        Returns:
            Dictionary with usage statistics
        """
        current_time = time.time()
        recent_requests = [t for t in self.request_times if current_time - t < 3600]
        
        return {
            "requests_last_hour": len(recent_requests),
            "requests_per_hour_limit": self.requests_per_hour,
            "remaining_requests": max(0, self.requests_per_hour - len(recent_requests)),
            "api_key_configured": bool(self.api_key)
        }
