"""
Professional Amazon E-Commerce Data Scraper
Configuration-driven implementation with robust error handling
"""

import re
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from config_manager import BaseComponent, PathConfig

@dataclass
class ProductData:
    """Data structure for product information"""
    source: str
    product_id: str
    name: str
    price: Optional[str]
    mrp: Optional[str]
    discount: Optional[str]
    rating: Optional[str]
    url: str
    scraped_at: str

@dataclass
class ReviewData:
    """Data structure for review information"""
    source: str
    product_id: str
    product_name: str
    user_id: str
    review_text: str
    rating: Optional[str]
    review_date: str

class AmazonScraper(BaseComponent):
    """
    Professional Amazon scraper with configuration-driven architecture
    """

    def __init__(self):
        super().__init__("amazon_scraper")
        self.driver: Optional[webdriver.Chrome] = None
        self.scraper_config = self.get_component_config('scraping')['amazon']
        self.global_config = self.get_component_config('scraping')

    def _setup_selenium_driver(self) -> webdriver.Chrome:
        """Setup Selenium WebDriver with configuration"""
        selenium_config = self.global_config.get('selenium', {})

        options = webdriver.ChromeOptions()

        # Configure options based on config
        if selenium_config.get('headless', True):
            options.add_argument("--headless=new")
        if selenium_config.get('disable_gpu', True):
            options.add_argument("--disable-gpu")
        if selenium_config.get('disable_dev_shm', True):
            options.add_argument("--disable-dev-shm-usage")
        if selenium_config.get('no_sandbox', True):
            options.add_argument("--no-sandbox")
        if selenium_config.get('disable_logging', True):
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            options.add_experimental_option('useAutomationExtension', False)

        # Window size
        window_size = selenium_config.get('window_size', '1920,1080')
        options.add_argument(f"--window-size={window_size}")

        # User agent
        user_agent = selenium_config.get('user_agent')
        if user_agent:
            options.add_argument(f"--user-agent={user_agent}")

        try:
            driver = webdriver.Chrome(options=options)
            self.logger.info(" Selenium WebDriver initialized successfully")
            return driver
        except Exception as e:
            self.logger.error(f"Failed to initialize WebDriver: {e}")
            raise

    def _clean_price(self, price_text: str) -> Optional[str]:
        """Extract numeric value from price string"""
        if not price_text:
            return None
        return re.sub(r"[^\d]", "", price_text.strip()) or None

    def _extract_product_data(self, product_element, base_url: str, scraped_at: str) -> Optional[ProductData]:
        """Extract product data from a product element"""
        try:
            # Get product ID
            product_id = product_element.get("data-asin")
            if not product_id:
                return None

            selectors = self.scraper_config['selectors']

            # Extract title
            title_element = None
            for title_class in selectors['title_classes']:
                title_element = product_element.find("span", {"class": title_class})
                if title_element:
                    break

            if not title_element and product_element.find("h2"):
                title_element = product_element.find("h2").find("span")

            product_name = title_element.get_text(strip=True) if title_element else "Unknown"

            # Extract price
            price_element = product_element.find("span", {"class": selectors['price_class']})
            price = self._clean_price(price_element.get_text() if price_element else None)

            # Extract MRP
            mrp_element = product_element.find("span", {"class": selectors['mrp_class']})
            mrp = mrp_element.get_text(strip=True).replace("₹", "").replace(",", "") if mrp_element else None

            # Extract rating
            rating_element = product_element.find("span", {"class": selectors['rating_class']})
            rating = rating_element.get_text(strip=True) if rating_element else None

            # Extract URL
            link_element = product_element.find("a", {"class": selectors['link_class']}, href=True)
            url = f"{base_url}{link_element['href']}" if link_element else ""

            return ProductData(
                source="amazon",
                product_id=product_id,
                name=product_name,
                price=price,
                mrp=mrp,
                discount=None,  # Amazon doesn't show discount percentage directly
                rating=rating,
                url=url,
                scraped_at=scraped_at
            )

        except Exception as e:
            self.logger.warning(f"Failed to extract product data: {e}")
            return None

    def _scrape_product_listings(self) -> Tuple[List[ProductData], Set[Tuple[str, str, str]]]:
        """Scrape product listings from Amazon"""
        products_data: List[ProductData] = []
        product_links: Set[Tuple[str, str, str]] = set()

        base_url = self.scraper_config['base_url']
        search_endpoint = self.scraper_config['search_endpoint']
        max_pages = self.scraper_config['max_listing_pages']
        wait_time = self.global_config.get('default_wait_time', 3)
        scraped_at = datetime.utcnow().isoformat()

        for page in range(1, max_pages + 1):
            try:
                self.logger.info(f" Scraping listing page {page}/{max_pages}")

                # Navigate to search page
                search_url = f"{base_url}{search_endpoint}".format(page)
                self.driver.get(search_url)
                time.sleep(wait_time)

                # Parse page content
                soup = BeautifulSoup(self.driver.page_source, "lxml")
                product_elements = soup.find_all("div", {"data-component-type": "s-search-result"})

                if not product_elements:
                    self.logger.warning(f"No products found on page {page}")
                    continue

                # Extract product data
                for element in product_elements:
                    product_data = self._extract_product_data(element, base_url, scraped_at)
                    if product_data:
                        products_data.append(product_data)
                        if product_data.url:
                            product_links.add((product_data.product_id, product_data.name, product_data.url))

                self.logger.info(f"Extracted {len(product_elements)} products from page {page}")

            except Exception as e:
                self.logger.error(f"Failed to scrape page {page}: {e}")
                continue

        self.logger.info(f"Total products scraped: {len(products_data)}")
        return products_data, product_links

    def _extract_review_data(self, review_element, product_id: str, product_name: str) -> Optional[ReviewData]:
        """Extract review data from a review element"""
        try:
            selectors = self.scraper_config['selectors']

            # Extract reviewer name
            reviewer_element = review_element.find("span", {"class": selectors['reviewer_name']})
            reviewer_name = reviewer_element.get_text(strip=True) if reviewer_element else "Anonymous"

            # Extract rating
            rating_element = review_element.find("i", {"data-hook": "review-star-rating"})
            rating = rating_element.get_text(strip=True) if rating_element else None

            # Extract review text
            text_element = review_element.find("span", {"data-hook": "review-body"})
            review_text = text_element.get_text(strip=True) if text_element else ""

            # Extract date
            date_element = review_element.find("span", {"data-hook": "review-date"})
            review_date = date_element.get_text(strip=True) if date_element else ""

            return ReviewData(
                source="amazon",
                product_id=product_id,
                product_name=product_name,
                user_id=reviewer_name,
                review_text=review_text,
                rating=rating,
                review_date=review_date
            )

        except Exception as e:
            self.logger.warning(f"Failed to extract review data: {e}")
            return None

    def _scrape_product_reviews(self, product_links: Set[Tuple[str, str, str]]) -> List[ReviewData]:
        """Scrape reviews for products"""
        reviews_data: List[ReviewData] = []
        max_review_pages = self.scraper_config['max_review_pages']
        timeout = self.global_config.get('request_timeout', 30)

        for product_id, product_name, product_url in product_links:
            if not product_url:
                continue

            try:
                self.logger.info(f" Scraping reviews for: {product_name[:50]}...")

                # Navigate to product page
                self.driver.get(product_url)
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.ID, "reviews-medley-footer"))
                )

                # Find reviews link
                soup = BeautifulSoup(self.driver.page_source, "lxml")
                reviews_link = soup.find("a", href=re.compile("/product-reviews/"))

                if not reviews_link:
                    self.logger.warning(f"No reviews link found for {product_name}")
                    continue

                reviews_base_url = f"{self.scraper_config['base_url']}{reviews_link['href']}"

                # Scrape review pages
                for page in range(1, max_review_pages + 1):
                    try:
                        review_page_url = f"{reviews_base_url}&pageNumber={page}"
                        self.driver.get(review_page_url)

                        WebDriverWait(self.driver, timeout).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "review"))
                        )

                        # Parse reviews
                        review_soup = BeautifulSoup(self.driver.page_source, "lxml")
                        review_elements = review_soup.find_all("div", {"data-hook": "review"})

                        if not review_elements:
                            break

                        # Extract review data
                        for review_element in review_elements:
                            review_data = self._extract_review_data(review_element, product_id, product_name)
                            if review_data:
                                reviews_data.append(review_data)

                        time.sleep(1)  # Rate limiting

                    except TimeoutException:
                        self.logger.warning(f"Timeout loading reviews page {page} for {product_name}")
                        break
                    except Exception as e:
                        self.logger.warning(f"Error scraping reviews page {page}: {e}")
                        break

            except Exception as e:
                self.logger.error(f"Failed to scrape reviews for {product_name}: {e}")
                continue

        self.logger.info(f"Total reviews scraped: {len(reviews_data)}")
        return reviews_data

    def _save_data_to_csv(self, data: List, file_path, subset_cols: List[str] = None) -> None:
        """Save data to CSV with deduplication"""
        if not data:
            self.logger.warning("No data to save")
            return

        # Convert to DataFrame
        if isinstance(data[0], (ProductData, ReviewData)):
            df_new = pd.DataFrame([vars(item) for item in data])
        else:
            df_new = pd.DataFrame(data)

        # Load existing data if file exists
        if file_path.exists():
            try:
                df_existing = pd.read_csv(file_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            except Exception as e:
                self.logger.warning(f"Error loading existing data: {e}")
                df_combined = df_new
        else:
            df_combined = df_new

        # Remove duplicates
        if subset_cols:
            df_combined = df_combined.drop_duplicates(subset=subset_cols, keep="last")

        # Save to file
        try:
            df_combined.to_csv(file_path, index=False, encoding="utf-8-sig")
            self.logger.info(f" Saved {len(df_new)} new rows (total {len(df_combined)}) → {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save data to {file_path}: {e}")
            raise

    def run(self) -> bool:
        """Main scraper execution method"""
        try:
            # Setup driver
            self.driver = self._setup_selenium_driver()

            # Scrape product listings
            self.logger.info("Starting Amazon product listing scraping...")
            products_data, product_links = self._scrape_product_listings()

            if not products_data:
                self.logger.warning("No products scraped")
                return False

            # Save product data
            self._save_data_to_csv(
                products_data, 
                self.paths.raw_mobile_file,
                subset_cols=["product_id", "scraped_at"]
            )

            # Scrape reviews
            self.logger.info("Starting Amazon review scraping...")
            reviews_data = self._scrape_product_reviews(product_links)

            if reviews_data:
                # Save review data
                self._save_data_to_csv(
                    reviews_data,
                    self.paths.raw_reviews_file,
                    subset_cols=["product_id", "user_id", "review_text"]
                )
            else:
                self.logger.warning("No reviews scraped")

            self.logger.info(" Amazon scraping completed successfully")
            return True

        except Exception as e:
            self.logger.error(f" Amazon scraping failed: {e}")
            return False
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                    self.logger.info("WebDriver closed successfully")
                except Exception as e:
                    self.logger.warning(f"Error closing WebDriver: {e}")

def run_scraper(scraper_config: Dict[str, Any], paths: PathConfig) -> bool:
    """Entry point for running Amazon scraper"""
    try:
        scraper = AmazonScraper()
        return scraper.run()
    except Exception as e:
        print(f"Amazon scraper failed: {e}")
        return False

if __name__ == "__main__":
    # For testing
    from config_manager import get_config_manager
    config_manager = get_config_manager()
    scraper_config = config_manager.get_scraper_config('amazon')
    paths = config_manager.paths
    run_scraper(scraper_config, paths)
