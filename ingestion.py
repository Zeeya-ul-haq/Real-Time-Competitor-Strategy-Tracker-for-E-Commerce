"""
Production-ready Flipkart mobile data and review scraper.
Now fully configurable via config.yaml
"""

import os
import re
import time
import logging
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

class FlipkartScraper:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_selenium()
        self.setup_output_directory()

    def setup_selenium(self):
        """Setup Selenium WebDriver with configuration"""
        selenium_config = self.config.get('selenium', {})

        options = webdriver.ChromeOptions()
        if selenium_config.get('headless', True):
            options.add_argument("--headless=new")
        if selenium_config.get('disable_gpu', True):
            options.add_argument("--disable-gpu")
        if selenium_config.get('disable_logging', True):
            options.add_experimental_option('excludeSwitches', ['enable-logging'])

        self.driver = webdriver.Chrome(options=options)
        self.logger.info("✅ Selenium WebDriver initialized")

    def setup_output_directory(self):
        """Create output directory"""
        self.output_dir = "My_docs"
        os.makedirs(self.output_dir, exist_ok=True)

    def clean_price(self, txt):
        """Extract digits from price string."""
        return re.sub(r"[^\d]", "", txt) if txt else None

    def save_csv(self, df_new, path, subset_cols):
        """Append to CSV if exists, drop duplicates by subset_cols."""
        if os.path.exists(path):
            df_old = pd.read_csv(path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        if subset_cols:
            df = df.drop_duplicates(subset=subset_cols, keep="last")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        self.logger.info(f"✅ Saved {len(df_new)} new rows (total {len(df)}) → {path}")

    def scrape_product_listings(self):
        """Scrape product listings from Flipkart"""
        mobile_rows = []
        product_links = set()
        scraped_at = datetime.utcnow().isoformat()

        search_url = self.config['search_url']
        listing_pages = self.config['listing_pages']
        wait_time = self.config['wait_time']

        for page in range(1, listing_pages + 1):
            self.logger.info(f"📄 Scraping listing page {page}")
            self.driver.get(search_url.format(page))
            time.sleep(wait_time)
            soup = BeautifulSoup(self.driver.page_source, "lxml")

            # Use configurable selectors
            selectors = self.config.get('selectors', {})
            products = soup.find_all("div", {"class": "tUxRFH"})

            for p in products:
                title = p.find("div", {"class": selectors.get('title_class', 'KzDlHZ')})
                price = p.find("div", {"class": selectors.get('price_class', 'Nx9bqj _4b5DiR')})
                mrp = p.find("div", {"class": selectors.get('mrp_class', 'yRaY8j')})
                discount = p.find("div", {"class": selectors.get('discount_class', 'UkUFwK')})
                rating = p.find("div", {"class": selectors.get('rating_class', 'XQDdHH')})
                link = p.find("a", {"class": selectors.get('link_class', 'CGtC98')})

                mobilename = title.get_text(strip=True) if title else "Unknown"
                sellingprice = self.clean_price(price.get_text()) if price else None
                mrp_val = mrp.get_text(strip=True).replace("₹", "").replace(",", "") if mrp else None
                discountoffering = discount.get_text(strip=True) if discount else None
                rating_val = rating.get_text(strip=True) if rating else None
                url = "https://www.flipkart.com" + link["href"] if link else None

                pid = None
                if url:
                    m = re.search(r"/p/itm([0-9a-z]+)", url)
                    pid = m.group(1) if m else None
                    product_links.add((pid, mobilename, url))

                mobile_rows.append({
                    "source": "flipkart",
                    "productid": pid,
                    "mobilename": mobilename,
                    "sellingprice": sellingprice,
                    "mrp": mrp_val,
                    "discountoffering": discountoffering,
                    "rating": rating_val,
                    "url": url,
                    "scraped_at": scraped_at
                })

        return mobile_rows, product_links

    def scrape_reviews(self, product_links):
        """Scrape reviews for products"""
        review_rows = []
        review_pages = self.config['review_pages']

        for pid, name, url in product_links:
            if not url:
                continue
            try:
                self.driver.get(url)
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "VU-ZEz"))
                )
                soup = BeautifulSoup(self.driver.page_source, "lxml")
                all_reviews = soup.find("a", href=re.compile("/product-reviews/"))
                if not all_reviews:
                    continue

                reviews_base = "https://www.flipkart.com" + all_reviews["href"]
                self.logger.info(f"💬 Scraping reviews for {name}")

                for rpage in range(1, review_pages + 1):
                    self.driver.get(f"{reviews_base}&page={rpage}")
                    try:
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CLASS_NAME, "cPHDOP"))
                        )
                    except TimeoutException:
                        break

                    rsoup = BeautifulSoup(self.driver.page_source, "lxml")
                    containers = rsoup.find_all("div", {"class": "cPHDOP"})

                    for c in containers:
                        user = c.find("p", {"class": "_2NsDsF AwS1CA"})
                        rating = c.find("div", {"class": "_3LWZlK"})
                        text = c.find("div", {"class": "ZmyHeo"})
                        all_p = c.find_all("p", {"class": "_2NsDsF"})
                        date = all_p[-1].get_text(strip=True) if len(all_p) > 1 else ""

                        review_rows.append({
                            "source": "flipkart",
                            "productid": pid,
                            "mobilename": name,
                            "userid": user.get_text(strip=True) if user else "Anonymous",
                            "review": text.get_text(strip=True).replace("READ MORE", "") if text else "",
                            "rating": rating.get_text(strip=True) if rating else None,
                            "reviewdate": date
                        })
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"⚠ Error scraping {name}: {e}")

        return review_rows

    def run_scraping(self):
        """Main scraping method"""
        try:
            # Scrape product listings
            self.logger.info("Starting Flipkart product scraping...")
            mobile_rows, product_links = self.scrape_product_listings()

            # Save product data
            mobile_df = pd.DataFrame(mobile_rows)
            mobile_path = os.path.join(self.output_dir, self.config['output_file'])
            self.save_csv(mobile_df, mobile_path, ["productid", "scraped_at"])

            # Scrape reviews
            self.logger.info("Starting Flipkart review scraping...")
            review_rows = self.scrape_reviews(product_links)

            # Save review data
            review_df = pd.DataFrame(review_rows)
            review_path = os.path.join(self.output_dir, self.config['review_file'])
            self.save_csv(review_df, review_path, ["productid", "userid", "review"])

            self.logger.info("✅ Flipkart scraping completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Flipkart scraping failed: {e}")
            return False
        finally:
            self.driver.quit()

def run_scraper(config):
    """Function to run Flipkart scraper with config"""
    scraper = FlipkartScraper(config)
    return scraper.run_scraping()

if __name__ == "__main__":
    # For testing
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    run_scraper(config['flipkart_scraper'])
