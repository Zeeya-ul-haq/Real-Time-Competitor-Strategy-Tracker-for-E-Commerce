"""
Professional Data Processing and Cleaning System
Configuration-driven data validation, cleaning, and transformation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

from config_manager import BaseComponent, PathConfig

@dataclass
class DataQualityReport:
    """Data structure for data quality metrics"""
    total_records: int
    valid_records: int
    duplicate_records: int
    missing_values: Dict[str, int]
    outliers_removed: int
    data_quality_score: float

class DataProcessor(BaseComponent):
    """
    Professional data processing system with comprehensive validation and cleaning
    """

    def __init__(self):
        super().__init__("data_processor")
        self.processing_config = self.get_component_config('data_processing')
        self.validation_config = self.processing_config.get('validation', {})
        self.cleaning_config = self.processing_config.get('cleaning', {})
        self.text_config = self.processing_config.get('text_processing', {})

        # Quality tracking
        self.quality_reports: Dict[str, DataQualityReport] = {}

    def _parse_relative_date(self, date_text: Union[str, pd.Timestamp]) -> Optional[datetime]:
        """Convert relative dates to absolute dates with robust parsing"""
        if pd.isna(date_text):
            return None

        text = str(date_text).strip().lower()
        now = datetime.now()

        try:
            # Try parsing as standard date first
            if not any(word in text for word in ['day', 'month', 'year', 'ago']):
                parsed_date = pd.to_datetime(text, errors='coerce')
                if not pd.isna(parsed_date):
                    return parsed_date.to_pydatetime()

            # Handle relative dates
            number_match = re.search(r'(\d+)', text)
            if not number_match:
                return None

            number = int(number_match.group(1))

            if 'day' in text:
                return now - timedelta(days=number)
            elif 'month' in text:
                # Approximate months as 30 days
                return now - timedelta(days=number * 30)
            elif 'year' in text:
                return now - timedelta(days=number * 365)
            else:
                return None

        except Exception as e:
            self.logger.warning(f"Failed to parse date '{date_text}': {e}")
            return None

    def _clean_text(self, text: str) -> str:
        """Clean text data based on configuration"""
        if not isinstance(text, str):
            return str(text) if text is not None else ""

        cleaned_text = text

        # Lowercase
        if self.text_config.get('lowercase', True):
            cleaned_text = cleaned_text.lower()

        # Remove special characters
        if self.text_config.get('remove_special_chars', True):
            cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'"()\-]', '', cleaned_text)

        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text.strip())

        return cleaned_text

    def _extract_brand(self, product_name: str) -> str:
        """Extract brand name from product name"""
        if not isinstance(product_name, str):
            return "Unknown"

        # Common brand patterns
        brand_patterns = [
            r'^(apple|iphone)',
            r'^(samsung)',
            r'^(oneplus|one\s*plus)',
            r'^(xiaomi|mi|redmi)',
            r'^(oppo)',
            r'^(vivo)',
            r'^(realme)',
            r'^(huawei)',
            r'^(honor)',
            r'^(motorola|moto)',
            r'^(nokia)',
            r'^(nothing)',
            r'^(google|pixel)'
        ]

        product_lower = product_name.lower().strip()

        for pattern in brand_patterns:
            match = re.search(pattern, product_lower)
            if match:
                return match.group(1).title()

        # Fallback: first word
        first_word = product_lower.split()[0] if product_lower.split() else "Unknown"
        return first_word.title()

    def _detect_outliers(self, series: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
        """Detect outliers using specified method"""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (series < lower_bound) | (series > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold

        elif method == 'percentile':
            lower_percentile = (1 - threshold) / 2
            upper_percentile = 1 - lower_percentile
            lower_bound = series.quantile(lower_percentile)
            upper_bound = series.quantile(upper_percentile)
            return (series < lower_bound) | (series > upper_bound)

        else:
            return pd.Series([False] * len(series), index=series.index)

    def _add_sentiment_analysis(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Add sentiment analysis if TextBlob is available"""
        if not TEXTBLOB_AVAILABLE or text_column not in df.columns:
            return df

        try:
            self.logger.info("Adding sentiment analysis...")

            sentiments = []
            polarities = []

            for text in df[text_column]:
                try:
                    if pd.isna(text) or str(text).strip() == "":
                        sentiments.append('neutral')
                        polarities.append(0.0)
                    else:
                        blob = TextBlob(str(text))
                        polarity = blob.sentiment.polarity

                        if polarity > 0.1:
                            sentiment = 'positive'
                        elif polarity < -0.1:
                            sentiment = 'negative'
                        else:
                            sentiment = 'neutral'

                        sentiments.append(sentiment)
                        polarities.append(polarity)

                except Exception:
                    sentiments.append('neutral')
                    polarities.append(0.0)

            df['sentiment'] = sentiments
            df['sentiment_polarity'] = polarities

            self.logger.info(f"Added sentiment analysis for {len(df)} records")

        except Exception as e:
            self.logger.warning(f"Sentiment analysis failed: {e}")

        return df

    def _validate_data(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Validate data against configuration rules"""
        initial_count = len(df)
        issues = {}

        # Required fields validation
        required_fields = self.validation_config.get('required_fields', [])
        missing_required = [col for col in required_fields if col not in df.columns]

        if missing_required:
            self.logger.warning(f"Missing required fields: {missing_required}")

        # Price validation
        if 'sellingprice' in df.columns:
            price_col = 'sellingprice'
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

            min_price = self.validation_config.get('min_price', 0)
            max_price = self.validation_config.get('max_price', float('inf'))

            price_mask = (df[price_col] >= min_price) & (df[price_col] <= max_price)
            invalid_prices = (~price_mask & df[price_col].notna()).sum()

            if invalid_prices > 0:
                df = df[price_mask | df[price_col].isna()]
                issues['invalid_prices'] = invalid_prices
                self.logger.info(f"Removed {invalid_prices} records with invalid prices")

        # Rating validation
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

            min_rating = self.validation_config.get('min_rating', 0.0)
            max_rating = self.validation_config.get('max_rating', 5.0)

            rating_mask = (df['rating'] >= min_rating) & (df['rating'] <= max_rating)
            invalid_ratings = (~rating_mask & df['rating'].notna()).sum()

            if invalid_ratings > 0:
                df.loc[~rating_mask, 'rating'] = np.nan
                issues['invalid_ratings'] = invalid_ratings
                self.logger.info(f"Set {invalid_ratings} invalid ratings to NaN")

        # Missing values analysis
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: v for k, v in missing_values.items() if v > 0}

        # Calculate data quality score
        valid_count = len(df)
        quality_score = (valid_count / initial_count) * 100 if initial_count > 0 else 0

        quality_report = DataQualityReport(
            total_records=initial_count,
            valid_records=valid_count,
            duplicate_records=0,  # Will be updated after deduplication
            missing_values=missing_values,
            outliers_removed=0,  # Will be updated after outlier removal
            data_quality_score=quality_score
        )

        return df, quality_report

    def _clean_mobile_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean mobile/product data"""
        self.logger.info("Cleaning mobile/product data...")

        # Standardize column names
        column_mapping = {
            'mobilename': 'product_name',
            'sellingprice': 'price',
            'discountoffering': 'discount',
            'productid': 'product_id'
        }

        df = df.rename(columns=column_mapping)

        # Clean text fields
        text_fields = ['product_name', 'source']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].astype(str).apply(self._clean_text)

        # Extract brand if configured
        if self.text_config.get('extract_brands', True) and 'product_name' in df.columns:
            df['brand'] = df['product_name'].apply(self._extract_brand)

        # Clean numeric fields
        numeric_fields = ['price', 'rating', 'discount']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')

        # Clean discount field specifically
        if 'discount' in df.columns:
            # Remove percentage signs and other non-numeric characters
            df['discount'] = df['discount'].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df['discount'] = pd.to_numeric(df['discount'], errors='coerce')

        # Parse dates
        if self.cleaning_config.get('parse_dates', True) and 'scraped_at' in df.columns:
            df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')

        return df

    def _clean_reviews_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean reviews data"""
        self.logger.info("Cleaning reviews data...")

        # Standardize column names
        column_mapping = {
            'mobilename': 'product_name',
            'review': 'review_text',
            'userid': 'user_id',
            'reviewdate': 'review_date',
            'productid': 'product_id'
        }

        df = df.rename(columns=column_mapping)

        # Clean text fields
        text_fields = ['product_name', 'user_id', 'review_text']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].astype(str).apply(self._clean_text)

        # Remove empty reviews
        if 'review_text' in df.columns:
            df = df[df['review_text'].str.strip().str.len() > 0]

        # Clean rating
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        # Parse review dates
        if self.cleaning_config.get('parse_dates', True) and 'review_date' in df.columns:
            df['review_date'] = df['review_date'].apply(self._parse_relative_date)

        # Add sentiment analysis if configured
        if (self.text_config.get('sentiment_analysis', True) and 
            'review_text' in df.columns and 
            TEXTBLOB_AVAILABLE):
            df = self._add_sentiment_analysis(df, 'review_text')

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Remove outliers based on configuration"""
        if not self.cleaning_config.get('outlier_removal', {}).get('enabled', True):
            return df, 0

        outlier_config = self.cleaning_config['outlier_removal']
        method = outlier_config.get('method', 'iqr')
        threshold = outlier_config.get('threshold', 1.5)

        initial_count = len(df)
        outlier_mask = pd.Series([False] * len(df), index=df.index)

        # Check price outliers
        if 'price' in df.columns:
            price_outliers = self._detect_outliers(df['price'].dropna(), method, threshold)
            outlier_mask = outlier_mask | price_outliers.reindex(df.index, fill_value=False)

        # Remove outliers
        df_clean = df[~outlier_mask]
        outliers_removed = initial_count - len(df_clean)

        if outliers_removed > 0:
            self.logger.info(f"Removed {outliers_removed} outliers using {method} method")

        return df_clean, outliers_removed

    def _remove_duplicates(self, df: pd.DataFrame, subset_cols: List[str]) -> Tuple[pd.DataFrame, int]:
        """Remove duplicates based on subset columns"""
        if not self.cleaning_config.get('remove_duplicates', True):
            return df, 0

        initial_count = len(df)

        # Filter subset_cols to only include existing columns
        existing_subset_cols = [col for col in subset_cols if col in df.columns]

        if existing_subset_cols:
            df_clean = df.drop_duplicates(subset=existing_subset_cols, keep='last')
            duplicates_removed = initial_count - len(df_clean)

            if duplicates_removed > 0:
                self.logger.info(f"Removed {duplicates_removed} duplicate records")
        else:
            df_clean = df
            duplicates_removed = 0

        return df_clean, duplicates_removed

    def process_mobile_data(self) -> bool:
        """Process mobile/product data"""
        try:
            input_file = self.paths.raw_mobile_file
            output_file = self.paths.processed_mobile_file

            if not input_file.exists():
                self.logger.warning(f"Mobile data file not found: {input_file}")
                return False

            # Load data
            df = pd.read_csv(input_file)
            self.logger.info(f"Loaded {len(df)} mobile records from {input_file}")

            # Validate data
            df, quality_report = self._validate_data(df, 'mobile')

            # Clean data
            df = self._clean_mobile_data(df)

            # Remove duplicates
            df, duplicates = self._remove_duplicates(df, ['product_id', 'scraped_at'])
            quality_report.duplicate_records = duplicates

            # Remove outliers
            df, outliers = self._remove_outliers(df)
            quality_report.outliers_removed = outliers

            # Update quality report
            quality_report.valid_records = len(df)
            quality_report.data_quality_score = (quality_report.valid_records / 
                                               quality_report.total_records) * 100

            self.quality_reports['mobile'] = quality_report

            # Save processed data
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"✅ Processed mobile data saved: {output_file}")
            self.logger.info(f"   Final records: {len(df)}")
            self.logger.info(f"   Data quality: {quality_report.data_quality_score:.1f}%")

            return True

        except Exception as e:
            self.logger.error(f"Failed to process mobile data: {e}")
            return False

    def process_reviews_data(self) -> bool:
        """Process reviews data"""
        try:
            input_file = self.paths.raw_reviews_file
            output_file = self.paths.processed_reviews_file

            if not input_file.exists():
                self.logger.warning(f"Reviews data file not found: {input_file}")
                return False

            # Load data
            df = pd.read_csv(input_file)
            self.logger.info(f"Loaded {len(df)} review records from {input_file}")

            # Load valid product IDs from processed mobile data
            valid_product_ids = set()
            if self.paths.processed_mobile_file.exists():
                mobile_df = pd.read_csv(self.paths.processed_mobile_file)
                valid_product_ids = set(mobile_df['product_id'].dropna().unique())
                self.logger.info(f"Loaded {len(valid_product_ids)} valid product IDs for filtering")

            # Filter by valid product IDs
            if valid_product_ids:
                initial_count = len(df)
                df = df[df['productid'].isin(valid_product_ids)]
                filtered_count = initial_count - len(df)
                if filtered_count > 0:
                    self.logger.info(f"Filtered out {filtered_count} reviews for invalid products")

            # Validate data
            df, quality_report = self._validate_data(df, 'reviews')

            # Clean data
            df = self._clean_reviews_data(df)

            # Remove duplicates
            df, duplicates = self._remove_duplicates(df, ['product_id', 'user_id', 'review_text'])
            quality_report.duplicate_records = duplicates

            # Update quality report
            quality_report.valid_records = len(df)
            quality_report.data_quality_score = (quality_report.valid_records / 
                                               quality_report.total_records) * 100

            self.quality_reports['reviews'] = quality_report

            # Save processed data
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            self.logger.info(f"✅ Processed reviews data saved: {output_file}")
            self.logger.info(f"   Final records: {len(df)}")
            self.logger.info(f"   Data quality: {quality_report.data_quality_score:.1f}%")

            return True

        except Exception as e:
            self.logger.error(f"Failed to process reviews data: {e}")
            return False

    def generate_quality_report(self) -> None:
        """Generate comprehensive data quality report"""
        try:
            if not self.quality_reports:
                self.logger.warning("No quality reports available")
                return

            report_lines = ["\n" + "="*80, "📊 DATA QUALITY REPORT", "="*80]

            for data_type, report in self.quality_reports.items():
                report_lines.extend([
                    f"\n{data_type.upper()} DATA:",
                    f"  Total Records: {report.total_records:,}",
                    f"  Valid Records: {report.valid_records:,}",
                    f"  Duplicates Removed: {report.duplicate_records:,}",
                    f"  Outliers Removed: {report.outliers_removed:,}",
                    f"  Data Quality Score: {report.data_quality_score:.1f}%"
                ])

                if report.missing_values:
                    report_lines.append("  Missing Values:")
                    for col, count in report.missing_values.items():
                        percentage = (count / report.total_records) * 100
                        report_lines.append(f"    {col}: {count:,} ({percentage:.1f}%)")

            report_lines.append("="*80)

            # Print report
            for line in report_lines:
                print(line)

            # Save report to file
            report_file = self.paths.output_dir / 'data_quality_report.txt'
            with open(report_file, 'w') as f:
                f.write('\n'.join(report_lines))

            self.logger.info(f"Quality report saved to {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to generate quality report: {e}")

    def run(self) -> bool:
        """Main data processing pipeline"""
        try:
            self.logger.info("Starting data processing pipeline...")

            success = True

            # Process mobile data
            if not self.process_mobile_data():
                success = False

            # Process reviews data
            if not self.process_reviews_data():
                success = False

            # Generate quality report
            self.generate_quality_report()

            if success:
                self.logger.info("✅ Data processing completed successfully")
            else:
                self.logger.warning("⚠️ Data processing completed with some failures")

            return success

        except Exception as e:
            self.logger.error(f"Data processing pipeline failed: {e}")
            return False

def run_processing(processing_config: Dict[str, Any], paths: PathConfig) -> bool:
    """Entry point for running data processing"""
    try:
        processor = DataProcessor()
        return processor.run()
    except Exception as e:
        print(f"Data processing failed: {e}")
        return False

if __name__ == "__main__":
    # For testing
    from config_manager import get_config_manager
    config_manager = get_config_manager()
    processing_config = config_manager.get('data_processing')
    paths = config_manager.paths
    run_processing(processing_config, paths)
