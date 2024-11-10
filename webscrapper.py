import os
import time
from typing import List, Optional, Dict, Set, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
from threading import Lock

# Configuration flags - Processing
RELOAD_EVENT_LIST = False  # Set to True to force reload of ufc_stats.html
RELOAD_EVENT_HTML = False  # Set to True to force reload of all event HTML files
RELOAD_EVENT_CSV = True  # Set to True to force reload of all event CSV files (reparse HTML)  # noqa: E501
EVENT_LIMIT = None  # Set to a number to limit how many events to process (None for all events)  # noqa: E501

RELOAD_FIGHTER_LIST = False  # Set to True to force reload of ufc_stats.html
RELOAD_FIGHTER_HTML = False  # Set to True to force reload of all event HTML files
RELOAD_FIGHTER_CSV = False  # Set to True to force reload of all event CSV files (reparse HTML)  # noqa: E501
FIGHTER_LIMIT = None  # Set to a number to limit how many events to process (None for all events)  # noqa: E501

# Configuration flags - Timeouts and Delays
REQUEST_TIMEOUT = 5  # 5 seconds is usually enough for the page to load
MAX_RETRIES = 2  # 2 retries should be sufficient for most cases
RETRY_DELAY = 1  # 1 second between retries is minimal but acceptable
BETWEEN_URL_DELAY = 1  # 1 second between events should be polite enough

# Headers to mimic a browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"  # noqa: E501
}

# Add new configuration flags
MAX_WORKERS = 5  # Number of parallel workers for processing
PARALLEL_PROCESSING = True  # Enable/disable parallel processing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HTMLFileHandler:
    """Utility class to handle HTML file operations."""

    def __init__(self, base_folder: str = "data"):
        # Setup folder structure
        self.base_folder = base_folder
        self.html_folder = os.path.join(base_folder, "html")
        self.html_events_folder = os.path.join(self.html_folder, "events")
        self.html_fighters_folder = os.path.join(self.html_folder, "fighters")
        self.html_event_list_folder = os.path.join(self.html_folder, "event_list")
        self.html_fighter_list_folder = os.path.join(self.html_folder, "fighter_list")

        # Create all required folders
        for folder in [
            self.html_folder,
            self.html_events_folder,
            self.html_fighters_folder,
            self.html_event_list_folder,
            self.html_fighter_list_folder,
        ]:
            os.makedirs(folder, exist_ok=True)

    def save_html(self, html_content: str, file_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as file:
                soup = BeautifulSoup(html_content, "html.parser")
                file.write(soup.prettify())
        except OSError as e:
            logging.error(f"OS error saving HTML to {file_path}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error saving HTML to {file_path}: {str(e)}")
            raise

    def load_html(self, file_path: str) -> str:
        """Load HTML content from a file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            logging.error(f"Error loading HTML from {file_path}: {str(e)}")
            raise


class WebRequester:
    """Utility class to handle web requests."""

    def get_page_with_retry(
        self, url: str, max_retries: int = MAX_RETRIES, timeout: int = REQUEST_TIMEOUT
    ) -> str:
        """Get page content with retry mechanism."""
        for retry in range(max_retries):
            try:
                response = requests.get(url, headers=HEADERS, timeout=timeout)
                response.raise_for_status()  # Raise an exception for bad status codes
                return response.text
            except requests.ConnectionError:
                logging.error(f"Connection error on attempt {retry + 1}/{max_retries}")
            except requests.Timeout:
                logging.error(f"Timeout error on attempt {retry + 1}/{max_retries}")
            except requests.RequestException as e:
                logging.error(f"Request failed on attempt {retry + 1}/{max_retries}: {str(e)}")
            
            if retry < max_retries - 1:
                time.sleep(RETRY_DELAY * (retry + 1))
        
        raise requests.RequestException(f"Failed to get {url} after {max_retries} attempts")


class URLCollector:
    def __init__(self, page: str):
        if page not in ["events", "fighters"]:
            raise ValueError("Page must be either 'events' or 'fighters'")
        self.data_folder = "data"
        self.page = page
        if page == "events":
            self.base_url = "http://ufcstats.com/statistics/events/completed?page=all"
            self.file_path = os.path.join(
                self.data_folder, "html", "event_list", "ufc_events.html"
            )
        elif page == "fighters":
            self.file_path = os.path.join(
                self.data_folder, "html", "fighter_list", "ufc_fighters.html"
            )

        self.urls: List[str] = []
        self.html_handler = HTMLFileHandler(self.data_folder)
        self.csv_handler = CSVFileHandler(self.data_folder)
        self.web_requester = WebRequester()
        self._lock = Lock()  # Add lock for thread safety

    def save_urls_to_csv(self, urls: List[str], filename: str) -> None:
        """Save URLs to a CSV file in a dedicated subfolder."""
        # Create subfolder for URLs
        urls_folder = os.path.join(self.csv_handler.csv_folder, "urls")
        os.makedirs(urls_folder, exist_ok=True)
        
        # Convert list of URLs to DataFrame
        df = pd.DataFrame(urls, columns=['url'])
        
        # Create file path in the urls subfolder
        file_path = os.path.join(urls_folder, filename)
        
        # Save using existing CSV handler
        self.csv_handler.save(df, file_path)
        logging.info(f"Saved {len(urls)} URLs to {file_path}")

    def _scrape_page(self) -> str:
        """Scrape the UFC stats page."""
        logging.info("Scraping UFC events from website...")
        return self.web_requester.get_page_with_retry(self.base_url)

    def _extract_urls(self, html_content: str) -> List[str]:
        soup = BeautifulSoup(html_content, "html.parser")
        urls = []
        if self.page == "events":
            for link in soup.find_all("a", href=True):
                if link["href"].startswith("http://ufcstats.com/event-details"):
                    urls.append(link["href"])
        elif self.page == "fighters":
            for link in soup.find_all("a", href=True):
                if "ufcstats.com/fighter-details/" in link["href"]:
                    urls.append(link["href"])

        # Remove duplicates while preserving order using dict.fromkeys()
        return list(dict.fromkeys(urls))

    def refresh_list(self) -> List[str]:
        """Force redownload of the UFC event list."""
        logging.info("Forcing redownload of UFC event list...")
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        return self.get_urls()

    def get_urls(self) -> List[str]:
        if os.path.exists(self.file_path):
            html_content = self.html_handler.load_html(self.file_path)
        else:
            html_content = self._scrape_page()
            self.html_handler.save_html(html_content, self.file_path)

        if self.page == "events":
            self.urls = self._extract_urls(html_content)
            logging.info(f"Found {len(self.urls)} event URLs")
        elif self.page == "fighters":
            self.urls = self._extract_urls(html_content)
            logging.info(f"Found {len(self.urls)} fighters URLs")
        return self.urls

    def get_fighter_urls_from_events(self, event_urls: List[str]) -> List[str]:
        fighter_urls = set()
        for event_url in event_urls:
            try:
                logging.info(f'Processing {event_url}')
                html_content = self.web_requester.get_page_with_retry(event_url)
                soup = BeautifulSoup(html_content, "html.parser")
                for link in soup.find_all("a", class_="b-link b-link_style_black", href=True):
                    if "ufcstats.com/fighter-details/" in link["href"]:
                        fighter_urls.add(link["href"])
            except Exception as e:
                logging.error(f"Error processing {event_url}: {e}")
                continue

        # Save URLs to CSV
        urls_list = list(fighter_urls)
        self.save_urls_to_csv(urls_list, "fighter_urls.csv")
        
        return urls_list


class CSVFileHandler:
    """Utility class to handle CSV file operations."""

    def __init__(self, base_folder: str = "data"):
        self.csv_folder = os.path.join(base_folder, "csv")
        self.csv_events_folder = os.path.join(self.csv_folder, "events")
        self.csv_fighters_folder = os.path.join(self.csv_folder, "fighters")

        # Create folders if they don't exist
        for folder in [
            self.csv_folder,
            self.csv_events_folder,
            self.csv_fighters_folder,
        ]:
            os.makedirs(folder, exist_ok=True)

    def exists(self, file_path: str) -> bool:
        """Check if CSV file exists."""
        return os.path.exists(file_path)

    def load(self, file_path: str) -> pd.DataFrame:
        """Load existing CSV file."""
        try:
            logging.info(f"Loading existing CSV file: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error loading CSV file: {str(e)}")
            raise

    def save(self, df: pd.DataFrame, file_path: str) -> None:
        """Save DataFrame to CSV file."""
        try:
            df.to_csv(file_path, index=False)
            logging.info(f"Successfully saved data to CSV: {file_path}")
        except Exception as e:
            logging.error(f"Error saving to CSV: {str(e)}")
            raise


class DataScrapper:
    def __init__(self, url: str, page_type: str):
        self.data_folder = "data"
        self.url = url
        self.page_type = page_type  # 'events' or 'fighters'
        self.soup: Optional[BeautifulSoup] = None

        # Use both handlers
        self.html_handler = HTMLFileHandler(self.data_folder)
        self.csv_handler = CSVFileHandler(self.data_folder)
        self.web_requester = WebRequester()

        # Setup file paths
        self.id = urlparse(url).path.split("/")[-1]
        # Use html_handler for HTML paths
        self.html_path = os.path.join(
            getattr(self.html_handler, f"html_{page_type}_folder"),
            f"{page_type}_{self.id}.html",
        )
        # Use csv_handler for CSV paths
        self.csv_path = os.path.join(
            getattr(self.csv_handler, f"csv_{page_type}_folder"),
            f"{page_type}_{self.id}.csv",
        )

    def get_data(self) -> BeautifulSoup:
        """Get data with caching mechanism."""
        try:
            if os.path.exists(self.html_path):
                html_content = self.html_handler.load_html(self.html_path)
            else:
                logging.info(f"Scraping {self.page_type} data from {self.url}")
                html_content = self.web_requester.get_page_with_retry(self.url)
                self.html_handler.save_html(html_content, self.html_path)

            self.soup = BeautifulSoup(html_content, "html.parser")
            return self.soup

        except Exception as e:
            logging.error(f"Error getting {self.page_type} data: {str(e)}")
            raise

    def event_extract_data(self) -> pd.DataFrame:
        """
        Extract all fight data from an event page, including event details.
        
        Returns:
            pd.DataFrame: DataFrame containing fight data with columns:
                - Event_Date: str
                - Event_Location: str
                - Fighter_A: str
                - Fighter_B: str
                ...
        
        Raises:
            RuntimeError: If BeautifulSoup object is not initialized
            Exception: If there's an error processing fight data
        """
        if self.soup is None:
            self.get_data()

        if self.soup is None:  # Double check after get_data
            raise RuntimeError("Failed to initialize BeautifulSoup object")

        data_l = []  # List to store multiple fights
        data_row_d = {}  # Dictionary for each element

        # Find all list items by class name
        list_items = self.soup.find_all('li', class_='b-list__box-list-item')

        # Loop through list items to find date and location based on preceding <i> tag text
        for item in list_items:
            title = item.find('i').get_text(strip=True)
            text = item.get_text(strip=True).replace(title, '').strip()  # Remove the title part

            if title == 'Date:':
                event_date = text
            elif title == 'Location:':
                event_location = text

        # Get all fight rows
        fight_rows = self.soup.select("table.b-fight-details__table tbody tr")

        # Process each fight
        for row in fight_rows:
            try:
                data_row_d = {}  # Dictionary for each element
                columns = row.select("td")

                if not columns:  # Skip rows without data
                    continue

                # Add event date and location to each fight
                data_row_d["Event_Date"] = event_date
                data_row_d["Event_Location"] = event_location
                
                # Extract each required field
                data_row_d["W_L"] = columns[0].get_text(strip=True)
                data_row_d["Fighter_A"] = columns[1].select("p")[0].get_text(strip=True)
                data_row_d["Fighter_B"] = columns[1].select("p")[1].get_text(strip=True)
                data_row_d["KD_A"] = columns[2].select("p")[0].get_text(strip=True)
                data_row_d["KD_B"] = columns[2].select("p")[1].get_text(strip=True)
                data_row_d["STR_A"] = columns[3].select("p")[0].get_text(strip=True)
                data_row_d["STR_B"] = columns[3].select("p")[1].get_text(strip=True)
                data_row_d["TD_A"] = columns[4].select("p")[0].get_text(strip=True)
                data_row_d["TD_B"] = columns[4].select("p")[1].get_text(strip=True)
                data_row_d["SUB_A"] = columns[5].select("p")[0].get_text(strip=True)
                data_row_d["SUB_B"] = columns[5].select("p")[1].get_text(strip=True)
                data_row_d["Weight_Class"] = columns[6].get_text(strip=True)

                method_text = columns[7].select("p")
                data_row_d["Method"] = method_text[0].get_text(strip=True)
                data_row_d["Method_Detail"] = (
                    method_text[1].get_text(strip=True) if len(method_text) > 1 else ""
                )
                data_row_d["Round"] = columns[8].get_text(strip=True)
                data_row_d["Time"] = columns[9].get_text(strip=True)
                data_l.append(data_row_d)

            except Exception as e:
                logging.error(f"Error processing fight row: {str(e)}")
                continue

        # Convert to DataFrame
        df = pd.DataFrame(data_l)

        return df

    def get_data_as_df(self) -> pd.DataFrame:
        """
        Get data as a DataFrame, either from existing CSV or by scraping.

        Returns:
            pd.DataFrame: DataFrame containing all fights data from the event
        """
        # Check if CSV already exists
        if self.csv_exists():
            logging.info(f"Found existing CSV for {self.page_type} {self.id}")
            return self.load_existing_csv()

        # If no CSV exists, scrape the data
        logging.info(f"No existing CSV found for {self.page_type} {self.id}, scraping data...")
        if self.page_type == "events":
            df = self.event_extract_data()
        elif self.page_type == "fighters":
            df = self.fight_extract_data()
        
        # Save the newly scraped data
        self.save_to_csv(df)

        return df

    def fight_extract_data(self) -> pd.DataFrame:
        """
        Extract fighter data from the UFC stats page and return it as a DataFrame.
        """
        try:
            if self.soup is None:
                self.get_data()

            if self.soup is None:
                raise RuntimeError("Failed to initialize BeautifulSoup object")

            fighter_data: dict[str, str] = {}
            
            try:
                # Extract basic info
                name_element = self.soup.find('span', class_='b-content__title-highlight')
                record_element = self.soup.find('span', class_='b-content__title-record')
                
                if not name_element or not record_element:
                    raise RuntimeError("Could not find fighter's basic information")
                    
                fighter_data['Name'] = name_element.text.strip()
                fighter_data['Record'] = record_element.text.strip().replace('Record: ', '')

                # Extract parameters
                info_box = self.soup.find('div', class_='b-list__info-box b-list__info-box_style_small-width js-guide')
                if info_box and isinstance(info_box, Tag):
                    for item in info_box.find_all('li', class_='b-list__box-list-item'):
                        if not isinstance(item, Tag):
                            continue
                        title = item.find('i', class_='b-list__box-item-title')
                        if title:
                            key = title.text.strip().replace(':', '')
                            value = item.text.replace(f"{key}:", '').strip()
                            fighter_data[key] = value

                # Extract career statistics from both columns
                for section in ['left', 'right']:
                    stats_div = self.soup.find('div', class_=f'b-list__info-box-{section}')
                    if stats_div and isinstance(stats_div, Tag):
                        for item in stats_div.find_all('li', class_='b-list__box-list-item'):
                            if not isinstance(item, Tag):
                                continue
                            title = item.find('i', class_='b-list__box-item-title')
                            if title:
                                key = title.text.strip().replace(':', '')
                                value = item.text.replace(f"{key}:", '').strip()
                                fighter_data[key] = value

                # Create DataFrame without any data processing
                fighter_df = pd.DataFrame([fighter_data])
                return fighter_df

            except Exception as e:
                logging.error(f"Error extracting fighter data: {str(e)}")
                raise

        except Exception as e:
            logging.error(f"Error in fight_get_data: {str(e)}")
            raise

    def csv_exists(self) -> bool:
        return self.csv_handler.exists(self.csv_path)

    def load_existing_csv(self) -> pd.DataFrame:
        return self.csv_handler.load(self.csv_path)

    def save_to_csv(self, df: pd.DataFrame) -> None:
        self.csv_handler.save(df, self.csv_path)

    def refresh_data(self) -> pd.DataFrame:
        """Force refresh  data by scraping website."""
        logging.info(f"Forcing refresh of event data from {self.url}")
        self.soup = None  # Reset soup to force new request
        if self.page_type == "events":
            df = self.event_extract_data()
        elif self.page_type == "fighters":
            df = self.fight_extract_data()
        self.save_to_csv(df)
        return df

    def reparse_data(self) -> pd.DataFrame:
        """Reparse existing HTML file to CSV."""
        logging.info(f"Reparsing HTML file for {self.page_type} {self.id}")
        if self.page_type == "events":
            df = self.event_extract_data()
        elif self.page_type == "fighters":
            df = self.fight_extract_data()
        self.save_to_csv(df)
        return df

    def check_files_exist(self) -> tuple[bool, bool]:
        """Check if both HTML and CSV files exist for this URL.
        Returns: (html_exists, csv_exists)
        """
        return (
            os.path.exists(self.html_path),
            os.path.exists(self.csv_path)
        )


def process_single_url(url: str, page_type: str, reload_html: bool, reload_csv: bool) -> None:
    """Process a single URL with the given settings."""
    try:
        scraper = DataScrapper(url=url, page_type=page_type)
        html_exists, csv_exists = scraper.check_files_exist()

        # Skip if both files exist and no reload is requested
        if html_exists and csv_exists and not (reload_html or reload_csv):
            logging.info(f"Skipping {url} - files already exist")
            return

        # Get/refresh HTML if needed
        if reload_html or not html_exists:
            scraper.refresh_data()
        # Reparse CSV if needed
        elif reload_csv or not csv_exists:
            scraper.reparse_data()
        else:
            scraper.get_data_as_df()

    except Exception as e:
        logging.error(f"Error processing {url}: {str(e)}")

def process_urls(
    urls: List[str], 
    page_type: str, 
    reload_html: bool, 
    reload_csv: bool
) -> None:
    """Process multiple URLs with parallel execution."""
    if PARALLEL_PROCESSING:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_single_url, url, page_type, reload_html, reload_csv)
                for url in urls
            ]
            for future in futures:
                future.result()  # This will raise any exceptions that occurred
    else:
        for url in urls:
            process_single_url(url, page_type, reload_html, reload_csv)

# Example usage
if __name__ == "__main__":
    try:
        # Step 1: Handle event list
        event_collector = URLCollector(page="events")

        if RELOAD_EVENT_LIST:
            logging.info("=== Reloading UFC event list ===")
            event_urls = event_collector.refresh_list()
        else:
            logging.info("=== Getting UFC event list ===")
            event_urls = event_collector.get_urls()

        if not event_urls:
            logging.error("No event URLs found. Exiting...")
            exit(1)

        # Apply event limit if specified
        if EVENT_LIMIT is not None:
            event_urls = event_urls[:EVENT_LIMIT]
            logging.info(f"\n=== Limited to processing {EVENT_LIMIT} events ===")

        # Process events
        process_urls(
            urls=event_urls,
            page_type="events",
            reload_html=RELOAD_EVENT_HTML,
            reload_csv=RELOAD_EVENT_CSV
        )

        # Step 1: Handle event list
        fighter_collector = URLCollector(page="fighters")

        if RELOAD_FIGHTER_LIST:
            logging.info("=== Reloading UFC fighter list ===")
            fighter_urls = fighter_collector.get_fighter_urls_from_events(event_collector.get_urls())
        else:
            # Try to load from CSV first
            try:
                fighter_urls_path = os.path.join(fighter_collector.csv_handler.csv_folder, "urls", "fighter_urls.csv")
                if os.path.exists(fighter_urls_path):
                    fighter_urls_df = pd.read_csv(fighter_urls_path)
                    fighter_urls = fighter_urls_df['url'].tolist()
                    logging.info(f"Loaded {len(fighter_urls)} fighter URLs from CSV")
                else:
                    logging.info("\n=== Getting UFC fighter list ===")
                    fighter_urls = fighter_collector.get_fighter_urls_from_events(event_collector.get_urls())
            except Exception as e:
                logging.error(f"Error loading fighter URLs from CSV: {str(e)}")
                logging.info("\n=== Getting UFC fighter list ===")
                fighter_urls = fighter_collector.get_fighter_urls_from_events(event_collector.get_urls())

        if not fighter_urls:
            logging.error("No fighter URLs found. Exiting...")
            exit(1)

        # Apply event limit if specified
        if FIGHTER_LIMIT is not None:
            fighter_urls = fighter_urls[:FIGHTER_LIMIT]
            logging.info(f"\n=== Limited to processing {FIGHTER_LIMIT} events ===")

        # Process fighters
        process_urls(
            urls=fighter_urls,
            page_type="fighters",
            reload_html=RELOAD_FIGHTER_HTML,
            reload_csv=RELOAD_FIGHTER_CSV
        )

        logging.info("\n=== Event Processing complete ===")
        logging.info("\n=== Fighter Processing complete ===")
        logging.info("\n=== All Processing complete ===")

    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        raise
