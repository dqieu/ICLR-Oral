import aiohttp
import asyncio
import os
import csv
import logging
import ssl
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from asyncio import Semaphore

# YEARS = [2023, 2024]
YEARS = [2023]  # doing year by year to avoid OpenReview Limit
BASE_URL = "https://neurips.cc"
DOWNLOAD_DIR = "downloaded_pdfs_neurips"
CSV_FILE = "papers_neurips.csv"
MAX_CONCURRENT_REQUESTS = 3  # Reduced from 5 to avoid overwhelming the server
TIMEOUT = aiohttp.ClientTimeout(total=30)  # 30-second timeout

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; NeurIPSBot/1.0; +http://example.com)'
}

# Ensure the download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

# Create a custom SSL context to disable certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Create semaphore for rate limiting
semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)


def sanitize_filename(name):
    """Sanitize string to be used as a filename."""
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name[:200]  # Limit filename length


async def fetch_html(session, url):
    """Fetch HTML content asynchronously with timeout and rate limiting."""
    async with semaphore:  # Rate limiting
        try:
            async with session.get(url, ssl=ssl_context,
                                   headers=HEADERS) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logging.error(f"HTTP {response.status} for {url}")
                    return None
        except asyncio.TimeoutError:
            logging.error(f"Timeout while fetching {url}")
            return None
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return None


async def get_oral_links(session, url):
    """Get all the oral links from a given year. Verified to work for ICLR 2021-2024.
    url: link to schedule of the target year
    """
    html = await fetch_html(session, url)
    if html is None:
        return []

    soup = BeautifulSoup(html, 'html.parser')

    oral_links = []
    for a in soup.select('div.hdr.eventsession div.sessiontitle a[href]'):
        if a.text.strip().startswith('Oral') or a.text.strip().startswith(
                'Outstanding Paper Session'):
            oral_links.append(BASE_URL + a['href'])

    return oral_links


async def get_paper_links(session, url) -> (str, list[str]):
    """Get the session name and all the paper links from an oral session."""
    html = await fetch_html(session, url)
    if html is None:
        return []

    subsoup = BeautifulSoup(html, 'html.parser')

    # Get session name
    h2 = subsoup.select_one('div.card-header h2.card-title')
    session_name = h2.text.strip() if h2 else "Unknown Session"

    # Get links to individual presentations
    talk_links = [
        BASE_URL + a['href'] for a in
        subsoup.select('h5 strong a[href^="/virtual/"]')
        if a.text.strip() != "Q&A"]

    return session_name, talk_links


async def download_pdf(session, url, title):
    """Download and save a PDF file."""
    safe_title = sanitize_filename(title)
    filename = os.path.join(DOWNLOAD_DIR, f"{safe_title}.pdf")

    if os.path.exists(filename):
        logging.info(f"Skipping existing PDF: {filename}")
        return

    async with semaphore:  # Rate limiting
        try:
            async with session.get(url, ssl=ssl_context,
                                   headers=HEADERS) as response:
                if response.status == 200:
                    with open(filename, "wb") as f:
                        f.write(await response.read())
                    logging.info(f"Downloaded: {title}")
                else:
                    logging.error(f"HTTP {response.status} for PDF {url}")
        except Exception as e:
            logging.error(f"Error downloading {url}: {e}")


def forum_to_pdf_link(forum_url=None):
    """
    Convert OpenReview forum URL or title to a direct PDF URL.

    Args:
        forum_url (str): Forum URL like 'https://openreview.net/forum?id=abc123'

    Returns:
        str or None: Direct link to the PDF file
    """
    if forum_url:
        return forum_url.replace('/forum?', '/pdf?')

    return None


async def process_papers_from_orals(session, oral_url, pbar):
    """Process a single paper and download it."""
    oral, papers = await get_paper_links(session, oral_url)
    for paper_url in papers:
        html = await fetch_html(session, paper_url)
        if html is None:
            return []

        soup = BeautifulSoup(html, 'html.parser')

        # === 1. Title ===
        title_tag = soup.select_one('div.card-header h2.card-title')
        title = title_tag.text.strip() if title_tag else "Unknown Title"

        # === 2. Authors ===
        authors_tag = soup.select_one('div.card-header h3.card-subtitle')
        authors = authors_tag.text.strip().split(
            " · ") if authors_tag else "Unknown Authors"

        # === 3. Abstract ===
        abstract_tag = soup.select_one('#abstract_details #abstractExample')
        abstract = abstract_tag.get_text(separator=' ', strip=True).replace(
            'Abstract:',
            '') if abstract_tag else "No abstract found"

        # === 4. OpenReview Link ===
        links = soup.select('a[href*="openreview.net/forum?id="]')
        filtered_links = {
            a['href'] for a in links if
            '2024-Oral' not in a['href'] and 'Oral-2024' not in a['href']
        }
        openreview_link = next(iter(filtered_links),
                               None)  # pick first if exists

        # === 5. Get PDF ===

        pdf_url = forum_to_pdf_link(openreview_link)
        if pdf_url:
            await download_pdf(session, pdf_url, title)

        # === 6. Get Year ===
        year = None
        for y in YEARS:
            if str(y) in paper_url[:-4]:
                year = y
                break

        metadata = {
            "title": sanitize_filename(title),
            "author": authors,
            "abstract": abstract,
            "openreview_url": openreview_link,
            "url": paper_url,
            "pdf_url": pdf_url,
            "publisher": "NeurIPS",
            "session": oral,
            "year": year,

        }

        save_metadata(metadata)
        pbar.update(1)
        logging.info(f"Processed: {title}")

    return None


def save_metadata(metadata):
    """Append paper metadata to the CSV file."""
    fieldnames = ["author", "publisher",
                  "title", "url", "year", "abstract", "session", "pdf_url",
                  "openreview_url"]

    file_exists = os.path.isfile(CSV_FILE)
    rows = []

    try:
        if file_exists:
            with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [row for row in reader if
                        row['title'].strip() != metadata['title'].strip()]

        rows.append(metadata)

        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        logging.error(f"Error saving metadata: {e}")


async def main():
    """Main function to orchestrate the scraping."""
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(ssl=ssl_context,
                                     limit=MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=HEADERS
    ) as session:

        # Process each year in smaller chunks
        for year in tqdm(YEARS, desc="Processing years"):
            year_url = f"{BASE_URL}/virtual/{year}/calendar"
            oral_links = await get_oral_links(session, year_url)
            logging.info(f"Found {len(oral_links)} orals for {year_url}")
            with tqdm(
                    total=len(oral_links),
                    desc=f"Papers from {str(year)}"
            ) as pbar:
                tasks = [process_papers_from_orals(session, oral_link, pbar)
                         for oral_link in oral_links]
                await asyncio.gather(*tasks)
                await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
