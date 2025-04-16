import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

def download_pdf(url, folder_path):
    """
    Download a PDF file from the given URL to the specified folder
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created directory: {folder_path}")
    
    # Get the filename from the URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # If the filename is empty or doesn't end with .pdf, create a default name
    if not filename or not filename.lower().endswith('.pdf'):
        filename = f"document_{hash(url) % 10000}.pdf"
    
    # Full path where the file will be saved
    file_path = os.path.join(folder_path, filename)
    
    try:
        # Send a GET request to the URL with headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
        }
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Check if the response is a PDF
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' in content_type:
            # Save the PDF
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            print(f"Successfully downloaded: {filename}")
            return True
        else:
            print(f"Skipping {url}: Not a PDF (Content-Type: {content_type})")
            return False
    
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def is_likely_pdf_link(href, link_text):
    """Check if a link is likely to be a PDF based on href or link text"""
    if not href:
        return False
        
    # Check if href contains pdf indicators
    pdf_indicators_in_href = [
        href.lower().endswith('.pdf'),
        'pdf' in href.lower(),
        'documento' in href.lower(),
        'download' in href.lower(),
        'archivo' in href.lower(),
        'attachment' in href.lower()
    ]
    
    # Check if link text contains pdf indicators
    pdf_indicators_in_text = []
    if link_text:
        link_text_lower = link_text.lower()
        pdf_indicators_in_text = [
            'pdf' in link_text_lower,
            'descargar' in link_text_lower,
            'documento' in link_text_lower,
            'informe' in link_text_lower,
            'anexo' in link_text_lower,
            'ver' in link_text_lower and 'documento' in link_text_lower
        ]
    
    return any(pdf_indicators_in_href) or any(pdf_indicators_in_text)

def scrape_pdfs(webpage_url, folder_path):
    """
    Scrape all PDF links from a webpage and download them
    """
    try:
        # Send a GET request to the webpage with headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
        }
        response = requests.get(webpage_url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all anchor tags
        links = soup.find_all('a')
        
        # Filter links that are likely to point to PDFs
        pdf_links = []
        for link in links:
            href = link.get('href')
            link_text = link.get_text(strip=True)
            
            if is_likely_pdf_link(href, link_text):
                # Convert relative URLs to absolute URLs
                absolute_url = urljoin(webpage_url, href)
                print(f"Found potential PDF link: {absolute_url}")
                pdf_links.append(absolute_url)
        
        if not pdf_links:
            print("No PDF links found on the page.")
            return []
        
        print(f"Found {len(pdf_links)} potential PDF links on the page.")
        
        # Download each PDF
        downloaded_files = []
        for pdf_url in pdf_links:
            if download_pdf(pdf_url, folder_path):
                downloaded_files.append(pdf_url)
        
        return downloaded_files
    
    except Exception as e:
        print(f"Error scraping the webpage: {e}")
        return []

def print_html_structure(webpage_url):
    """Print the structure of the webpage to help debug"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
        }
        response = requests.get(webpage_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Print all links
        print("All links on the page:")
        for link in soup.find_all('a'):
            href = link.get('href')
            text = link.get_text(strip=True)
            print(f"Text: '{text}', Link: {href}")
            
    except Exception as e:
        print(f"Error analyzing webpage: {e}")

if __name__ == "__main__":
    # Set pre-defined values
    webpage_url = "https://www.gob.pe/institucion/minsa/informes-publicaciones/6649378-avances-de-los-resultados-de-la-evaluacion-para-el-serums-2025-i"
    folder_path = r"C:\Users\Alex\Desktop\Analytics\Sewrums"
    
    # Print the HTML structure to help debug
    print("Analyzing webpage structure...")
    print_html_structure(webpage_url)
    
    # Scrape and download PDFs
    downloaded_files = scrape_pdfs(webpage_url, folder_path)
    
    # Print summary
    print("\nSummary:")
    print(f"Total PDFs downloaded: {len(downloaded_files)}")
    print(f"PDFs downloaded to: {os.path.abspath(folder_path)}")