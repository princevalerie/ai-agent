import streamlit as st
import google.generativeai as genai
from firecrawl import FirecrawlApp, ScrapeOptions, JsonConfig
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os
from dotenv import load_dotenv, set_key
import pathlib
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Get the absolute path to the .env file
env_path = pathlib.Path(os.path.join(os.getcwd(), '.env'))

# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)

# Streamlit UI
st.set_page_config(page_title="AI Competitor Intelligence Agent", layout="wide")

# Initialize session state for API keys if not already set
if "api_keys_initialized" not in st.session_state:
    # Get API keys from environment variables
    st.session_state.env_gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    st.session_state.env_firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")
    
    # Initialize the working API keys with environment values
    st.session_state.gemini_api_key = st.session_state.env_gemini_api_key
    st.session_state.firecrawl_api_key = st.session_state.env_firecrawl_api_key
    
    st.session_state.api_keys_initialized = True

# Function to save API keys to .env file
def save_api_keys_to_env():
    try:
        # Save Gemini API key
        if st.session_state.gemini_api_key:
            set_key(env_path, "GEMINI_API_KEY", st.session_state.gemini_api_key)
            
        # Save Firecrawl API key
        if st.session_state.firecrawl_api_key:
            set_key(env_path, "FIRECRAWL_API_KEY", st.session_state.firecrawl_api_key)
            
        # Update environment variables in session state
        st.session_state.env_gemini_api_key = st.session_state.gemini_api_key
        st.session_state.env_firecrawl_api_key = st.session_state.firecrawl_api_key
        
        return True
    except Exception as e:
        st.error(f"Error saving API keys to .env file: {str(e)}")
        return False

# Sidebar for API keys
with st.sidebar:
    st.title("AI Competitor Intelligence")
    
    # API Key Management Section
    st.subheader("API Key Management")
    
    # Add option to show/hide API key inputs with expander
    with st.expander("Configure API Keys", expanded=False):
        st.info("API keys from .env file are used by default. You can override them here.")
        
        # Function to handle API key updates
        def update_api_key(key_name, env_key_name):
            new_value = st.text_input(
                f"{key_name} API Key", 
                value=st.session_state[env_key_name] if st.session_state[env_key_name] else "",
                type="password",
                help=f"Enter your {key_name} API key or leave blank to use the one from .env file"
            )
            
            # Only update if user entered something or if we have an env value
            if new_value:
                st.session_state[key_name.lower() + "_api_key"] = new_value
                return True
            elif st.session_state[env_key_name]:
                st.session_state[key_name.lower() + "_api_key"] = st.session_state[env_key_name]
                return True
            return False
        
        # Required API keys
        has_gemini = update_api_key("Gemini", "env_gemini_api_key")
        has_firecrawl = update_api_key("Firecrawl", "env_firecrawl_api_key")
        
        # Buttons for API key management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset to .env values"):
                st.session_state.gemini_api_key = st.session_state.env_gemini_api_key
                st.session_state.firecrawl_api_key = st.session_state.env_firecrawl_api_key
                st.experimental_rerun()
        
        with col2:
            if st.button("Save to .env file"):
                if save_api_keys_to_env():
                    st.success("API keys saved to .env file!")
                    st.experimental_rerun()
    
    # Display API status
    api_status_ok = has_gemini and has_firecrawl
    
    if api_status_ok:
        st.success("✅ All required API keys are configured")
    else:
        missing_keys = []
        if not has_gemini:
            missing_keys.append("Gemini")
        if not has_firecrawl:
            missing_keys.append("Firecrawl")
        
        st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")

# Main UI
st.title("🧲 AI Competitor Intelligence Agent")
st.info(
    """
    This app helps businesses analyze their competitors by extracting structured data from competitor websites and generating insights using AI.
    - Enter the URLs of your competitors' websites.
    - The app will extract relevant information and generate a detailed analysis report.
    """
)

# Input fields for competitor URLs
num_competitors = st.number_input("Jumlah Kompetitor:", 1, 5, 3)
competitor_urls = [st.text_input(f"URL Kompetitor #{i+1}:", placeholder="https://example.com", key=f"comp_{i}") for i in range(num_competitors)]
competitor_urls = [url for url in competitor_urls if url]

# Initialize API keys and tools
if api_status_ok:
    # Initialize Gemini
    genai.configure(api_key=st.session_state.gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')

    class CompetitorDataSchema(BaseModel):
        company_name: str = Field(description="Name of the company")
        pricing: str = Field(description="Pricing details, tiers, and plans")
        key_features: List[str] = Field(description="Main features and capabilities of the product/service")
        tech_stack: List[str] = Field(description="Technologies, frameworks, and tools used")
        marketing_focus: str = Field(description="Main marketing angles and target audience")
        customer_feedback: str = Field(description="Customer testimonials, reviews, and feedback")

    def extract_competitor_info(competitor_url: str) -> Optional[dict]:
        try:
            # Initialize FirecrawlApp dengan API key
            app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)
            
            # 1. Crawl untuk mendapatkan daftar URL
            crawl_result = app.crawl_url(
                competitor_url,
                limit=50,
                scrape_options=ScrapeOptions(
                    formats=['markdown'],
                    onlyMainContent=True,
                    includeTags=['article', 'main', 'section'],
                    excludeTags=['nav', 'footer', 'header'],
                    mobile=False,
                    skipTlsVerification=False
                )
            )

            # Periksa status crawling
            if not crawl_result or not hasattr(crawl_result, 'success') or not crawl_result.success:
                if hasattr(crawl_result, 'error') and 'Insufficient credits' in str(crawl_result.error):
                    st.error(f"Kredit Firecrawl tidak mencukupi. Silakan upgrade plan atau coba lagi nanti.")
                else:
                    st.warning(f"Gagal melakukan crawling pada: {competitor_url}")
                return None

            # 2. Ekstrak data dari setiap halaman yang di-crawl
            competitor_data = {
                'company_name': 'N/A',
                'pricing': 'N/A',
                'key_features': [],
                'tech_stack': [],
                'marketing_focus': 'N/A',
                'customer_feedback': 'N/A',
                'competitor_url': competitor_url,
                'scraped_at': pd.Timestamp.now().isoformat()
            }

            # Konfigurasi ekstraksi JSON
            json_config = JsonConfig(
                prompt="""
                Ekstrak informasi berikut dari konten website:
                1. Nama perusahaan (company_name)
                2. Informasi harga dan paket (pricing)
                   - Tuliskan semua tier dan harga
                   - Sertakan fitur spesifik untuk setiap tier
                3. Fitur utama produk/layanan (key_features)
                   - Pilih 5 fitur paling penting
                   - Sertakan deskripsi singkat untuk setiap fitur
                4. Teknologi yang digunakan (tech_stack)
                   - Identifikasi framework, bahasa pemrograman, dan tools
                   - Fokus pada teknologi yang disebutkan secara eksplisit
                5. Fokus pemasaran (marketing_focus)
                   - Target pasar
                   - Value proposition utama
                   - Unique selling points
                6. Umpan balik pelanggan (customer_feedback)
                   - Testimoni yang ada
                   - Rating atau review jika tersedia

                Jika informasi tidak ditemukan, gunakan 'N/A' sebagai nilai default.
                Pastikan semua informasi yang diekstrak akurat dan relevan.
                """,
                pageOptions={"onlyMainContent": True}
            )

            # Proses setiap halaman yang di-crawl
            if hasattr(crawl_result, 'data'):
                for page in crawl_result.data:
                    if 'metadata' in page and 'sourceURL' in page['metadata']:
                        url = page['metadata']['sourceURL']
                        # Ekstrak data dari setiap halaman
                        scrape_result = app.scrape_url(
                            url,
                            formats=["json"],
                            json_options=json_config
                        )
                        
                        if scrape_result and hasattr(scrape_result, 'data') and 'json' in scrape_result.data:
                            data = scrape_result.data['json']
                            # Update competitor_data dengan data baru jika tidak 'N/A'
                            for key in ['company_name', 'pricing', 'marketing_focus', 'customer_feedback']:
                                if data.get(key) and data[key] != 'N/A':
                                    competitor_data[key] = data[key]
                            
                            # Gabungkan array
                            if data.get('key_features'):
                                competitor_data['key_features'].extend([f for f in data['key_features'] if f != 'N/A'])
                            if data.get('tech_stack'):
                                competitor_data['tech_stack'].extend([t for t in data['tech_stack'] if t != 'N/A'])

            # Hapus duplikat dari array
            competitor_data['key_features'] = list(set(competitor_data['key_features']))
            competitor_data['tech_stack'] = list(set(competitor_data['tech_stack']))

            return competitor_data

        except Exception as e:
            if 'Insufficient credits' in str(e):
                st.error(f"Kredit Firecrawl tidak mencukupi. Silakan upgrade plan atau coba lagi nanti.")
            else:
                st.error(f"Error mengekstrak data dari {competitor_url}: {str(e)}")
            return None

    def generate_comparison_report(competitor_data: list) -> None:
        # Format the competitor data for the prompt
        formatted_data = json.dumps(competitor_data, indent=2)
        
        prompt = f"""As an expert business analyst, analyze the following competitor data in JSON format and create a structured comparison.
        Extract and summarize the key information into concise points.

        {formatted_data}

        Return the data in a structured format with EXACTLY these columns:
        Company, Pricing, Key Features, Tech Stack, Marketing Focus, Customer Feedback

        Rules:
        1. For Company: Include company name and URL
        2. For Key Features: List top 3 most important features only
        3. For Tech Stack: List top 3 most relevant technologies only
        4. Keep all entries clear and concise
        5. Format feedback as brief quotes
        6. Return ONLY the structured data, no additional text"""

        try:
            response = model.generate_content(prompt)
            table_text = response.text
            
            # Split the response into lines and clean them
            table_lines = [
                line.strip() 
                for line in table_text.split('\n') 
                if line.strip() and '|' in line
            ]
            
            # Extract headers (first row)
            headers = [
                col.strip() 
                for col in table_lines[0].split('|') 
                if col.strip()
            ]
            
            # Extract data rows (skip header and separator rows)
            data_rows = []
            for line in table_lines[2:]:  # Skip header and separator rows
                row_data = [
                    cell.strip() 
                    for cell in line.split('|') 
                    if cell.strip()
                ]
                if len(row_data) == len(headers):
                    data_rows.append(row_data)
            
            # Create DataFrame
            df = pd.DataFrame(
                data_rows,
                columns=headers
            )
            
            # Display the table
            st.subheader("Competitor Comparison")
            st.table(df)
            
        except Exception as e:
            st.error(f"Error creating comparison table: {str(e)}")
            st.write("Raw comparison data for debugging:", table_text)

    def generate_analysis_report(competitor_data: list):
        # Format the competitor data for the prompt
        formatted_data = json.dumps(competitor_data, indent=2)
        
        prompt = f"""Analyze the following competitor data in JSON format and identify market opportunities to improve my own company:
            
        {formatted_data}

        Tasks:
        1. Identify market gaps and opportunities based on competitor offerings
        2. Analyze competitor weaknesses that we can capitalize on
        3. Recommend unique features or capabilities we should develop
        4. Suggest pricing and positioning strategies to gain competitive advantage
        5. Outline specific growth opportunities in underserved market segments
        6. Provide actionable recommendations for product development and go-to-market strategy

        Focus on finding opportunities where we can differentiate and do better than competitors.
        Highlight any unmet customer needs or pain points we can address."""

        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating analysis report: {str(e)}")
            return "Error generating analysis report."

    # Run analysis when the user clicks the button
    if st.button("Analyze Competitors"):
        if not api_status_ok:
            st.error("⚠️ Please configure all required API keys in the sidebar before proceeding.")
        elif competitor_urls:
            competitor_data = []
            for comp_url in competitor_urls:
                with st.spinner(f"Analyzing Competitor: {comp_url}..."):
                    st.write(f"Processing URL: {comp_url}")
                    competitor_info = extract_competitor_info(comp_url)
                    if competitor_info is not None:
                        competitor_data.append(competitor_info)
                        st.success(f"Successfully extracted data from: {comp_url}")
                    else:
                        st.error(f"Failed to extract data from: {comp_url}")
            
            if competitor_data:
                # Generate and display comparison report
                with st.spinner("Generating comparison table..."):
                    generate_comparison_report(competitor_data)
                
                # Generate and display final analysis report
                with st.spinner("Generating analysis report..."):
                    analysis_report = generate_analysis_report(competitor_data)
                    st.subheader("Competitor Analysis Report")
                    st.markdown(analysis_report)
                
                st.success("Analysis complete!")
            else:
                st.error("Could not extract data from any competitor URLs")
        else:
            st.error("Please provide at least one competitor URL.")
    else:
        # Display API key status message when the app first loads
        if not api_status_ok:
            st.warning("⚠️ Configure your API keys in the sidebar before analyzing competitors.")
else:
        # Display API key status message when the app first loads
        if not api_status_ok:
            st.warning("⚠️ Configure your API keys in the sidebar before analyzing competitors.")
