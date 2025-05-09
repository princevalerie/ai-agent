import streamlit as st
from agno.agent import Agent
from agno.tools.firecrawl import FirecrawlTools
import google.generativeai as genai
from agno.tools.duckduckgo import DuckDuckGoTools
import pandas as pd
import requests
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os
from dotenv import load_dotenv, set_key
import pathlib

# Get the absolute path to the .env file
env_path = pathlib.Path(os.path.join(os.getcwd(), '.env'))

# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)

# Streamlit UI
st.set_page_config(page_title="AI Competitor Intelligence Agent Team", layout="wide")

# Initialize session state for API keys
if "api_keys_initialized" not in st.session_state:
    st.session_state.env_gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    st.session_state.env_firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")
    st.session_state.gemini_api_key = st.session_state.env_gemini_api_key
    st.session_state.firecrawl_api_key = st.session_state.env_firecrawl_api_key
    st.session_state.api_keys_initialized = True

if "competitor_urls" not in st.session_state:
    st.session_state.competitor_urls = []

def save_api_keys_to_env():
    try:
        if st.session_state.gemini_api_key:
            set_key(env_path, "GEMINI_API_KEY", st.session_state.gemini_api_key)
        if st.session_state.firecrawl_api_key:
            set_key(env_path, "FIRECRAWL_API_KEY", st.session_state.firecrawl_api_key)
        st.session_state.env_gemini_api_key = st.session_state.gemini_api_key
        st.session_state.env_firecrawl_api_key = st.session_state.firecrawl_api_key
        return True
    except Exception as e:
        st.error(f"Error saving API keys: {str(e)}")
        return False

# Sidebar
with st.sidebar:
    st.title("AI Competitor Intelligence")
    st.subheader("API Key Management")
    
    with st.expander("Configure API Keys", expanded=True):
        st.info("API keys from .env are used by default. Override here if needed.")
        
        def update_api_key(key_name, env_key_name):
            new_value = st.text_input(
                f"{key_name} API Key", 
                value=st.session_state[env_key_name] or "",
                type="password",
                key=f"input_{key_name.lower()}"
            )
            if new_value:
                st.session_state[key_name.lower() + "_api_key"] = new_value
                return True
            elif st.session_state[env_key_name]:
                st.session_state[key_name.lower() + "_api_key"] = st.session_state[env_key_name]
                return True
            return False
        
        has_gemini = update_api_key("Gemini", "env_gemini_api_key")
        has_firecrawl = update_api_key("Firecrawl", "env_firecrawl_api_key")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset to .env"):
                st.session_state.gemini_api_key = st.session_state.env_gemini_api_key
                st.session_state.firecrawl_api_key = st.session_state.env_firecrawl_api_key
                st.rerun()
        with col2:
            if st.button("Save to .env"):
                if save_api_keys_to_env():
                    st.success("API keys saved!")
                    st.rerun()

    api_status_ok = bool(st.session_state.gemini_api_key) and bool(st.session_state.firecrawl_api_key)
    if api_status_ok:
        st.success("✅ API keys configured")
    else:
        missing = [k for k, v in {'Gemini': st.session_state.gemini_api_key, 
                                 'Firecrawl': st.session_state.firecrawl_api_key}.items() if not v]
        st.error(f"❌ Missing: {', '.join(missing)}")

# Main UI
st.title("🧲 AI Competitor Intelligence Agent Team")
st.info("""
Analyze competitors by extracting structured data from websites and generating AI insights.
1. Enter your company URL
2. Add competitor URLs
3. Get automated analysis
""")

# Input fields
company_url = st.text_input("Your Company URL:", key="company_url")
num_competitors = st.number_input("Number of Competitors:", 1, 5, 3)
competitor_urls = [st.text_input(f"Competitor URL #{i+1}:", key=f"comp_{i}") for i in range(num_competitors)]
competitor_urls = [url for url in competitor_urls if url]

# Initialize APIs
if api_status_ok:
    genai.configure(api_key=st.session_state.gemini_api_key)
    firecrawl_tools = FirecrawlTools(
        api_key=st.session_state.firecrawl_api_key,
        crawl=True,
        limit=5
    )

    generation_config = {
        "temperature": 0.2,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    gemini_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    class CompetitorDataSchema(BaseModel):
        company_name: str = Field(description="Company name")
        pricing: str = Field(description="Pricing details")
        key_features: List[str] = Field(description="Key features")
        marketing_focus: str = Field(description="Marketing focus")
        customer_feedback: str = Field(description="Customer feedback")

    def extract_data(url: str, is_competitor: bool = False) -> Optional[dict]:
        try:
            app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)
            url_pattern = f"{url}/*"
            
            extraction_prompt = """
            Extract detailed company information from website content including:
            - Company name
            - Pricing structure
            - Key product features
            - Marketing strategies
            - Customer testimonials
            """
            
            # Prepare the schema as a dictionary
            schema = {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "description": "Company name"},
                    "pricing": {"type": "string", "description": "Pricing details"},
                    "key_features": {"type": "array", "items": {"type": "string"}, "description": "Key features"},
                    "marketing_focus": {"type": "string", "description": "Marketing focus"},
                    "customer_feedback": {"type": "string", "description": "Customer feedback"}
                },
                "required": ["company_name"]
            }
            
            # Try extracting with URLs and configuration in separate parameters
            try:
                response = app.extract(
                    [url_pattern],
                    {
                        "prompt": extraction_prompt,
                        "schema": schema
                    }
                )
            except TypeError:
                # If the above fails, try the alternative format as a single parameter
                extract_config = {
                    "urls": [url_pattern],
                    "prompt": extraction_prompt,
                    "schema": schema
                }
                response = app.extract(extract_config)
            
            if response and response.get('data'):
                data = response['data']
                return {
                    "url": url,
                    "company_name": data.get('company_name', 'N/A'),
                    "pricing": data.get('pricing', 'N/A'),
                    "key_features": data.get('key_features', [])[:5],
                    "marketing_focus": data.get('marketing_focus', 'N/A'),
                    "customer_feedback": data.get('customer_feedback', 'N/A')
                }
            return None
        except Exception as e:
            st.error(f"Error processing {url}: {str(e)}")
            return None

    def generate_comparison(company: dict, competitors: list) -> pd.DataFrame:
        all_data = [company] + competitors
        prompt = f"""
        Create comparison table from this JSON data:
        {json.dumps(all_data, indent=2)}
        
        Format Requirements:
        - Table columns: Company | Pricing | Key Features | Marketing Focus | Customer Feedback
        - Company format: Name (URL)
        - Features: Top 3 only
        - Feedback: Short quotes
        - Return ONLY markdown table
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            lines = [line.strip() for line in response.text.split('\n') if '|' in line]
            headers = [h.strip() for h in lines[0].split('|') if h]
            rows = [[cell.strip() for cell in line.split('|') if cell] for line in lines[2:]]
            return pd.DataFrame(rows, columns=headers)
        except Exception as e:
            st.error(f"Comparison error: {str(e)}")
            return pd.DataFrame()

    def generate_strategic_analysis(company: dict, competitors: list) -> str:
        prompt = f"""
        Analyze this market data and provide strategic recommendations:
        
        Company Data:
        {json.dumps(company, indent=2)}
        
        Competitor Data:
        {json.dumps(competitors, indent=2)}
        
        Analysis Requirements:
        1. Identify market gaps
        2. Competitor weaknesses
        3. Feature opportunities
        4. Pricing strategies
        5. Growth recommendations
        
        Format: Markdown with clear sections and bullet points
        """
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Analysis error: {str(e)}"

    if st.button("Run Full Analysis"):
        if not company_url or len(competitor_urls) < 1:
            st.error("Please fill all required fields")
        else:
            with st.spinner("Analyzing your company..."):
                company_data = extract_data(company_url)
            
            competitors_data = []
            for url in competitor_urls:
                with st.spinner(f"Processing {url}..."):
                    if data := extract_data(url, is_competitor=True):
                        competitors_data.append(data)
            
            if company_data and competitors_data:
                # Uncomment to enable comparison table
                st.subheader("Competitive Comparison")
                comparison_df = generate_comparison(company_data, competitors_data)
                if not comparison_df.empty:
                    st.table(comparison_df)
                
                st.subheader("Strategic Recommendations")
                analysis = generate_strategic_analysis(company_data, competitors_data)
                st.markdown(analysis)
            else:
                st.error("Failed to collect sufficient data for analysis")
else:
    st.warning("Configure API keys in sidebar to begin")
