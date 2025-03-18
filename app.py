# import streamlit as st
# from agno.agent import Agent
# from agno.tools.firecrawl import FirecrawlTools
# import google.generativeai as genai
# from agno.tools.duckduckgo import DuckDuckGoTools
# import pandas as pd
# import requests
# from firecrawl import FirecrawlApp
# from pydantic import BaseModel, Field
# from typing import List, Optional
# import json
# import os
# from dotenv import load_dotenv, set_key
# import pathlib

# # Get the absolute path to the .env file
# env_path = pathlib.Path(os.path.join(os.getcwd(), '.env'))

# # Load environment variables from .env file
# load_dotenv(dotenv_path=env_path)

# # Streamlit UI
# st.set_page_config(page_title="AI Competitor Intelligence Agent Team", layout="wide")

# # Initialize session state for API keys if not already set
# if "api_keys_initialized" not in st.session_state:
#     # Get API keys from environment variables
#     st.session_state.env_gemini_api_key = os.getenv("GEMINI_API_KEY", "")
#     st.session_state.env_firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")
    
#     # Initialize the working API keys with environment values
#     st.session_state.gemini_api_key = st.session_state.env_gemini_api_key
#     st.session_state.firecrawl_api_key = st.session_state.env_firecrawl_api_key
    
#     st.session_state.api_keys_initialized = True

# # Initialize session state for competitor URLs
# if "competitor_urls" not in st.session_state:
#     st.session_state.competitor_urls = []

# # Function to save API keys to .env file
# def save_api_keys_to_env():
#     try:
#         # Save Gemini API key
#         if st.session_state.gemini_api_key:
#             set_key(env_path, "GEMINI_API_KEY", st.session_state.gemini_api_key)
            
#         # Save Firecrawl API key
#         if st.session_state.firecrawl_api_key:
#             set_key(env_path, "FIRECRAWL_API_KEY", st.session_state.firecrawl_api_key)
            
#         # Update environment variables in session state
#         st.session_state.env_gemini_api_key = st.session_state.gemini_api_key
#         st.session_state.env_firecrawl_api_key = st.session_state.firecrawl_api_key
        
#         return True
#     except Exception as e:
#         st.error(f"Error saving API keys to .env file: {str(e)}")
#         return False

# # Sidebar for API keys
# with st.sidebar:
#     st.title("AI Competitor Intelligence")
    
#     # API Key Management Section
#     st.subheader("API Key Management")
    
#     # Add option to show/hide API key inputs with expander
#     with st.expander("Configure API Keys", expanded=True):
#         st.info("API keys from .env file are used by default. You can override them here.")
        
#         # Function to handle API key updates with better validation
#         def update_api_key(key_name, env_key_name):
#             new_value = st.text_input(
#                 f"{key_name} API Key", 
#                 value=st.session_state[env_key_name] if st.session_state[env_key_name] else "",
#                 type="password",
#                 help=f"Enter your {key_name} API key or leave blank to use the one from .env file",
#                 key=f"input_{key_name.lower()}"
#             )
            
#             # Update session state regardless of input
#             if new_value:
#                 st.session_state[key_name.lower() + "_api_key"] = new_value
#                 return True
#             elif st.session_state[env_key_name]:
#                 st.session_state[key_name.lower() + "_api_key"] = st.session_state[env_key_name]
#                 return True
#             else:
#                 # Set to empty string to avoid None values
#                 st.session_state[key_name.lower() + "_api_key"] = ""
#                 return False
        
#         # Required API keys
#         has_gemini = update_api_key("Gemini", "env_gemini_api_key")
#         has_firecrawl = update_api_key("Firecrawl", "env_firecrawl_api_key")
        
#         # Debug information
#         st.write("API Key Status:")
#         st.write(f"Gemini: {'Set' if st.session_state.gemini_api_key else 'Not Set'}")
#         st.write(f"Firecrawl: {'Set' if st.session_state.firecrawl_api_key else 'Not Set'}")
        
#         # Buttons for API key management
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("Reset to .env values"):
#                 st.session_state.gemini_api_key = st.session_state.env_gemini_api_key
#                 st.session_state.firecrawl_api_key = st.session_state.env_firecrawl_api_key
#                 st.rerun()
        
#         with col2:
#             if st.button("Save to .env file"):
#                 if save_api_keys_to_env():
#                     st.success("API keys saved to .env file!")
#                     st.rerun()
    
#     # Display API status
#     api_status_ok = bool(st.session_state.gemini_api_key) and bool(st.session_state.firecrawl_api_key)
    
#     if api_status_ok:
#         st.success("✅ All required API keys are configured")
#     else:
#         missing_keys = []
#         if not st.session_state.gemini_api_key:
#             missing_keys.append("Gemini")
#         if not st.session_state.firecrawl_api_key:
#             missing_keys.append("Firecrawl")
        
#         st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")

# # Main UI
# st.title("🧲 AI Competitor Intelligence Agent Team")
# st.info(
#     """
#     This app helps businesses analyze their competitors by extracting structured data from competitor websites and generating insights using AI.
#     - Provide your **company URL** and the **number of competitors** you want to analyze
#     - Enter the competitor URLs directly
#     - The app will extract relevant information and generate a detailed analysis report
#     """
# )

# # Input field for company URL
# company_url = st.text_input("Enter your company URL:", key="company_url")

# # Input field for number of competitors
# num_competitors = st.number_input("Number of competitor URLs to analyze:", min_value=1, max_value=5, value=3, step=1)

# # Generate input fields for competitor URLs
# competitor_urls = []
# for i in range(int(num_competitors)):
#     competitor_url = st.text_input(f"Competitor URL #{i+1}:", key=f"competitor_url_{i}")
#     if competitor_url:
#         competitor_urls.append(competitor_url)

# # Initialize API keys and tools
# if api_status_ok:
#     # Configure Gemini
#     genai.configure(api_key=st.session_state.gemini_api_key)
    
#     firecrawl_tools = FirecrawlTools(
#         api_key=st.session_state.firecrawl_api_key,
#         scrape=False,
#         crawl=True,
#         limit=5
#     )

#     # Initialize Gemini model
#     generation_config = {
#         "temperature": 0.3,
#         "top_p": 1,
#         "top_k": 32,
#         "max_output_tokens": 4096,
#     }

#     safety_settings = [
#         {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#         {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#         {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#         {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#     ]

#     gemini_model = genai.GenerativeModel(
#         model_name="gemini-1.5-flash-latest",
#         generation_config=generation_config,
#         safety_settings=safety_settings
#     )

#     class CompetitorDataSchema(BaseModel):
#         company_name: str = Field(description="Name of the company")
#         pricing: str = Field(description="Pricing details, tiers, and plans")
#         key_features: List[str] = Field(description="Main features and capabilities of the product/service")
#         marketing_focus: str = Field(description="Main marketing angles and target audience")
#         customer_feedback: str = Field(description="Customer testimonials, reviews, and feedback")

#     def extract_competitor_info(competitor_url: str) -> Optional[dict]:
#         try:
#             # Initialize FirecrawlApp with API key
#             app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)
            
#             # Add wildcard to crawl subpages
#             url_pattern = f"{competitor_url}/*"
            
#             extraction_prompt = """
#             Extract detailed information about the company's offerings, including:
#             - Company name and basic information
#             - Pricing details, plans, and tiers
#             - Key features and main capabilities
#             - Marketing focus and target audience
#             - Customer feedback and testimonials
            
#             Analyze the entire website content to provide comprehensive information for each field.
#             """
            
#             response = app.extract(
#                 [url_pattern],
#                 {
#                     'prompt': extraction_prompt,
#                     'schema': CompetitorDataSchema.model_json_schema(),
#                 }
#             )
            
#             if response.get('success') and response.get('data'):
#                 extracted_info = response['data']
                
#                 # Create JSON structure
#                 competitor_json = {
#                     "competitor_url": competitor_url,
#                     "company_name": extracted_info.get('company_name', 'N/A'),
#                     "pricing": extracted_info.get('pricing', 'N/A'),
#                     "key_features": extracted_info.get('key_features', [])[:5],  # Top 5 features
#                     "marketing_focus": extracted_info.get('marketing_focus', 'N/A'),
#                     "customer_feedback": extracted_info.get('customer_feedback', 'N/A')
#                 }
                
#                 return competitor_json
                
#             else:
#                 return None
                
#         except Exception as e:
#             st.error(f"Error extracting data from {competitor_url}: {str(e)}")
#             return None

#     def extract_company_info(url: str) -> Optional[dict]:
#         try:
#             app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)
#             url_pattern = f"{url}/*"
            
#             extraction_prompt = """
#             Extract detailed information about the company's offerings, including:
#             - Company name and basic information
#             - Pricing details, plans, and tiers
#             - Key features and main capabilities
#             - Marketing focus and target audience
#             - Customer feedback and testimonials
            
#             Analyze the entire website content to provide comprehensive information for each field.
#             """
            
#             response = app.extract(
#                 [url_pattern],
#                 {
#                     'prompt': extraction_prompt,
#                     'schema': CompetitorDataSchema.model_json_schema(),
#                 }
#             )
            
#             if response.get('success') and response.get('data'):
#                 extracted_info = response['data']
                
#                 company_json = {
#                     "url": url,
#                     "company_name": extracted_info.get('company_name', 'N/A'),
#                     "pricing": extracted_info.get('pricing', 'N/A'),
#                     "key_features": extracted_info.get('key_features', [])[:5],
#                     "marketing_focus": extracted_info.get('marketing_focus', 'N/A'),
#                     "customer_feedback": extracted_info.get('customer_feedback', 'N/A')
#                 }
                
#                 return company_json
                
#             else:
#                 return None
                
#         except Exception as e:
#             st.error(f"Error extracting data from your company URL {url}: {str(e)}")
#             return None

#     def generate_comparison_report(company_data: dict, competitor_data: list) -> None:
#         all_data = [company_data] + competitor_data
#         formatted_data = json.dumps(all_data, indent=2)
        
#         system_prompt = f"""
#         As an expert business analyst, analyze the following data in JSON format and create a structured comparison.
#         The first entry is the user's company, followed by competitor data.
#         Extract and summarize the key information into concise points.

#         {formatted_data}

#         Return the data in a structured format with EXACTLY these columns:
#         Company, Pricing, Key Features, Marketing Focus, Customer Feedback

#         Rules:
#         1. For Company: Include company name and URL
#         2. For Key Features: List top 3 most important features only
#         3. Keep all entries clear and concise
#         4. Format feedback as brief quotes
#         5. Return ONLY the structured data as a markdown table, no additional text
#         """

#         try:
#             response = gemini_model.generate_content(system_prompt)
#             table_lines = [
#                 line.strip() 
#                 for line in response.text.split('\n') 
#                 if line.strip() and '|' in line
#             ]
            
#             headers = [
#                 col.strip() 
#                 for col in table_lines[0].split('|') 
#                 if col.strip()
#             ]
            
#             data_rows = []
#             for line in table_lines[2:]:
#                 row_data = [
#                     cell.strip() 
#                     for cell in line.split('|') 
#                     if cell.strip()
#                 ]
#                 if len(row_data) == len(headers):
#                     data_rows.append(row_data)
            
#             df = pd.DataFrame(
#                 data_rows,
#                 columns=headers
#             )
            
#             st.subheader("Company Comparison")
#             st.table(df)
            
#         except Exception as e:
#             st.error(f"Error creating comparison table: {str(e)}")
#             st.write("Raw comparison data for debugging:", response.text)

#     def generate_analysis_report(company_data: dict, competitor_data: list):
#         formatted_company = json.dumps(company_data, indent=2)
#         formatted_competitors = json.dumps(competitor_data, indent=2)
        
#         prompt = f"""Analyze the following data and identify market opportunities to improve the company:
        
#         USER'S COMPANY:
#         {formatted_company}
        
#         COMPETITORS:
#         {formatted_competitors}

#         Tasks:
#         1. Identify market gaps and opportunities based on competitor offerings
#         2. Analyze competitor weaknesses that the company can capitalize on
#         3. Recommend unique features or capabilities they should develop
#         4. Suggest pricing and positioning strategies to gain competitive advantage
#         5. Outline specific growth opportunities in underserved market segments
#         6. Provide actionable recommendations for product development and go-to-market strategy

#         Focus on finding opportunities where the company can differentiate and do better than competitors.
#         Highlight any unmet customer needs or pain points they can address.
#         """

#         try:
#             response = gemini_model.generate_content(prompt)
#             return response.text
#         except Exception as e:
#             return f"Error generating analysis: {str(e)}"

#     # Run analysis when the user clicks the button
#     if st.button("Analyze Competitors"):
#         if not api_status_ok:
#             st.error("⚠️ Please configure all required API keys in the sidebar before proceeding.")
#         elif not company_url:
#             st.error("Please enter your company URL.")
#         elif len(competitor_urls) == 0:
#             st.error("Please enter at least one competitor URL.")
#         elif len(competitor_urls) < int(num_competitors):
#             st.warning(f"You specified {num_competitors} competitors but only entered {len(competitor_urls)} URLs.")
#             st.error("Please fill in all competitor URL fields.")
#         else:
#             # Extract company information
#             with st.spinner(f"Analyzing your company: {company_url}..."):
#                 company_info = extract_company_info(company_url)
#                 if company_info is None:
#                     st.error(f"Could not extract data from your company URL: {company_url}")
#                     st.stop()
            
#             # Extract competitor information
#             competitor_data = []
#             for comp_url in competitor_urls:
#                 with st.spinner(f"Analyzing Competitor: {comp_url}..."):
#                     competitor_info = extract_competitor_info(comp_url)
#                     if competitor_info is not None:
#                         competitor_data.append(competitor_info)
            
#             if competitor_data:
#                 # Generate and display comparison report
#                 with st.spinner("Generating comparison table..."):
#                     generate_comparison_report(company_info, competitor_data)
                
#                 # Generate and display final analysis report
#                 with st.spinner("Generating analysis report..."):
#                     analysis_report = generate_analysis_report(company_info, competitor_data)
#                     st.subheader("Competitor Analysis Report")
#                     st.markdown(analysis_report)
                
#                 st.success("Analysis complete!")
#             else:
#                 st.error("Could not extract data from any competitor URLs")
#     else:
#         if not api_status_ok:
#             st.warning("⚠️ Configure your API keys in the sidebar before analyzing competitors.")






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
        "temperature": 0.5,
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
            
            response = app.extract(
                [url_pattern],
                {
                    'prompt': extraction_prompt,
                    'schema': CompetitorDataSchema.model_json_schema(),
                }
            )
            
            if response.get('success') and response.get('data'):
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
                # st.subheader("Competitive Comparison")
                # comparison_df = generate_comparison(company_data, competitors_data)
                # if not comparison_df.empty:
                #     st.table(comparison_df)
                
                st.subheader("Strategic Recommendations")
                analysis = generate_strategic_analysis(company_data, competitors_data)
                st.markdown(analysis)
            else:
                st.error("Failed to collect sufficient data for analysis")
else:
    st.warning("Configure API keys in sidebar to begin")
