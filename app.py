import streamlit as st
from exa_py import Exa
from agno.agent import Agent
from agno.tools.firecrawl import FirecrawlTools
from agno.models.openai import OpenAIChat
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

# Initialize session state for API keys if not already set
if "api_keys_initialized" not in st.session_state:
    # Get API keys from environment variables
    st.session_state.env_openai_api_key = os.getenv("OPENAI_API_KEY", "")
    st.session_state.env_firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")
    st.session_state.env_perplexity_api_key = os.getenv("PERPLEXITY_API_KEY", "")
    st.session_state.env_exa_api_key = os.getenv("EXA_API_KEY", "")
    
    # Initialize the working API keys with environment values
    st.session_state.openai_api_key = st.session_state.env_openai_api_key
    st.session_state.firecrawl_api_key = st.session_state.env_firecrawl_api_key
    st.session_state.perplexity_api_key = st.session_state.env_perplexity_api_key
    st.session_state.exa_api_key = st.session_state.env_exa_api_key
    
    st.session_state.api_keys_initialized = True

# Function to save API keys to .env file
def save_api_keys_to_env():
    try:
        # Save OpenAI API key
        if st.session_state.openai_api_key:
            set_key(env_path, "OPENAI_API_KEY", st.session_state.openai_api_key)
            
        # Save Firecrawl API key
        if st.session_state.firecrawl_api_key:
            set_key(env_path, "FIRECRAWL_API_KEY", st.session_state.firecrawl_api_key)
            
        # Save Perplexity API key
        if st.session_state.perplexity_api_key:
            set_key(env_path, "PERPLEXITY_API_KEY", st.session_state.perplexity_api_key)
            
        # Save Exa API key
        if st.session_state.exa_api_key:
            set_key(env_path, "EXA_API_KEY", st.session_state.exa_api_key)
            
        # Update environment variables in session state
        st.session_state.env_openai_api_key = st.session_state.openai_api_key
        st.session_state.env_firecrawl_api_key = st.session_state.firecrawl_api_key
        st.session_state.env_perplexity_api_key = st.session_state.perplexity_api_key
        st.session_state.env_exa_api_key = st.session_state.exa_api_key
        
        return True
    except Exception as e:
        st.error(f"Error saving API keys to .env file: {str(e)}")
        return False

# Sidebar for API keys
with st.sidebar:
    st.title("AI Competitor Intelligence")
    
    # Add search engine selection
    search_engine = st.selectbox(
        "Select Search Endpoint",
        options=["Perplexity AI - Sonar Pro", "Exa AI"],
        help="Choose which AI service to use for finding competitor URLs"
    )
    
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
        
        # Always required API keys
        has_openai = update_api_key("OpenAI", "env_openai_api_key")
        has_firecrawl = update_api_key("Firecrawl", "env_firecrawl_api_key")
        
        # Search engine specific API keys
        if search_engine == "Perplexity AI - Sonar Pro":
            has_search_engine = update_api_key("Perplexity", "env_perplexity_api_key")
        else:  # Exa AI
            has_search_engine = update_api_key("Exa", "env_exa_api_key")
        
        # Buttons for API key management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset to .env values"):
                st.session_state.openai_api_key = st.session_state.env_openai_api_key
                st.session_state.firecrawl_api_key = st.session_state.env_firecrawl_api_key
                st.session_state.perplexity_api_key = st.session_state.env_perplexity_api_key
                st.session_state.exa_api_key = st.session_state.env_exa_api_key
                st.rerun()
        
        with col2:
            if st.button("Save to .env file"):
                if save_api_keys_to_env():
                    st.success("API keys saved to .env file!")
                    st.rerun()
    
    # Display API status
    api_status_ok = has_openai and has_firecrawl and has_search_engine
    
    if api_status_ok:
        st.success("✅ All required API keys are configured")
    else:
        missing_keys = []
        if not has_openai:
            missing_keys.append("OpenAI")
        if not has_firecrawl:
            missing_keys.append("Firecrawl")
        if not has_search_engine:
            missing_keys.append("Search Engine")
        
        st.error(f"❌ Missing API keys: {', '.join(missing_keys)}")

# Main UI
st.title("🧲 AI Competitor Intelligence Agent Team")
st.info(
    """
    This app helps businesses analyze their competitors by extracting structured data from competitor websites and generating insights using AI.
    - Provide a **URL** or a **description** of your company.
    - The app will fetch competitor URLs, extract relevant information, and generate a detailed analysis report.
    """
)
st.success("For better results, provide both URL and a 5-6 word description of your company!")

# Input fields for URL and description
url = st.text_input("Enter your company URL :")
description = st.text_area("Enter a description of your company (if URL is not available):")

# Initialize API keys and tools
if api_status_ok:
    # Initialize Exa only if selected
    if search_engine == "Exa AI":
        exa = Exa(api_key=st.session_state.exa_api_key)

    firecrawl_tools = FirecrawlTools(
        api_key=st.session_state.firecrawl_api_key,
        scrape=False,
        crawl=True,
        limit=5
    )

    firecrawl_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state.openai_api_key),
        tools=[firecrawl_tools, DuckDuckGoTools()],
        show_tool_calls=True,
        markdown=True
    )

    analysis_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state.openai_api_key),
        show_tool_calls=True,
        markdown=True
    )

    # New agent for comparing competitor data
    comparison_agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini", api_key=st.session_state.openai_api_key),
        show_tool_calls=True,
        markdown=True
    )

    def get_competitor_urls(url: str = None, description: str = None) -> list[str]:
        if not url and not description:
            raise ValueError("Please provide either a URL or a description.")

        if search_engine == "Perplexity AI - Sonar Pro":
            perplexity_url = "https://api.perplexity.ai/chat/completions"
            
            content = "Find me 3 competitor company URLs similar to the company with "
            if url and description:
                content += f"URL: {url} and description: {description}"
            elif url:
                content += f"URL: {url}"
            else:
                content += f"description: {description}"
            content += ". ONLY RESPOND WITH THE URLS, NO OTHER TEXT."

            payload = {
                "model": "sonar-pro",
                "messages": [
                    {
                        "role": "system",
                        "content": "Be precise and only return 3 company URLs ONLY."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.2,
            }
            
            headers = {
                "Authorization": f"Bearer {st.session_state.perplexity_api_key}",
                "Content-Type": "application/json"
            }

            try:
                response = requests.post(perplexity_url, json=payload, headers=headers)
                response.raise_for_status()
                urls = response.json()['choices'][0]['message']['content'].strip().split('\n')
                return [url.strip() for url in urls if url.strip()]
            except Exception as e:
                st.error(f"Error fetching competitor URLs from Perplexity: {str(e)}")
                return []

        else:  # Exa AI
            try:
                if url:
                    result = exa.find_similar(
                        url=url,
                        num_results=3,
                        exclude_source_domain=True,
                        category="company"
                    )
                else:
                    result = exa.search(
                        description,
                        type="neural",
                        category="company",
                        use_autoprompt=True,
                        num_results=3
                    )
                return [item.url for item in result.results]
            except Exception as e:
                st.error(f"Error fetching competitor URLs from Exa: {str(e)}")
                return []

    class CompetitorDataSchema(BaseModel):
        company_name: str = Field(description="Name of the company")
        pricing: str = Field(description="Pricing details, tiers, and plans")
        key_features: List[str] = Field(description="Main features and capabilities of the product/service")
        tech_stack: List[str] = Field(description="Technologies, frameworks, and tools used")
        marketing_focus: str = Field(description="Main marketing angles and target audience")
        customer_feedback: str = Field(description="Customer testimonials, reviews, and feedback")

    def extract_competitor_info(competitor_url: str) -> Optional[dict]:
        try:
            # Initialize FirecrawlApp with API key
            app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)
            
            # Add wildcard to crawl subpages
            url_pattern = f"{competitor_url}/*"
            
            extraction_prompt = """
            Extract detailed information about the company's offerings, including:
            - Company name and basic information
            - Pricing details, plans, and tiers
            - Key features and main capabilities
            - Technology stack and technical details
            - Marketing focus and target audience
            - Customer feedback and testimonials
            
            Analyze the entire website content to provide comprehensive information for each field.
            """
            
            response = app.extract(
                [url_pattern],
                {
                    'prompt': extraction_prompt,
                    'schema': CompetitorDataSchema.model_json_schema(),
                }
            )
            
            if response.get('success') and response.get('data'):
                extracted_info = response['data']
                
                # Create JSON structure
                competitor_json = {
                    "competitor_url": competitor_url,
                    "company_name": extracted_info.get('company_name', 'N/A'),
                    "pricing": extracted_info.get('pricing', 'N/A'),
                    "key_features": extracted_info.get('key_features', [])[:5],  # Top 5 features
                    "tech_stack": extracted_info.get('tech_stack', [])[:5],      # Top 5 tech stack items
                    "marketing_focus": extracted_info.get('marketing_focus', 'N/A'),
                    "customer_feedback": extracted_info.get('customer_feedback', 'N/A')
                }
                
                return competitor_json
                
            else:
                return None
                
        except Exception as e:
            return None

    def generate_comparison_report(competitor_data: list) -> None:
        # Format the competitor data for the prompt
        formatted_data = json.dumps(competitor_data, indent=2)
        print(formatted_data)
        
        # Updated system prompt for more structured output
        system_prompt = f"""
        As an expert business analyst, analyze the following competitor data in JSON format and create a structured comparison.
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
        6. Return ONLY the structured data, no additional text
        """

        # Get comparison data from agent
        comparison_response = comparison_agent.run(system_prompt)
        
        try:
            # Split the response into lines and clean them
            table_lines = [
                line.strip() 
                for line in comparison_response.content.split('\n') 
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
            st.write("Raw comparison data for debugging:", comparison_response.content)

    def generate_analysis_report(competitor_data: list):
        # Format the competitor data for the prompt
        formatted_data = json.dumps(competitor_data, indent=2)
        print("Analysis Data:", formatted_data)  # For debugging
        
        report = analysis_agent.run(
            f"""Analyze the following competitor data in JSON format and identify market opportunities to improve my own company:
            
            {formatted_data}

            Tasks:
            1. Identify market gaps and opportunities based on competitor offerings
            2. Analyze competitor weaknesses that we can capitalize on
            3. Recommend unique features or capabilities we should develop
            4. Suggest pricing and positioning strategies to gain competitive advantage
            5. Outline specific growth opportunities in underserved market segments
            6. Provide actionable recommendations for product development and go-to-market strategy

            Focus on finding opportunities where we can differentiate and do better than competitors.
            Highlight any unmet customer needs or pain points we can address.
            """
        )
        return report.content

    # Run analysis when the user clicks the button
    if st.button("Analyze Competitors"):
        if not api_status_ok:
            st.error("⚠️ Please configure all required API keys in the sidebar before proceeding.")
        elif url or description:
            with st.spinner("Fetching competitor URLs..."):
                competitor_urls = get_competitor_urls(url=url, description=description)
                st.write(f"Competitor URLs: {competitor_urls}")
            
            competitor_data = []
            for comp_url in competitor_urls:
                with st.spinner(f"Analyzing Competitor: {comp_url}..."):
                    competitor_info = extract_competitor_info(comp_url)
                    if competitor_info is not None:
                        competitor_data.append(competitor_info)
            
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
            st.error("Please provide either a URL or a description.")
    else:
        # Display API key status message when the app first loads
        if not api_status_ok:
            st.warning("⚠️ Configure your API keys in the sidebar before analyzing competitors.")
