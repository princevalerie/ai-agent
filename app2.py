import streamlit as st
import google.generativeai as genai
from agno.agent import Agent
from agno.tools.firecrawl import FirecrawlTools
from agno.tools.duckduckgo import DuckDuckGoTools
import pandas as pd
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os
from dotenv import load_dotenv, set_key
import pathlib

# Konfigurasi path .env
env_path = pathlib.Path(os.path.join(os.getcwd(), '.env'))
load_dotenv(dotenv_path=env_path)

# Konfigurasi UI Streamlit
st.set_page_config(page_title="Competitor Intelligence", layout="wide")

# Inisialisasi session state
if "api_keys_initialized" not in st.session_state:
    st.session_state.env_gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    st.session_state.env_firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")
    st.session_state.gemini_api_key = st.session_state.env_gemini_api_key
    st.session_state.firecrawl_api_key = st.session_state.env_firecrawl_api_key
    st.session_state.api_keys_initialized = True

# Model data
class CompetitorSchema(BaseModel):
    company_name: str = Field(description="Nama perusahaan")
    pricing: str = Field(description="Struktur harga dan paket")
    key_features: List[str] = Field(description="Fitur utama produk/jasa")
    marketing_focus: str = Field(description="Strategi pemasaran")
    customer_feedback: str = Field(description="Ulasan pelanggan")

# Fungsi ekstraksi data
def extract_data(url: str) -> Optional[dict]:
    try:
        app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)
        prompt = """
        Ekstrak informasi berikut dari website:
        - Nama perusahaan
        - Detail harga dan paket
        - Fitur utama (maksimal 5)
        - Strategi pemasaran
        - Testimoni pelanggan
        
        Hasilkan dalam format JSON.
        """
        
        response = app.extract(
            url_patterns=[f"{url}/*"],
            params={
                "prompt": prompt,
                "schema": CompetitorSchema.model_json_schema()
            }
        )
        
        if response.get('data'):
            return {
                "url": url,
                "company_name": response['data'].get('company_name', 'Tidak diketahui'),
                "pricing": response['data'].get('pricing', 'Tidak tersedia'),
                "key_features": response['data'].get('key_features', [])[:5],
                "marketing_focus": response['data'].get('marketing_focus', 'Tidak tersedia'),
                "customer_feedback": response['data'].get('customer_feedback', 'Tidak tersedia')
            }
        return None
    except Exception as e:
        st.error(f"Gagal mengekstrak {url}: {str(e)}")
        return None

# Fungsi analisis
def generate_insights(company_data: dict, competitors_data: list) -> str:
    genai.configure(api_key=st.session_state.gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    Analisis perbandingan kompetitor untuk:
    Perusahaan Utama: {company_data['company_name']}
    
    Data Perusahaan Utama:
    {json.dumps(company_data, indent=2)}
    
    Data Kompetitor:
    {json.dumps(competitors_data, indent=2)}
    
    Buat laporan yang berisi:
    1. Perbandingan fitur utama
    2. Analisis harga
    3. Keunggulan kompetitif
    4. Rekomendasi strategi
    5. Peluang pasar
    
    Format dalam Markdown dengan header yang jelas.
    Gunakan bahasa Indonesia formal.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# UI Utama
st.title("📊 Competitive Intelligence Analyzer")
st.write("Analisis kompetitor bisnis Anda secara otomatis")

# Input utama
company_url = st.text_input("URL Perusahaan Anda", help="Masukkan URL website perusahaan Anda")
num_competitors = st.number_input("Jumlah Kompetitor", 1, 5, 3)
competitor_urls = [st.text_input(f"URL Kompetitor #{i+1}") for i in range(num_competitors)]

# Fungsi helper untuk tabel fitur
def create_feature_table(company_data, competitors_data):
    # Cari jumlah fitur maksimum
    max_features = max(
        len(company_data['key_features']),
        max(len(c['key_features']) for c in competitors_data)
    )
    
    # Buat header kolom
    feature_columns = [f'Fitur {i+1}' for i in range(max_features)]
    
    # Fungsi helper
    def get_feature(features, index):
        return features[index] if index < len(features) else '-'
    
    # Bangun data tabel
    table_data = {
        "Perusahaan": [company_data['company_name']] + [c['company_name'] for c in competitors_data]
    }
    
    for i in range(max_features):
        table_data[f'Fitur {i+1}'] = (
            [get_feature(company_data['key_features'], i)] +
            [get_feature(c['key_features'], i) for c in competitors_data]
        )
    
    return pd.DataFrame(table_data)

if st.button("Mulai Analisis"):
    # Validasi input
    if not company_url or not all(competitor_urls):
        st.error("Harap isi semua URL yang diperlukan")
        st.stop()
        
    if not st.session_state.gemini_api_key or not st.session_state.firecrawl_api_key:
        st.error("Harap konfigurasi API key di sidebar")
        st.stop()

    # Ekstraksi data
    with st.spinner("Menganalisis perusahaan Anda..."):
        company_data = extract_data(company_url)
        
    competitors_data = []
    for url in competitor_urls:
        with st.spinner(f"Menganalisis {url}..."):
            if data := extract_data(url):
                competitors_data.append(data)
    
    if not company_data or not competitors_data:
        st.error("Gagal mendapatkan data yang diperlukan")
        st.stop()

    # Tampilkan data mentah
    st.subheader("📂 Data Mentah")
    with st.expander("Lihat Data Perusahaan Anda"):
        st.json(company_data)
        
    with st.expander("Lihat Data Kompetitor"):
        for data in competitors_data:
            st.json(data)

    # Generate laporan analisis
    st.subheader("📈 Analisis Kompetitif")
    analysis = generate_insights(company_data, competitors_data)
    st.markdown(analysis)

    # Tabel perbandingan fitur
    st.subheader("🔍 Perbandingan Fitur Utama")
    feature_df = create_feature_table(company_data, competitors_data)
    st.dataframe(
        feature_df,
        use_container_width=True,
        height=(len(feature_df) + 1) * 35 + 3
    )

# Sidebar untuk API keys
with st.sidebar:
    st.title("⚙️ Konfigurasi")
    with st.expander("Kelola API Keys", expanded=True):
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.env_gemini_api_key,
            type="password"
        )
        
        st.session_state.firecrawl_api_key = st.text_input(
            "Firecrawl API Key", 
            value=st.session_state.env_firecrawl_api_key,
            type="password"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset API Keys"):
                st.session_state.gemini_api_key = st.session_state.env_gemini_api_key
                st.session_state.firecrawl_api_key = st.session_state.env_firecrawl_api_key
        with col2:
            if st.button("Simpan ke .env"):
                try:
                    set_key(env_path, "GEMINI_API_KEY", st.session_state.gemini_api_key)
                    set_key(env_path, "FIRECRAWL_API_KEY", st.session_state.firecrawl_api_key)
                    st.success("API keys tersimpan!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

st.info("""
**Petunjuk Penggunaan:**
1. Masukkan API keys di sidebar
2. Input URL perusahaan Anda
3. Tentukan jumlah dan URL kompetitor
4. Klik tombol "Mulai Analisis"
""")
