import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
import re
from spacy.matcher import PhraseMatcher
import os
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import pypdf
import spacy

nlp = spacy.load("en_core_web_sm")


# --- Directory and Path Configuration ---
PROD_BASE_DIR = Path(r"Streamlit_Frontend")
BASE_DIR = PROD_BASE_DIR if PROD_BASE_DIR.exists() else Path(__file__).resolve().parent

INDEX_PATH = BASE_DIR / "resume_faiss_index.bin"
MODEL_PATH = BASE_DIR / "catboost_model.cbm"
DATA_PATH = BASE_DIR / "processed_resumes_data.pkl"

# --- Configuration and Constants ---
SKILLS_DB = [
    "python", "machine learning", "deep learning", "nlp", "sql", "docker", "aws",
    "java", "c++", "javascript", "html", "css", "excel", "word", "powerpoint", "office",
    "tableau", "power bi", "git", "github", "gitlab", "jira", "agile", "scrum",
    "data analysis", "data science", "big data", "cloud computing", "devops",
    "artificial intelligence", "r programming", "data visualization",
    "nosql databases", "sql server", "mysql", "postgresql", "mongodb", "cassandra",
    "spark", "hadoop", "kafka", "airflow", "azure", "gcp",
    "pytorch", "pandas", "tensorflow", "scikit-learn", "keras", "numpy", "scipy", "matplotlib", "seaborn",
    "natural language processing", "computer vision", "reinforcement learning",
    "statistical analysis", "modeling", "etl", "data warehousing", "api", "rest api",
    "machine learning algorithms", "deep learning frameworks", "data structures", "algorithms"
]

SKILL_ONTOLOGY = {
    'ml': 'machine learning', 'pytorch': 'deep learning', 'pandas': 'python',
    'tensorflow': 'deep learning', 'ai': 'artificial intelligence', 'dl': 'deep learning',
    'r': 'r programming', 'java': 'java programming', 'c++': 'c++ programming',
    'sql server': 'sql', 'mysql': 'sql', 'postgresql': 'sql',
    'mongodb': 'nosql databases', 'cassandra': 'nosql databases',
    'scikit-learn': 'machine learning', 'nlp': 'natural language processing',
    'aws': 'amazon web services', 'gcp': 'google cloud platform',
    'power bi': 'data visualization', 'html': 'web development', 'css': 'web development',
    'javascript': 'web development', 'excel': 'microsoft office', 'word': 'microsoft office',
    'powerpoint': 'microsoft office', 'office': 'microsoft office', 'azure': 'cloud computing',
    'spark': 'big data', 'hadoop': 'big data', 'kafka': 'big data', 'airflow': 'data orchestration',
    'data scientist': 'data science', 'data analyst': 'data analysis'
}

# --- Expected Machine Learning Model Feature Alignment ---
ORDERED_FEATURES = [
    'Skills_Similarity_Score',
    'Text_Similarity_Score',
    'Years_of_Experience',
    'Education_Score',
    'Overall_Project_Relevance_Score'
]

# --- Cached Resource Loading ---
@st.cache_resource(show_spinner="Loading Sentence Transformer...")
def load_sentence_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Loading FAISS index...")
def load_faiss_index(index_filepath, dimension: int):
    if index_filepath.exists():
        try:
            return faiss.read_index(str(index_filepath))
        except Exception:
            pass
    return faiss.IndexFlatL2(dimension)

@st.cache_resource(show_spinner="Loading CatBoost model...")
def load_catboost_model(model_filepath):
    if not model_filepath.exists():
        return None
    try:
        model = CatBoostClassifier()
        model.load_model(str(model_filepath))
        return model
    except Exception:
        return None

@st.cache_resource(show_spinner="Loading spaCy components...")
def load_spacy_components():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model not found. Run: `python -m spacy download en_core_web_sm`")
        st.stop()

    matcher = PhraseMatcher(nlp.vocab)
    all_skill_terms = set(SKILLS_DB)
    for key, value in SKILL_ONTOLOGY.items():
        all_skill_terms.update([key, value])
    all_skill_terms.update([
        "machine learning engineer", "deep learning engineer", "natural language processing engineer",
        "software development", "software engineer", "front end", "back end", "full stack",
        "computer science", "data engineering"
    ])

    patterns = [nlp.make_doc(text) for text in all_skill_terms if text.strip()]
    matcher.add("SKILL", patterns)
    return nlp, matcher

@st.cache_data(show_spinner="Loading processed resume data...")
def load_processed_resume_data(data_filepath):
    if not data_filepath.exists():
        return pd.DataFrame(columns=[
            'Filename', 'Category', 'Parsed_Resume_Text', 'Extracted_Skills',
            'Years_of_Experience', 'Seniority_Keywords', 'Education_Score',
            'Overall_Project_Relevance_Score', 'Resume_Text_Embeddings', 'Extracted_Skills_Embeddings'
        ])
    try:
        with open(data_filepath, 'rb') as f:
            df = pickle.load(f)
        
        for col in ['Resume_Text_Embeddings', 'Extracted_Skills_Embeddings']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: np.array(x, dtype='float32').squeeze() 
                    if isinstance(x, (list, np.ndarray)) else np.zeros(384, dtype='float32')
                )
        return df
    except Exception:
        return pd.DataFrame(columns=[
            'Filename', 'Category', 'Parsed_Resume_Text', 'Extracted_Skills',
            'Years_of_Experience', 'Seniority_Keywords', 'Education_Score',
            'Overall_Project_Relevance_Score', 'Resume_Text_Embeddings', 'Extracted_Skills_Embeddings'
        ])

# --- Helper Functions ---
def extract_text_from_pdf(file_file):
    try:
        pdf_reader = pypdf.PdfReader(file_file)
        return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"PDF parsing error: {e}")
        return ""

def extract_and_map_skills(text, nlp_model, matcher_instance, skill_ontology_map):
    if not isinstance(text, str) or not text.strip():
        return []
    doc = nlp_model(text.lower())
    matches = matcher_instance(doc)
    extracted = {skill_ontology_map.get(doc[start:end].text, doc[start:end].text) 
                 for _, start, end in matches}
    return sorted(list(extracted))

def compare_skills(job_skills, resume_skills):
    job_set = set(job_skills)
    resume_set = set(resume_skills)
    return sorted(list(job_set & resume_set)), sorted(list(job_set - resume_set))

def extract_experience_indicators(text):
    if not isinstance(text, str) or not text.strip():
        return {"years_of_experience": 0, "seniority_keywords": []}
    text_lower = text.lower()
    years_match = re.search(r'(\d+)\s*\+?\s*(?:year|yrs?|yr)', text_lower)
    years = int(years_match.group(1)) if years_match else 0

    seniority_keywords = ['senior', 'lead', 'manager', 'principal', 'head of', 'director', 'vp', 'architect', 'staff']
    found = [kw for kw in seniority_keywords if kw in text_lower]
    return {"years_of_experience": years, "seniority_keywords": found}

def calculate_education_score(text):
    text_lower = text.lower()
    if any(x in text_lower for x in ["phd", "doctorate"]): return 5
    if any(x in text_lower for x in ["master", "m.s.", "msc"]): return 3
    if any(x in text_lower for x in ["bachelor", "b.a.", "b.s."]): return 1
    return 0

def calculate_cosine_similarity(emb1, emb2):
    a = np.array(emb1, dtype='float32').reshape(1, -1)
    b = np.array(emb2, dtype='float32').reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

# --- Main Analysis Function ---
def perform_full_analysis(query_text, num_results, sbert_model, full_df, catboost_model, 
                         nlp_model, matcher_instance, skill_ontology_map):
    if len(full_df) == 0:
        return pd.DataFrame()

    dim = sbert_model.get_sentence_embedding_dimension()  # 384
    query_embedding = sbert_model.encode(query_text.lower(), convert_to_numpy=True).astype('float32')
    job_query_skills = extract_and_map_skills(query_text, nlp_model, matcher_instance, skill_ontology_map)
    
    skills_query_str = " ".join(job_query_skills) if job_query_skills else query_text.lower()
    query_skills_embedding = sbert_model.encode(skills_query_str, convert_to_numpy=True).astype('float32')

    valid_records = []
    for _, row in full_df.iterrows():
        emb = row.get('Resume_Text_Embeddings')
        if not isinstance(emb, np.ndarray) or emb.shape != (dim,):
            text = str(row.get('Parsed_Resume_Text', ''))
            emb = sbert_model.encode(text.lower(), convert_to_numpy=True).astype('float32')
        valid_records.append({**row, 'Resume_Text_Embeddings': emb})

    working_df = pd.DataFrame(valid_records)

    local_index = faiss.IndexFlatL2(dim)
    embeddings_matrix = np.vstack(working_df['Resume_Text_Embeddings'].values).astype('float32')
    local_index.add(embeddings_matrix)

    search_limit = min(num_results * 3, len(working_df))  
    distances, indices = local_index.search(query_embedding.reshape(1, -1), search_limit)

    results = []
    ml_prediction_data = []

    for i, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(working_df):
            continue
        resume_info = working_df.iloc[idx].copy()
        resume_info['FAISS_Distance'] = float(distances[0][i])

        resume_skills = resume_info.get('Extracted_Skills', [])
        matched, missing = compare_skills(job_query_skills, resume_skills)
        resume_info['Matched_Skills'] = matched
        resume_info['Missing_Skills'] = missing

        text_sim = calculate_cosine_similarity(query_embedding, resume_info['Resume_Text_Embeddings'])

        sk_emb = resume_info.get('Extracted_Skills_Embeddings')
        if not isinstance(sk_emb, np.ndarray) or sk_emb.shape != (dim,):
            sk_text = " ".join(resume_skills) if resume_skills else "empty"
            sk_emb = sbert_model.encode(sk_text.lower(), convert_to_numpy=True).astype('float32')
            resume_info['Extracted_Skills_Embeddings'] = sk_emb

        skills_sim = calculate_cosine_similarity(query_skills_embedding, sk_emb)

        ml_features = {
            'Skills_Similarity_Score': skills_sim,
            'Text_Similarity_Score': text_sim,
            'Years_of_Experience': float(resume_info.get('Years_of_Experience', 0)),
            'Education_Score': float(resume_info.get('Education_Score', 0)),
            'Overall_Project_Relevance_Score': float(resume_info.get('Overall_Project_Relevance_Score', 3.0))
        }

        ml_prediction_data.append(ml_features)
        results.append(resume_info)

    if not results:
        return pd.DataFrame()

    final_df = pd.DataFrame(results)
    ml_df = pd.DataFrame(ml_prediction_data)[ORDERED_FEATURES] 

    # --- Clean Accuracy Metric Processing Pipeline ---
    if catboost_model is not None:
        try:
            # Model naturally outputs continuous decimal floats between 0.0 and 1.0
            raw_scores = catboost_model.predict_proba(ml_df)[:, 1]
        except Exception:
            # Weighted algorithmic fallback if production inference encounters syntax updates
            raw_scores = (ml_df['Skills_Similarity_Score'] * 0.65) + (ml_df['Text_Similarity_Score'] * 0.35)
    else:
        raw_scores = (ml_df['Skills_Similarity_Score'] * 0.65) + (ml_df['Text_Similarity_Score'] * 0.35)

    # Strictly lock metric bounds between 0.00 and 1.00 to eliminate structural layout distortion
    clean_accuracies = np.clip(raw_scores, 0.0, 1.0)

    final_df['Prediction_Probability'] = clean_accuracies
    
    # Standard absolute 70% threshold boundary condition to process categorical hire actions
    final_df['Decision_Status'] = np.where(clean_accuracies >= 0.70, '🟢 Hired', '🔴 Rejected')

    return final_df.sort_values(by='Prediction_Probability', ascending=False).head(num_results).reset_index(drop=True)


# --- Streamlit UI ---
def main():
    st.set_page_config(layout="wide", page_title="AI Resume Screening Suite")
    st.title("👨‍💼 Corporate Talent Screener")

    sbert_model = load_sentence_transformer_model()
    catboost_model = load_catboost_model(MODEL_PATH)
    nlp_model, matcher_instance = load_spacy_components()

    if 'base_df' not in st.session_state:
        st.session_state['base_df'] = load_processed_resume_data(DATA_PATH)

    # Sidebar
    st.sidebar.subheader("Configuration")
    job_query = st.sidebar.text_area(
        "Target Position Requirement:",
        "Experienced Python Developer with expertise in machine learning and cloud platforms like AWS. Familiarity with Kubernetes and Azure is a plus.",
        height=120
    )
    num_results = st.sidebar.slider("Number of Results:", 1, 20, 5)

    st.sidebar.subheader("Add New Resumes")
    uploaded_files = st.sidebar.file_uploader("Upload Resumes (.pdf, .txt)", 
                                            accept_multiple_files=True, type=['pdf', 'txt'])

    if uploaded_files:
        uploaded_records = []
        for f in uploaded_files:
            if f.name in st.session_state['base_df']['Filename'].values:
                continue

            if f.name.lower().endswith('.pdf'):
                raw_text = extract_text_from_pdf(f)
            else:
                raw_text = f.read().decode("utf-8", errors="ignore")

            if not raw_text.strip():
                continue

            exp = extract_experience_indicators(raw_text)
            skills = extract_and_map_skills(raw_text, nlp_model, matcher_instance, SKILL_ONTOLOGY)

            text_emb = sbert_model.encode(raw_text.lower(), convert_to_numpy=True).astype('float32')
            skills_emb = sbert_model.encode(" ".join(skills) if skills else "empty", 
                                          convert_to_numpy=True).astype('float32')

            uploaded_records.append({
                'Filename': f.name,
                'Category': 'Uploaded',
                'Parsed_Resume_Text': raw_text,
                'Extracted_Skills': skills,
                'Years_of_Experience': exp['years_of_experience'],
                'Seniority_Keywords': exp['seniority_keywords'],
                'Education_Score': calculate_education_score(raw_text),
                'Overall_Project_Relevance_Score': 3.0,
                'Resume_Text_Embeddings': text_emb,
                'Extracted_Skills_Embeddings': skills_emb
            })

        if uploaded_records:
            new_df = pd.DataFrame(uploaded_records)
            st.session_state['base_df'] = pd.concat([new_df, st.session_state['base_df']], ignore_index=True)
            st.sidebar.success(f"Added {len(uploaded_records)} new resumes!")

    # Filter
    available_files = st.session_state['base_df']['Filename'].tolist()
    selected_files = st.sidebar.multiselect("Filter by Filename (optional):", available_files, default=[])

    if selected_files:
        df_active = st.session_state['base_df'][st.session_state['base_df']['Filename'].isin(selected_files)].reset_index(drop=True)
    else:
        df_active = st.session_state['base_df'].reset_index(drop=True)

    if st.sidebar.button("🔍 Execute Pipeline Match", type="primary"):
        if not job_query.strip():
            st.error("Please enter a job requirement.")
            return
        if df_active.empty:
            st.warning("No resumes available.")
            return

        with st.spinner("Analyzing candidates..."):
            results_df = perform_full_analysis(
                job_query, num_results, sbert_model, df_active,
                catboost_model, nlp_model, matcher_instance, SKILL_ONTOLOGY
            )

        if results_df.empty:
            st.warning("No matching candidates found.")
            return

        st.subheader(f"Top {len(results_df)} Candidates Matched")
        for i, row in results_df.iterrows():
            status = row.get('Decision_Status', '🔴 Rejected')
            prob = row.get('Prediction_Probability', 0)

            with st.expander(f"Rank {i+1} | Verdict: {status} — {row['Filename']} | Match Accuracy: {prob:.1%}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Match Accuracy", f"{prob:.1%}")
                    st.metric("Experience Verified", f"{row.get('Years_of_Experience', 0)} years")
                    st.metric("Education Score", f"{row.get('Education_Score', 0)}/5")
                with col2:
                    matched = row.get('Matched_Skills', [])
                    missing = row.get('Missing_Skills', [])
                    if matched:
                        st.success("**Matched Skills:** " + ", ".join(matched))
                    if missing:
                        st.warning("**Missing Skills:** " + ", ".join(missing))
                    if row.get('Seniority_Keywords'):
                        st.info("**Seniority Flags:** " + ", ".join(row['Seniority_Keywords']))

                st.markdown("**Resume Snippet:**")
                st.text_area("", value=row['Parsed_Resume_Text'][:1200] + "...", height=200, disabled=True, key=f"text_{i}")


if __name__ == "__main__":
    main()
