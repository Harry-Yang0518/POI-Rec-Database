# POI-Rec-Database

This repository contains a complete pipeline for extracting structured POI mentions from Japan travel Reddit posts and converting them into embeddings for downstream recommendation tasks.

## Pipeline Overview

The pipeline performs two main steps:

### 1. POI Extraction (LLM)
`poi_pipeline.py` extracts for each Reddit post:
- city
- poi_name
- poi_type
- landscape & content
- activities
- atmosphere
- time & schedule

and stores them in: poi_mentions.jsonl

### 2. Embeddings
The script then creates a unified text representation and generates embeddings using: text-embedding-3-large

Results saved to: poi_mentions_with_emb.jsonl

##  How to Run

### 1. Create environment
```bash
conda create -n poi-env python=3.10 -y
conda activate poi-env
pip install -r requirements.txt
# POI-Rec-Database

This repository contains a lightweight, single-file pipeline for extracting structured POI (Point of Interest) mentions from Japan travel–related Reddit posts and generating embeddings for downstream recommendation or retrieval tasks.

---

## 📌 Pipeline Overview

The pipeline runs in two stages:

### **1. POI Extraction (LLM-powered)**  
`poi_pipeline.py` reads each Reddit post and extracts:
- **city**
- **poi_name**
- **poi_type**
- **landscape & content**
- **activities**
- **atmosphere**
- **time & schedule**

These structured POI mentions are saved to:

```
poi_mentions.jsonl
```

---

### **2. Embedding Generation**  
For each POI mention, the script generates an embedding using:

```
text-embedding-3-large
```

and writes results to:

```
poi_mentions_with_emb.jsonl
```

---

## 🚀 How to Run the Pipeline

### **1. Create a Conda environment**
```bash
conda create -n poi-env python=3.10 -y
conda activate poi-env
pip install -r requirements.txt
```

### **2. Run POI extraction**
```bash
python poi_pipeline.py --step extract
```

### **3. Generate embeddings**
```bash
python poi_pipeline.py --step embed
```

### **4. Run both steps**
```bash
python poi_pipeline.py --step all
```

---

## 📂 Project Structure

```
POI-Rec-Database/
│
├── poi_pipeline.py
├── travel_japan.jsonl              # your crawled dataset
├── poi_mentions.jsonl              # extracted POIs (generated)
├── poi_mentions_with_emb.jsonl     # POIs + embeddings (generated)
├── requirements.txt
└── README.md
```

---

## 🔑 API Keys

Set your OpenAI API key before running:

```bash
export OPENAI_API_KEY="your-key-here"
```

Or configure via environment variables in your shell.

---

## 📝 Notes

- The pipeline is intentionally simple and self-contained.  
- For large datasets, batch processing or async calls can be added later.  
- This pipeline prepares data for downstream tasks such as recommendation, retrieval, or clustering.

---