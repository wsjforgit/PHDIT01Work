# PhD Thesis: Multi-Source Medical Information Retrieval with Instruction Following

## Chapter 3: Research Methodology

### 3.1 Overview

This research integrates multiple state-of-the-art instruction-aware retrieval approaches to build a comprehensive medical information retrieval system. We leverage datasets and methodologies from six seminal papers:

1. **TART** (Task-aware Retrieval with Instructions) - Multi-task instruction tuning
2. **INSTRUCTOR** - Instruction-finetuned text embeddings
3. **I3** - Intent-Introspective retrieval
4. **FollowIR** - Instruction following evaluation framework
5. **MAIR** - Massive instructed retrieval benchmark
6. **Instruction Embedding** - Latent task identification

### 3.2 Multi-Source Medical Datasets

Based on the surveyed literature, we construct a unified multi-source medical retrieval dataset:

#### 3.2.1 Primary Data Sources

| Dataset | Source | Size | Description | Task Type |
|---------|--------|------|-------------|-----------|
| **BioASQ** | Biomedical QA | 3,743 questions + PubMed articles | Biomedical semantic QA | Retrieval + QA |
| **TREC-COVID** | COVID-19 research | 50 topics, 191K articles (CORD-19) | Pandemic information retrieval | Ad-hoc retrieval |
| **TREC-CDS** | Clinical narratives | 90 case reports (2014-2016) | Clinical decision support | Case-based retrieval |
| **PubMed Central** | Open Access Subset | ~3M full-text articles | General biomedical literature | Document retrieval |
| **MedMCQA** | Medical exams | 194K MCQs | Medical reasoning | Classification |
| **PubMedQA** | Research QA | 1K expert-annotated | Evidence-based QA | QA + Evidence retrieval |

#### 3.2.2 Instruction Annotation Schema

Following the **FollowIR** methodology, we annotate each query with detailed instructions:

```json
{
  "query_id": "TREC-CDS-2014-01",
  "case_narrative": "A 58-year-old male presents with chest pain...",
  "instruction": "Retrieve articles that help diagnose the patient's condition based on the symptoms: chest pain, shortness of breath, elevated troponin levels. Focus on differential diagnosis for acute coronary syndrome.",
  "task_type": "diagnosis",
  "domain": "cardiology",
  "population_constraints": ["adult male", "age 55-65"],
  "safety_considerations": ["consider comorbidities", "exclude experimental treatments"]
}
```

### 3.3 System Architecture

Our system integrates three instruction-aware components:

```
┌─────────────────────────────────────────────────────────────┐
│                     Input: Medical Query                      │
│             + Detailed Clinical Instructions                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Component 1: Instruction Encoder (INSTRUCTOR-based)         │
│  • Domain-specific encoding: "medicine", "clinical"          │
│  • Task-specific encoding: "diagnosis", "treatment"          │
│  • Constraint parsing: population, safety, exclusions        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Component 2: Multi-Task Retriever (TART + I3)              │
│  • Intent introspection (I3 methodology)                     │
│  • Multi-source document encoding:                           │
│    - Clinical Guidelines (structured)                        │
│    - PubMed Articles (unstructured)                          │
│    - Clinical Trials (semi-structured)                       │
│    - Case Reports (narrative)                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Component 3: Instruction-Following Ranker (FollowIR)       │
│  • Evaluate instruction adherence                            │
│  • Safety constraint checking                                │
│  • Evidence completeness scoring                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Ranked Medical Evidence Results                │
│          + Instruction Adherence Scores                      │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Training Procedure

#### Phase 1: Pre-training on Medical Corpora
- **Corpus**: PubMed abstracts (33M), PMC full-text (3M)
- **Method**: Contrastive learning (INSTRUCTOR approach)
- **Duration**: 5 epochs on 8×A100 GPUs

#### Phase 2: Instruction Fine-tuning
- **Dataset**: MAIR medical subset (126 tasks) + BERRI medical tasks
- **Method**: Multi-task instruction tuning (TART approach)
- **Objective**: Minimize instruction-query-document triplet loss

#### Phase 3: Intent Introspection Training
- **Method**: I3 progressive pruning
- **Data**: LLM-generated instruction-intent pairs
- **Objective**: Learn to infer retrieval intent from instructions

### 3.5 Evaluation Framework

#### 3.5.1 Datasets for Evaluation

| Test Set | # Queries | # Documents | Instruction Type |
|----------|-----------|-------------|------------------|
| BioASQ-Test | 500 | PubMed (12M) | Biomedical QA |
| TREC-CDS-2016 | 30 cases | PMC (1.25M) | Clinical narratives |
| TREC-COVID-Round5 | 50 topics | CORD-19 (191K) | Pandemic retrieval |
| Custom Medical IR | 1,000 | Multi-source | Complex instructions |

#### 3.5.2 Evaluation Metrics

**Standard IR Metrics:**
- nDCG@10, nDCG@20
- Recall@100
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)

**Instruction-Following Metrics (FollowIR):**
- **Instruction Adherence Score (IAS)**: Pairwise evaluation of instruction following
- **Constraint Satisfaction Rate (CSR)**: % of results satisfying all constraints
- **Safety Compliance Score (SCS)**: % avoiding contraindicated/unsafe evidence

**Medical-Specific Metrics:**
- **Clinical Relevance Score (CRS)**: Expert-rated (0-2 scale)
- **Evidence Completeness (EC)**: Coverage of required evidence components
- **Diagnostic Accuracy (DA)**: For diagnosis tasks, agreement with gold standard

### 3.6 Baseline Models

| Model | Type | Parameters | Description |
|-------|------|------------|-------------|
| BM25 | Lexical | - | Traditional keyword matching |
| DPR | Dense retrieval | 110M | Dense passage retrieval |
| ANCE | Dense retrieval | 110M | Approximate nearest neighbor |
| ColBERT | Late interaction | 110M | Token-level matching |
| INSTRUCTOR-base | Instruction-tuned | 335M | Base instruction embedder |
| TART-full | Multi-task | 220M | Task-aware retrieval |
| **Ours (Integrated)** | Multi-component | 450M | Full system |

### 3.7 Implementation Details

**Hardware:**
- Training: 8× NVIDIA A100 (40GB)
- Inference: 4× NVIDIA V100 (32GB)

**Software Stack:**
```python
torch==2.1.0
transformers==4.36.0
faiss-gpu==1.7.4
sentence-transformers==2.2.2
pytrec_eval==0.5
```

**Hyperparameters:**
- Learning rate: 2e-5 (AdamW)
- Batch size: 64 (gradient accumulation × 4)
- Max sequence length: 512 (queries), 384 (documents)
- Temperature (contrastive): 0.05
- Hard negative ratio: 4:1

### 3.8 Ethical Considerations

1. **Data Privacy**: All datasets use de-identified, publicly available medical literature
2. **Clinical Safety**: System includes explicit safety warnings for high-risk queries
3. **Bias Mitigation**: Balanced sampling across demographics and medical specialties
4. **Responsible Use**: Clear disclaimers that system is for research/decision support, not clinical diagnosis

---

## Next Steps

Chapter 4 will present comprehensive experimental results, including:
- Comparative performance on all benchmark datasets
- Ablation studies on each component
- Case studies with expert clinical evaluation
- Error analysis and failure mode taxonomy
