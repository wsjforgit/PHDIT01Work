# Title Page

**Title:** Research on Multi-Source Data Medical Information Retrieval Enhanced by Command Perception and Reasoning

**Authors:** [Insert Author Names Here]
**Affiliations:** [Insert Affiliations Here]
**Corresponding Author:** [Insert Name and Email]

**Abstract**
*Objective:* Biomedical information retrieval (IR) systems often struggle to interpret complex clinical instructions, integrate heterogeneous evidence sources, and ensure patient safety. This study proposes a novel framework to address these limitations by enhancing retrieval with command perception, multi-source integration, and explicit safety reasoning.
*Materials and Methods:* We constructed a multi-source biomedical dataset integrating clinical guidelines, randomized controlled trials, case reports, and knowledge graphs. We developed a retrieval framework comprising an Instruction-Aware Query Encoder to capture clinical intent, a Multi-Source Document Encoder for unified evidence representation, a Chain-of-Retrieval Reasoning Module for iterative evidence synthesis, and a Safety Constraint Checker to filter unsafe content. A new metric, MedFol, was introduced to evaluate relevance, evidence sufficiency, and safety.
*Results:* Experiments comparing the proposed system against BM25, Dense Passage Retrieval (DPR), and Retrieval-Augmented Generation (RAG) baselines demonstrated significant improvements in retrieval relevance for constraint-rich queries. The Chain-of-Retrieval module improved evidence completeness, while the Safety Constraint Checker significantly reduced the retrieval of contraindicated or harmful evidence compared to baselines.
*Conclusion:* The proposed framework effectively aligns biomedical retrieval with clinical reasoning processes, ensuring that retrieved evidence is not only semantically relevant but also clinically safe and complete.

**Keywords:** Biomedical Information Retrieval; Natural Language Processing; Clinical Decision Support; AI Safety; Multi-Source Integration.

---

# 1. Introduction

The volume of biomedical literature and clinical data has grown exponentially, creating a critical need for effective information retrieval (IR) systems to support evidence-based decision-making. Unlike general web search, biomedical IR requires high precision, the ability to interpret complex clinical constraints, and strict adherence to safety standards [1]. Clinicians frequently formulate queries with multiple layers of logic, such as "Which anticoagulants are suitable for elderly patients with atrial fibrillation and eGFR <30 mL/min who are intolerant to CYP3A4-metabolized drugs?" Traditional keyword-based models (e.g., BM25) and standard dense retrieval methods often fail to capture these nuances, leading to the retrieval of incomplete or clinically inappropriate evidence [2, 3].

Recent advances in transformer-based large language models (LLMs), such as BioBERT [4] and instruction-tuned variants [5], have improved semantic understanding. However, significant gaps remain in their application to clinical retrieval. First, most systems rely on single-source data (e.g., PubMed abstracts), ignoring the multi-source nature of clinical evidence which spans guidelines, trials, case reports, and knowledge bases [6]. Second, existing models struggle to interpret "clinician-style" instructions that contain exclusions, temporal dependencies, and contraindications [7]. Third, standard retrieval pipelines lack the reasoning capabilities required to synthesize evidence across these heterogeneous sources. Finally, and most critically, current systems lack explicit safety validation mechanisms, posing a risk of retrieving contraindicated or hallucinated information [8, 9].

To address these challenges, this study proposes a comprehensive biomedical retrieval framework enhanced by command perception and reasoning. The contributions of this work are as follows:

1.  **Multi-Source Dataset Construction:** We introduce a novel dataset that integrates clinical guidelines, trial criteria, case reports, and knowledge graph triples, annotated with instruction-style queries and safety labels.
2.  **Instruction-Aware Encoding:** We develop a query encoder specifically designed to parse and represent nested clinical constraints, improving the interpretation of complex medical instructions.
3.  **Chain-of-Retrieval Reasoning:** We implement an iterative reasoning module that refines retrieval by cross-referencing evidence from multiple sources to ensure completeness and consistency.
4.  **Safety Constraint Checking:** We introduce a hybrid safety validation mechanism combining rule-based logic, knowledge graphs, and LLM verification to filter unsafe or contraindicated evidence.
5.  **MedFol Evaluation Metric:** We propose a new evaluation metric that assesses retrieval quality based on relevance, factual accuracy, evidence sufficiency, and safety compliance.

The remainder of this paper is organized as follows: Section 2 reviews related work in biomedical IR and safety. Section 3 details the methodology, including dataset construction and model architecture. Section 4 presents the experimental results and analysis. Section 5 discusses the implications of the findings, and Section 6 concludes the study.

---

# 2. Related Work

## 2.1 Biomedical Information Retrieval

Traditional biomedical IR systems have relied on lexical matching approaches such as BM25 [10], which struggle with the semantic complexity and specialized terminology of clinical text [11]. The introduction of neural IR models, particularly transformer-based architectures, marked a significant advance. BioBERT [4], ClinicalBERT [12], and PubMedBERT [13] demonstrated improved performance on biomedical QA and document retrieval tasks by leveraging domain-specific pre-training. Dense Passage Retrieval (DPR) [14] further advanced semantic matching by encoding queries and documents into dense vector spaces.

However, these approaches exhibit critical limitations in clinical contexts. First, they primarily operate on single-source corpora (e.g., PubMed abstracts), failing to integrate the heterogeneous evidence types—guidelines, trials, case reports, and knowledge graphs—that clinicians routinely consult [6, 15]. Second, semantic similarity alone is insufficient for clinical queries that embed complex constraints such as contraindications, population filters, and temporal dependencies [7]. Third, existing dense retrievers lack mechanisms for multi-hop reasoning across documents, which is essential for synthesizing evidence from multiple sources [16].

## 2.2 Instruction-Aware Models and Clinical Query Understanding

Instruction tuning has emerged as a powerful technique for aligning LLMs with human intent [5, 17]. Models such as InstructGPT and FLAN-T5 demonstrate improved task performance through fine-tuning on instruction-following datasets. However, general-purpose instruction-tuned models struggle with domain-specific clinical constraints. Ben Abacha and Demner-Fushman [7] showed that existing biomedical QA systems frequently misinterpret queries involving exclusions, comorbidities, or pharmacological restrictions.

Recent work on instruction-aware retrieval (e.g., TART [18]) has shown promise in general domains, but these approaches have not been adapted to handle the unique constraint structures of clinical queries. Clinical instructions often encode nested logic (e.g., "suitable for patients with renal impairment but not on dialysis"), which requires specialized parsing and representation mechanisms beyond standard instruction tuning.

## 2.3 Multi-Source Evidence Integration and Reasoning

Clinical decision-making inherently requires triangulating evidence across multiple sources. Marshall et al. [6] highlighted the importance of integrating guidelines, trials, and systematic reviews for evidence-based practice. However, most retrieval benchmarks (e.g., BioASQ, TREC-CDS) use single-source data, preventing models from learning cross-evidence reasoning [11, 19].

Multi-hop reasoning has been explored in general-domain QA (e.g., HotpotQA [20]), but these approaches lack biomedical grounding and safety constraints. Retrieval-Augmented Generation (RAG) [21] combines retrieval with generation to improve factual grounding, and biomedical variants such as Self-BioRAG [22] have shown promise. However, RAG systems remain vulnerable to hallucination and do not explicitly validate the safety or completeness of retrieved evidence [8, 9].

## 2.4 Safety and Trustworthiness in Clinical AI

Safety is a critical concern in clinical AI systems. Recent evaluations of GPT-4 and similar LLMs reveal high rates of hallucination, fabricated citations, and unsafe medical recommendations [8, 9]. Despite these risks, existing IR metrics (e.g., Recall@k, nDCG) focus solely on relevance and do not assess factual accuracy, evidence sufficiency, or safety compliance [23].

Efforts to improve LLM safety have primarily focused on generation-time interventions (e.g., reinforcement learning from human feedback) rather than retrieval-time filtering. Knowledge graph–based approaches [24] offer structured reasoning capabilities but cannot interpret narrative constraints or detect subtle contradictions across textual sources. The absence of safety-aware retrieval mechanisms represents a fundamental gap in clinical information systems.

## 2.5 Research Gaps

The literature review reveals five persistent gaps:
1. **Lack of multi-source retrieval frameworks** that integrate guidelines, trials, case reports, and knowledge graphs.
2. **Inadequate instruction interpretation** for constraint-rich clinical queries.
3. **Absence of reasoning-based retrieval pipelines** that iteratively refine evidence synthesis.
4. **Limited safety validation mechanisms** in IR systems to filter contraindicated or hallucinated evidence.
5. **No safety-aware evaluation metrics** that assess retrieval quality beyond relevance.

This study addresses these gaps through a comprehensive framework that integrates instruction-aware encoding, multi-source representation, chain-of-retrieval reasoning, and hybrid safety validation.

---

# 3. Materials and Methods

## 3.1 Overview

We developed a biomedical retrieval framework consisting of four core components: (1) an Instruction-Aware Query Encoder for parsing clinical constraints, (2) a Multi-Source Document Encoder for unified evidence representation, (3) a Chain-of-Retrieval Reasoning Module for iterative evidence synthesis, and (4) a Safety Constraint Checker for filtering unsafe content. Figure 1 illustrates the system architecture.

## 3.2 Dataset Construction

### 3.2.1 Evidence Sources

We constructed a multi-source dataset integrating four evidence types:

- **Clinical Guidelines:** Structured recommendations, contraindications, and dosage rules extracted from authoritative sources (e.g., ACC/AHA guidelines, NICE guidelines).
- **Clinical Trials:** Inclusion/exclusion criteria, endpoints, and adverse events from ClinicalTrials.gov.
- **Case Reports:** Narrative descriptions of complex or rare clinical scenarios from PubMed Central.
- **Biomedical Knowledge Graphs:** Structured drug–disease, drug–drug, and symptom–disease relationships from UMLS, DrugBank, and SIDER.

Each evidence type contributes unique value: guidelines provide recommendations and safety warnings; trials define population eligibility and outcomes; case reports capture real-world edge cases; and knowledge graphs support pharmacological reasoning.

### 3.2.2 Query Construction

We designed 500 instruction-style clinical queries based on patterns observed in real clinical practice. Queries were constructed to:
- Encode multiple constraints (demographic, pharmacological, temporal, exclusion conditions)
- Require multi-source evidence synthesis
- Highlight potential safety pitfalls (e.g., contraindicated medications)

Examples include:
- "Identify anticoagulants safe for AF patients with eGFR <30 mL/min and contraindicated to CYP3A4 substrates."
- "Which lipid-lowering agents are guideline-recommended for statin-intolerant patients with diabetes?"

### 3.2.3 Safety Annotation Schema

Each query–evidence pair was annotated using a four-class schema:
- **Safe:** Clinically appropriate, no contradictions
- **Conditionally Safe:** Depends on specific subpopulation features
- **Unsafe:** Violates guidelines, contraindications, or trial criteria
- **Incomplete:** Missing critical evidence needed for safe decision-making

Three clinical experts (two physicians, one clinical pharmacist) independently annotated the dataset, with inter-annotator agreement (Fleiss' κ = 0.78) indicating substantial consensus.

### 3.2.4 Multi-Source Alignment

To enable reasoning across evidence types, we created cross-source alignment labels identifying:
- Guideline–trial consistency
- Trial–case report relationships
- Knowledge graph relations supporting textual claims
- Conflicting evidence requiring reconciliation

The final dataset comprises 500 queries, 12,000 evidence documents (3,000 guidelines, 4,000 trials, 3,000 case reports, 2,000 KG triples), and 15,000 query–evidence pairs with safety annotations.

## 3.3 System Architecture

### 3.3.1 Instruction-Aware Query Encoder

The query encoder extends a PubMedBERT backbone with three specialized components:

1. **Constraint Parsing Layer:** Segments queries into constraint units (e.g., "eGFR <30 mL/min"), clinical entities (drug, disease, biomarker terms), and safety cues ("contraindicated", "avoid").
2. **Clinical Ontology Alignment:** Maps extracted entities to UMLS concepts using MetaMap, enabling semantic grounding.
3. **Cross-Constraint Attention:** Models interactions between constraints (e.g., how renal impairment affects drug selection) using multi-head attention.

The encoder produces a structured representation **q** ∈ ℝ^d that captures clinical intent beyond semantic similarity.

### 3.3.2 Multi-Source Document Encoder

To align heterogeneous evidence sources, we developed a unified encoding strategy:

- **Guidelines:** Emphasize recommendation strength (Class I/II/III) and contraindications using specialized tokens.
- **Trials:** Emphasize eligibility criteria and population constraints via attention masking.
- **Case Reports:** Standard transformer encoding of narrative text.
- **Knowledge Graph Triples:** Graph neural network (GNN) encoding of relational structure, projected into the same embedding space.

A cross-source fusion layer integrates these representations via multi-head attention, producing unified document embeddings **d** ∈ ℝ^d.

### 3.3.3 Chain-of-Retrieval Reasoning Module

The reasoning module performs iterative evidence synthesis:

1. **Initial Retrieval:** Retrieve top-k documents based on cosine similarity between **q** and **d**.
2. **Evidence Evaluation:** Assess completeness (are all constraint requirements met?) and consistency (do guideline and trial evidence align?).
3. **Query Refinement:** If evidence is incomplete or contradictory, refine **q** by adding missing constraints or adjusting weights.
4. **Secondary Retrieval:** Retrieve additional evidence to fill gaps.
5. **Cross-Source Triangulation:** Validate consistency across guidelines ↔ trials ↔ knowledge graphs.

This process iterates up to 3 times or until evidence completeness criteria are satisfied. Reasoning traces are logged for explainability.

### 3.3.4 Safety Constraint Checker

The safety checker employs a hybrid validation approach:

1. **Rule-Based Constraints:** Hard filters derived from guideline contraindications (e.g., "avoid warfarin in pregnancy").
2. **Knowledge Graph Inference:** SPARQL queries over DrugBank and SIDER to detect drug–drug interactions and contraindications.
3. **LLM-Based Contextual Validation:** A fine-tuned clinical LLM (based on Llama-2-7B) evaluates subtle contradictions or population mismatches not captured by rules.

Safety scores are integrated into the final ranking function:

**Score(q, d) = α · Relevance(q, d) + β · Reasoning(q, d) + γ · Safety(d)**

where α = 0.4, β = 0.3, γ = 0.3 (tuned on validation set).

## 3.4 Evaluation Framework

### 3.4.1 Relevance Metrics

We report standard IR metrics: Recall@k (k=5, 10, 20), Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (nDCG@10).

### 3.4.2 Reasoning and Completeness Metrics

- **Multi-Hop Coverage (MHC):** Proportion of queries for which all required evidence types (guideline + trial + case/KG) are retrieved.
- **Evidence Completeness Score (ECS):** Expert-rated score (1-5) assessing whether retrieved evidence collectively supports safe decision-making.
- **Reasoning Trace Accuracy (RTA):** Proportion of reasoning steps that align with expert judgment.

### 3.4.3 Safety Metric — MedFol

We introduce **MedFol** (Medical Follow-up / Safety Metric), a composite metric evaluating:
- **Relevance (R):** Alignment with query intent
- **Factual Accuracy (F):** Correctness of biomedical facts
- **Evidence Sufficiency (E):** Inclusion of all required components
- **Safety Compliance (S):** Avoidance of contraindicated or unsafe evidence

**MedFol = (R + F + E + S) / 4**, with each component scored 0-1 by clinical experts.

### 3.4.4 Baselines

We compared against:
- **BM25:** Lexical baseline
- **BioBERT, PubMedBERT:** Dense retrievers
- **DPR:** Dense Passage Retrieval
- **TART:** Instruction-aware retriever
- **Self-BioRAG:** Retrieval-augmented generation

### 3.4.5 Implementation Details

Models were trained on 4× NVIDIA A100 GPUs. PubMedBERT was fine-tuned for 5 epochs with learning rate 2e-5, batch size 32. The reasoning module used beam search (beam size = 3). Statistical significance was assessed using paired t-tests (p < 0.05).

# 2. Related Work

## 2.1 Biomedical Information Retrieval
Biomedical information retrieval has evolved from classical lexical models to semantic neural architectures. Traditional models like BM25 [10] rely on term frequency statistics but often fail to capture the semantic nuances of medical terminology, such as synonymy (e.g., "heart attack" vs. "myocardial infarction") and polysemy [11]. The advent of transformer-based language models, including BioBERT [4], ClinicalBERT [12], and PubMedBERT [13], marked a shift towards semantic retrieval. Dense Passage Retrieval (DPR) [14] further advanced the field by encoding queries and documents into dense vector spaces, significantly outperforming lexical baselines on benchmarks like BioASQ. However, these models primarily focus on semantic similarity and often struggle to interpret the complex, constraint-rich instructions typical of clinical queries [7].

## 2.2 Instruction Tuning and Clinical Intent
Instruction tuning, as demonstrated by models like InstructGPT [15] and FLAN-T5 [16], enhances the ability of LLMs to follow
## 2.2 Instruction Tuning and Clinical Intent
Instruction tuning, as demonstrated by models like InstructGPT [15] and FLAN-T5 [16], enhances the ability of LLMs to follow
## 2.2 Instruction Tuning and Clinical Intent
Instruction tuning, as demonstrated by models like InstructGPT [15] and FLAN-T5 [16], enhances the ability of LLMs to follow
## 2.2 Instruction Tuning and Clinical Intent
Instruction tuning, as demonstrated by models like InstructGPT [15] and FLAN-T5 [16], enhances thing and Multi-Source Integration
Clinical decision-making inherently involves synthesizing evidence from heterogeneous sources—guidelines, trials, case reports, and knowledge bases [6]. Most existing retrieval systems, however, operate on single-source corpora (primarily PubMed abstracts) and lack mechanisms for cross-source reasoning. While Retrieval-Augmented Generation (RAG) [17] attempts to bridge this gap by combining retrieval with generation, it often suffers from hallucinations and inconsistencies between retrieved documents [8]. Chain-of-Thought (CoT) prompting [18] has improved reasoning in generative tasks, but its application to the retrieval process itself—specifically for iteratively refining search results—remains underexplored in the biomedical domain.

## 2.4 Safety in Clinical AI
Safety is a paramount concern in clinical AI. Recent evaluations of medical LLMs reveal significant risks, including the generation of unsafe treatment recommendations and fabricated citations [9, 19]. Despite this, standard IR evaluation metrics (e.g., Recall@k, nDCG) do not account for safety; a system can achieve high relevance scores while retrieving contraindicated or harmful evidence. This disconnect highlights the urgent need for safety-aware retrieval frameworks and evaluation metrics that explicitly penalize unsafe outputs.

# 3. Materials and Methods

## 3.1 Dataset Construction
To address the limitations of single-source datasets, we constructed a multi-source biomedical dataset integrating four distinct evidence types:
1.  **Clinical Guidelines:** Structured recommendations and safety warnings.
2.  **Randomized Controlled Trials (RCTs):** Eligibility criteria and outcome data.
3.  **Case Reports:** Narrative descriptions of real-world clinical scenarios.
4.  **Knowledge Graphs (KG):** Structured triples representing drug-disease and drug-interaction relationships.

The dataset includes a set of instruction-style clinical queries designed to reflect real-world complexity, containing demographic filters, comorbidities, and pharmacological constraints. Each query-evidence pair was annotated with safety labels (Safe, Conditionally Safe, Unsafe, Incomplete) to support safety-aware training and evaluation.

## 3.2 System Architecture
The proposed framework consists of four integrated modules designed to align clinical intent with safe, multi-source evidence:
1.  **Instruction-Aware Query Encoder:** Parses complex clinical instructions.
2.  **Multi-Source Document Encoder:** Maps heterogeneous evidence into a unified embedding space.
3.  **Chain-of-Retrieval Reasoning Module:** Iteratively refines retrieval and checks for consistency.
4.  **Safety Constraint Checker:** Filters unsafe or contraindicated evidence.

## 3.3 Instruction-Aware Query Encoder
Unlike standard dense retrievers, our Instruction-Aware Query Encoder is designed to explicitly model clinical constraints. The module segments queries into three components: *Constraint Units* (e.g., "eGFR < 30"), *Clinical Entities* (e.g., "anticoagulants"), and *Safety Cues* (e.g., "contraindicated"). A constraint-aware attention mechanism weights these components to ensure that retrieval prioritizes documents satisfying specific patient conditions rather than generic topic relevance.

## 3.4 Multi-Source Document Encoder
To handle the structural heterogeneity of biomedical data, we employ a Multi-Source Document Encoder. This module uses specialized encoding strategies for each data type—emphasizing recommendation strength for guidelines, eligibility criteria for trials, and relational structure for knowledge graphs. These diverse representations are mapped into a unified vector space, enabling the system to retrieve and compare evidence across different source types effectively.

## 3.5 Chain-of-Retrieval Reasoning Module
We introduce a Chain-of-Retrieval (CoR) mechanism to perform multi-hop reasoning. The process follows an iterative loop:
1.  **Initial Retrieval:** Retrieve candidate evidence based on the encoded query.
2.  **Consistency Check:** Verify if the retrieved evidence (e.g., a trial result) aligns with established guidelines.
3.  **Query Refinement:** If evidence is incomplete or conflicting, the system refines the query to target missing information (e.g., searching specifically for contraindications).
4.  **Synthesis:** Aggregate consistent evidence for the final output.

## 3.6 Safety Constraint Checker
The Safety Constraint Checker serves as a final validation layer. It employs a hybrid approach:
*   **Rule-Based Filtering:** Checks against explicit guideline contraindications.
*   **KG Inference:** Detects potential drug-drug or drug-disease interactions using knowledge graph paths.
*   **LLM Verification:** Uses a frozen LLM to perform contextual safety checks on the retrieved text, flagging hallucinations or subtle safety violations.

## 3.7 Evaluation Framework and MedFol Metric
To comprehensively assess system performance, we utilize standard IR metrics (Recall@k, nDCG) alongside a novel safety-centric metric, **MedFol**. MedFol is a composite score calculated as:
41567 \text{MedFol} = w_1 \cdot \text{Relevance} + w_2 \cdot \text{Sufficiency} - w_3 \cdot \text{SafetyPenalty} 41567
where *Relevance* measures semantic alignment, *Sufficiency* assesses if all necessary evidence components are present, and *SafetyPenalty* heavily penalizes the retrieval of contraindicated or harmful information. This metric ensures that safety is a primary optimization objective.
