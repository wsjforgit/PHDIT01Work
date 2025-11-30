"""
PhD Thesis - Medical Information Retrieval with Instructions
Data Processing Pipeline

This module handles multi-source medical data collection, processing, and instruction annotation.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datasets import load_dataset
import requests
from xml.etree import ElementTree as ET


class MedicalDataProcessor:
    """
    Unified processor for multiple medical IR datasets:
    - BioASQ
    - TREC-COVID
    - TREC-CDS
    - PubMedQA
    - MedMCQA
    """
    
    def __init__(self, output_dir: str = "./data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Instruction templates following INSTRUCTOR and FollowIR formats
        self.instruction_templates = {
            "bioasq_retrieval": "Represent the biomedical question for retrieving relevant PubMed articles that answer the question",
            "trec_diagnosis": "Represent the clinical case narrative for retrieving diagnostic articles. Focus on differential diagnosis based on patient symptoms and test results",
            "trec_treatment": "Represent the clinical case for retrieving treatment recommendation articles specific to the patient's condition",
            "trec_test": "Represent the clinical case for retrieving articles about appropriate diagnostic tests or interventions",
            "pubmedqa": "Represent the medical research question for retrieving evidence-based answers from scientific literature",
            "general_medical": "Represent the medical document for retrieval in the medicine domain"
        }
    
    def process_bioasq(self, bioasq_path: str) -> List[Dict]:
        """
        Process BioASQ dataset with instruction annotations.
        
        Format:
        {
            "query_id": "bioasq_001",
            "query_text": "What is the role of FOXP3 in allergic asthma?",
            "instruction": "Represent the biomedical question...",
            "domain": "medicine",
            "task_type": "retrieval",
            "relevant_docs": ["PMC12345", "PMC67890"],
            "constraints": ["peer-reviewed articles", "human studies"]
        }
        """
        print("Processing BioASQ dataset...")
        
        try:
            # Load BioASQ from HuggingFace datasets
            dataset = load_dataset("bioasq/bioasq_task_b", split="train")
            
            processed_data = []
            for idx, item in enumerate(dataset):
                query_data = {
                    "query_id": f"bioasq_{idx:04d}",
                    "query_text": item.get("body", ""),
                    "instruction": self.instruction_templates["bioasq_retrieval"],
                    "domain": "medicine",
                    "task_type": "retrieval",
                    "relevant_docs": item.get("documents", []),
                    "question_type": item.get("type", "unknown"),
                    "constraints": ["biomedical literature", "peer-reviewed"],
                    "ideal_answer": item.get("ideal_answer", "")
                }
                processed_data.append(query_data)
            
            # Save to JSONL
            output_file = self.output_dir / "bioasq_processed.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✓ Processed {len(processed_data)} BioASQ queries → {output_file}")
            return processed_data
            
        except Exception as e:
            print(f"✗ Error processing BioASQ: {e}")
            return []
    
    def process_trec_covid(self, topics_path: str) -> List[Dict]:
        """
        Process TREC-COVID topics with pandemic-specific instructions.
        
        Topics are provided as XML files from TREC-COVID shared task.
        """
        print("Processing TREC-COVID topics...")
        
        try:
            tree = ET.parse(topics_path)
            root = tree.getroot()
            
            processed_data = []
            for topic in root.findall('topic'):
                topic_num = topic.get('number')
                query_text = topic.find('query').text
                question_text = topic.find('question').text
                narrative = topic.find('narrative').text
                
                query_data = {
                    "query_id": f"trec_covid_{topic_num}",
                    "query_text": query_text,
                    "question": question_text,
                    "narrative": narrative,
                    "instruction": f"Retrieve biomedical research articles about COVID-19 that address: {question_text}. {narrative}",
                    "domain": "medicine",
                    "subdomain": "infectious disease",
                    "task_type": "pandemic_retrieval",
                    "corpus": "CORD-19",
                    "constraints": ["COVID-19 related", "scientific evidence", "published research"]
                }
                processed_data.append(query_data)
            
            output_file = self.output_dir / "trec_covid_processed.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✓ Processed {len(processed_data)} TREC-COVID topics → {output_file}")
            return processed_data
            
        except Exception as e:
            print(f"✗ Error processing TREC-COVID: {e}")
            return []
    
    def process_trec_cds(self, topics_path: str) -> List[Dict]:
        """
        Process TREC Clinical Decision Support topics.
        
        Each topic contains:
        - Case narrative (patient description)
        - Question type: diagnosis, test, or treatment
        """
        print("Processing TREC-CDS topics...")
        
        try:
            tree = ET.parse(topics_path)
            root = tree.getroot()
            
            processed_data = []
            for topic in root.findall('topic'):
                topic_num = topic.get('number')
                topic_type = topic.find('type').text  # diagnosis, test, or treatment
                summary = topic.find('summary').text
                description = topic.find('description').text
                
                # Select appropriate instruction based on task type
                if topic_type == "diagnosis":
                    instruction_template = self.instruction_templates["trec_diagnosis"]
                elif topic_type == "treatment":
                    instruction_template = self.instruction_templates["trec_treatment"]
                else:  # test
                    instruction_template = self.instruction_templates["trec_test"]
                
                query_data = {
                    "query_id": f"trec_cds_{topic_num}",
                    "case_narrative": summary,
                    "detailed_description": description,
                    "instruction": instruction_template,
                    "task_type": topic_type,
                    "domain": "medicine",
                    "subdomain": "clinical_decision_support",
                    "corpus": "PubMed Central",
                    "constraints": ["clinical relevance", "evidence-based"],
                    "patient_context": self._extract_patient_context(summary)
                }
                processed_data.append(query_data)
            
            output_file = self.output_dir / "trec_cds_processed.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✓ Processed {len(processed_data)} TREC-CDS cases → {output_file}")
            return processed_data
            
        except Exception as e:
            print(f"✗ Error processing TREC-CDS: {e}")
            return []
    
    def process_pubmedqa(self) -> List[Dict]:
        """Process PubMedQA dataset with instruction annotations."""
        print("Processing PubMedQA dataset...")
        
        try:
            dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
            
            processed_data = []
            for idx, item in enumerate(dataset):
                query_data = {
                    "query_id": f"pubmedqa_{idx:04d}",
                    "query_text": item["question"],
                    "context": " ".join(item.get("context", {}).get("contexts", [])),
                    "instruction": self.instruction_templates["pubmedqa"],
                    "domain": "medicine",
                    "task_type": "qa_retrieval",
                    "answer": item.get("final_decision", ""),
                    "long_answer": item.get("long_answer", ""),
                    "constraints": ["evidence-based", "biomedical research"]
                }
                processed_data.append(query_data)
            
            output_file = self.output_dir / "pubmedqa_processed.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✓ Processed {len(processed_data)} PubMedQA queries → {output_file}")
            return processed_data
            
        except Exception as e:
            print(f"✗ Error processing PubMedQA: {e}")
            return []
    
    def _extract_patient_context(self, narrative: str) -> Dict:
        """Extract patient demographics and clinical context from narrative."""
        # Simplified extraction - in real implementation, use NER/medical NLP
        context = {
            "age": None,
            "gender": None,
            "symptoms": [],
            "conditions": [],
            "medications": []
        }
        
        # Age extraction
        import re
        age_match = re.search(r'(\d+)[-\s]?year[-\s]?old', narrative, re.IGNORECASE)
        if age_match:
            context["age"] = int(age_match.group(1))
        
        # Gender extraction
        if re.search(r'\b(male|man|gentleman)\b', narrative, re.IGNORECASE):
            context["gender"] = "male"
        elif re.search(r'\b(female|woman|lady)\b', narrative, re.IGNORECASE):
            context["gender"] = "female"
        
        return context
    
    def generate_dataset_statistics(self):
        """Generate comprehensive statistics for all processed datasets."""
        stats = {
            "datasets": [],
            "total_queries": 0,
            "task_distribution": {},
            "domain_distribution": {}
        }
        
        for file_path in self.output_dir.glob("*.jsonl"):
            dataset_name = file_path.stem.replace("_processed", "")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            dataset_stats = {
                "name": dataset_name,
                "num_queries": len(data),
                "avg_query_length": sum(len(d.get("query_text", "").split()) for d in data) / len(data) if data else 0,
                "task_types": list(set(d.get("task_type", "unknown") for d in data)),
                "domains": list(set(d.get("domain", "unknown") for d in data))
            }
            
            stats["datasets"].append(dataset_stats)
            stats["total_queries"] += len(data)
        
        # Save statistics
        stats_file = self.output_dir / "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Total Queries: {stats['total_queries']}")
        for ds in stats["datasets"]:
            print(f"\n{ds['name'].upper()}:")
            print(f"  • Queries: {ds['num_queries']}")
            print(f"  • Avg Query Length: {ds['avg_query_length']:.1f} words")
            print(f"  • Task Types: {', '.join(ds['task_types'])}")
        print(f"{'='*60}\n")
        
        return stats


def main():
    """Main data processing pipeline."""
    processor = MedicalDataProcessor(output_dir="./data/processed")
    
    print("\n" + "="*60)
    print("PhD THESIS - MEDICAL IR DATA PROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Process all datasets
    # Note: You need to download the raw data first from respective sources
    
    # Example paths (adjust according to your data location)
    # processor.process_bioasq("./data/raw/bioasq/")
    # processor.process_trec_covid("./data/raw/trec_covid/topics-rnd5.xml")
    # processor.process_trec_cds("./data/raw/trec_cds/topics2016.xml")
    processor.process_pubmedqa()
    
    # Generate comprehensive statistics
    processor.generate_dataset_statistics()
    
    print("\n✓ Data processing pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
