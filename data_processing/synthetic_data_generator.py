"""
Synthetic Data Generator for RAG System
Creates sample medical cases when MIMIC data is not available
"""
import random
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic medical cases for RAG system"""
    
    CHIEF_COMPLAINTS = [
        "Chest pain",
        "Shortness of breath",
        "Abdominal pain",
        "Fever",
        "Headache",
        "Nausea and vomiting",
        "Back pain",
        "Cough",
        "Dizziness",
        "Fatigue",
        "Joint pain",
        "Rash",
        "Sore throat",
        "Urinary symptoms",
        "Seizure",
        "Trauma",
        "Burns",
        "Allergic reaction",
        "Poisoning",
        "Psychiatric emergency"
    ]
    
    SYMPTOM_PATTERNS = {
        "Chest pain": {
            "vitals": {"heart_rate": (90, 140), "blood_pressure": (120, 180)},
            "esi_levels": [1, 2, 3],
            "common": ["chest tightness", "radiating pain", "shortness of breath"]
        },
        "Shortness of breath": {
            "vitals": {"respiratory_rate": (20, 35), "oxygen_saturation": (85, 98)},
            "esi_levels": [1, 2, 3],
            "common": ["wheezing", "cough", "chest pain"]
        },
        "Fever": {
            "vitals": {"temperature": (37.5, 40.5), "heart_rate": (90, 120)},
            "esi_levels": [2, 3, 4],
            "common": ["chills", "body aches", "fatigue"]
        },
        "Abdominal pain": {
            "vitals": {"heart_rate": (70, 110), "blood_pressure": (100, 150)},
            "esi_levels": [2, 3, 4],
            "common": ["nausea", "vomiting", "diarrhea"]
        }
    }
    
    @staticmethod
    def generate_case(patient_id: str = None, complaint: str = None) -> Dict[str, Any]:
        """Generate a synthetic medical case"""
        if not patient_id:
            patient_id = f"SYNTH_{random.randint(1000, 9999)}"
        
        if not complaint:
            complaint = random.choice(SyntheticDataGenerator.CHIEF_COMPLAINTS)
        
        # Get pattern for complaint
        pattern = SyntheticDataGenerator.SYMPTOM_PATTERNS.get(
            complaint,
            {"vitals": {}, "esi_levels": [3, 4, 5], "common": []}
        )
        
        # Generate demographics
        age = random.randint(18, 85)
        gender = random.choice(["M", "F"])
        
        # Generate vitals based on pattern
        vitals = {}
        if "heart_rate" in pattern["vitals"]:
            vitals["heart_rate"] = random.randint(*pattern["vitals"]["heart_rate"])
        else:
            vitals["heart_rate"] = random.randint(60, 100)
        
        if "respiratory_rate" in pattern["vitals"]:
            vitals["respiratory_rate"] = random.randint(*pattern["vitals"]["respiratory_rate"])
        else:
            vitals["respiratory_rate"] = random.randint(12, 20)
        
        if "oxygen_saturation" in pattern["vitals"]:
            vitals["oxygen_saturation"] = round(random.uniform(*pattern["vitals"]["oxygen_saturation"]), 1)
        else:
            vitals["oxygen_saturation"] = round(random.uniform(95, 100), 1)
        
        if "temperature" in pattern["vitals"]:
            vitals["temperature"] = round(random.uniform(*pattern["vitals"]["temperature"]), 1)
        else:
            vitals["temperature"] = round(random.uniform(36.5, 37.5), 1)
        
        # Blood pressure
        systolic = random.randint(100, 160)
        diastolic = random.randint(60, 100)
        vitals["blood_pressure"] = f"{systolic}/{diastolic}"
        
        # Pain level
        vitals["pain_level"] = random.randint(0, 10)
        
        # Determine ESI level
        esi_level = random.choice(pattern["esi_levels"])
        
        # Adjust ESI based on vitals
        if vitals["heart_rate"] > 120 or vitals["oxygen_saturation"] < 90:
            esi_level = min(esi_level, 2)
        if vitals["pain_level"] > 7:
            esi_level = min(esi_level, 2)
        if vitals["temperature"] > 39:
            esi_level = min(esi_level, 2)
        
        # Create case text for vector DB
        symptoms = ", ".join(pattern["common"][:2]) if pattern["common"] else ""
        case_text = f"Age: {age} | Gender: {gender} | Chief Complaint: {complaint}"
        if symptoms:
            case_text += f" | Symptoms: {symptoms}"
        case_text += f" | Heart Rate: {vitals['heart_rate']} | Respiratory Rate: {vitals['respiratory_rate']}"
        case_text += f" | O2 Saturation: {vitals['oxygen_saturation']} | Temperature: {vitals['temperature']}"
        case_text += f" | Blood Pressure: {vitals['blood_pressure']} | Pain Level: {vitals['pain_level']}"
        case_text += f" | ESI Level: {esi_level}"
        
        return {
            "patient_id": patient_id,
            "subject_id": patient_id,
            "age": age,
            "gender": gender,
            "chief_complaint": complaint,
            "case_text": case_text,
            "esi": esi_level,
            "triage_level": esi_level,
            **vitals
        }
    
    @staticmethod
    def generate_cases(count: int = 100) -> List[Dict[str, Any]]:
        """Generate multiple synthetic cases"""
        cases = []
        for i in range(count):
            cases.append(SyntheticDataGenerator.generate_case())
        return cases


def populate_vector_db_with_synthetic_data(processor, count: int = 100):
    """Populate vector database with synthetic data if MIMIC data unavailable"""
    logger.info(f"Generating {count} synthetic cases for vector database...")
    
    try:
        cases = SyntheticDataGenerator.generate_cases(count)
        
        # Convert to DataFrame-like structure for processor
        import pandas as pd
        
        # Create DataFrame from cases
        df = pd.DataFrame(cases)
        
        # Store in processor's processed data
        processed_path = processor.processed_path
        csv_path = processed_path / "synthetic_cases.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(cases)} synthetic cases to {csv_path}")
        
        # Create vector database
        collection = processor.create_vector_db(force_recreate=False)
        
        # If collection is empty, populate it
        if collection.count() == 0:
            logger.info("Vector database is empty. Populating with synthetic cases...")
            collection = processor.create_vector_db(force_recreate=True)
        
        logger.info(f"Vector database now has {collection.count()} documents")
        return collection
        
    except Exception as e:
        logger.error(f"Error populating synthetic data: {e}")
        return None

