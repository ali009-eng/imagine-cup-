"""
RAG System - Retrieval-Augmented Generation for Triage Classification
Integrates vector database retrieval with LLM for intelligent patient classification
"""
from typing import Dict, Any, List, Optional
import torch
import json
import logging
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field

from data_processing.mimic_processor import MIMICProcessor
from config import Config
from utils.cache import cache_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import threading at module level
import threading

# Global model cache (singleton pattern)
_model_cache = {}
_model_lock = threading.Lock()

class TriageLLM(LLM):
    """Custom LLM wrapper for fine-tuned LLaMA model with caching"""
    model_name: str = Field(default=Config.MODEL_NAME)
    tokenizer: Any = None
    model: Any = None
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_model_cached()
    
    def _load_model_cached(self):
        """Load model with caching to avoid reloading"""
        global _model_cache
        
        cache_key = f"{self.model_name}_{self.device}"
        
        # Check cache first
        if cache_key in _model_cache:
            logger.info(f"Loading model from cache: {cache_key}")
            cached = _model_cache[cache_key]
            self.tokenizer = cached['tokenizer']
            self.model = cached['model']
            return
        
        # Load model (with lock to prevent concurrent loads)
        with _model_lock:
            # Double-check cache after acquiring lock
            if cache_key in _model_cache:
                cached = _model_cache[cache_key]
                self.tokenizer = cached['tokenizer']
                self.model = cached['model']
                return
            
            # Load model
            self._load_model()
            
            # Cache the model
            if self.model and self.tokenizer:
                _model_cache[cache_key] = {
                    'tokenizer': self.tokenizer,
                    'model': self.model
                }
                logger.info(f"Cached model: {cache_key}")
    
    def _load_model(self):
        """Load the fine-tuned model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            logger.info(f"Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=Config.MODEL_CACHE_DIR,
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            model_kwargs = {
                "cache_dir": Config.MODEL_CACHE_DIR,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None
    
    @property
    def _llm_type(self) -> str:
        return "triage_llm"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """Generate response from model"""
        if not self.model or not self.tokenizer:
            return "Error: Model not loaded"
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=Config.MAX_TOKENS,
                    temperature=Config.TEMPERATURE,
                    do_sample=Config.TEMPERATURE > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove prompt from response
            if prompt in response:
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"


class RAGSystem:
    """Retrieval-Augmented Generation system for triage classification"""
    
    def __init__(self):
        self.llm = TriageLLM()
        self.processor = MIMICProcessor()
        self.vector_db = None
        self._initialize()
    
    def _initialize(self):
        """Initialize RAG components"""
        try:
            logger.info("Initializing RAG system...")
            
            # Load or create vector database
            data = self.processor.load_data()
            
            # Check if we have real MIMIC data
            has_data = data and len(data) > 0
            
            if not has_data:
                logger.warning("No MIMIC data found. Checking for synthetic data or generating new...")
                # Try to use synthetic data
                from data_processing.synthetic_data_generator import populate_vector_db_with_synthetic_data
                try:
                    populate_vector_db_with_synthetic_data(self.processor, count=100)
                    logger.info("Populated vector database with synthetic data")
                except Exception as e:
                    logger.warning(f"Could not generate synthetic data: {e}")
            
            self.vector_db = self.processor.create_vector_db()
            
            if self.vector_db and self.vector_db.count() > 0:
                logger.info(f"RAG system initialized successfully with {self.vector_db.count()} documents")
            else:
                logger.warning("Vector database is empty. RAG will use LLM-only or rule-based classification")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            logger.warning("RAG system will use fallback classification")
    
    @cache_result(ttl=300, key_prefix="triage_classification:")
    def classify_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify patient using RAG-enhanced LLM classification
        
        Priority flow:
        1. Retrieve similar cases from vector database
        2. If cases found AND LLM available → Use LLM with context (RAG)
        3. If no cases but LLM available → Use LLM without context
        4. Otherwise → Rule-based with similarity weighting if cases found
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Dictionary with triage classification results
        """
        try:
            # Step 1: Create query from patient data
            query = self._create_patient_query(patient_data)
            logger.debug(f"Patient query created: {query[:100]}...")
            
            # Step 2: Retrieve similar cases from vector database
            similar_cases = []
            vector_db_available = False
            
            try:
                if self.vector_db is not None:
                    vector_count = self.vector_db.count()
                    vector_db_available = vector_count > 0
                    
                    if vector_db_available:
                        logger.info(f"Vector database available with {vector_count} documents. Retrieving similar cases...")
                        similar_cases = self.processor.search_similar_cases(
                            query,
                            top_k=Config.RAG_TOP_K
                        )
                        logger.info(f"Retrieved {len(similar_cases)} similar cases")
                    else:
                        logger.warning("Vector database is empty. Proceeding without retrieval context.")
                else:
                    logger.warning("Vector database not initialized. Attempting to initialize...")
                    # Try to initialize if not done
                    if self.processor:
                        try:
                            self.vector_db = self.processor.create_vector_db()
                            if self.vector_db and self.vector_db.count() > 0:
                                similar_cases = self.processor.search_similar_cases(query, top_k=Config.RAG_TOP_K)
                                vector_db_available = True
                                logger.info("Vector database initialized and cases retrieved")
                        except Exception as e:
                            logger.warning(f"Could not initialize vector database: {e}")
            except Exception as e:
                logger.error(f"Error retrieving from vector database: {e}")
                # Continue without vector DB retrieval
            
            # Step 3: Classification with priority: RAG (vector + LLM) > LLM-only > Rule-based
            classification = None
            
            # Option A: RAG - Vector DB cases found AND LLM available
            if similar_cases and self.llm.model is not None:
                logger.info("Using RAG classification: LLM with retrieved context")
                try:
                    classification = self._llm_classify(patient_data, similar_cases)
                    if classification and classification.get('triage_level'):
                        classification['classification_method'] = 'rag_with_context'
                        logger.info(f"RAG classification successful: Level {classification.get('triage_level')}")
                except Exception as e:
                    logger.error(f"RAG classification failed: {e}. Falling back to LLM-only.")
                    classification = None
            
            # Option B: LLM-only - No cases but LLM available
            if classification is None and self.llm.model is not None:
                logger.info("Using LLM classification without vector context")
                try:
                    classification = self._llm_classify(patient_data, [])
                    if classification and classification.get('triage_level'):
                        classification['classification_method'] = 'llm_without_context'
                        logger.info(f"LLM-only classification successful: Level {classification.get('triage_level')}")
                except Exception as e:
                    logger.error(f"LLM classification failed: {e}. Falling back to rule-based.")
                    classification = None
            
            # Option C: Rule-based with similarity weighting if cases found, otherwise pure rule-based
            if classification is None:
                if similar_cases:
                    logger.info("Using rule-based classification enhanced with similarity weighting")
                    classification = self._rule_based_classify(patient_data, similar_cases)
                    classification['classification_method'] = 'rule_based_with_similarity'
                else:
                    logger.info("Using pure rule-based classification")
                    classification = self._rule_based_classify(patient_data, [])
                    classification['classification_method'] = 'rule_based_pure'
            
            # Ensure classification is valid
            if not classification or not classification.get('triage_level'):
                logger.error("Classification failed. Using fallback.")
                classification = self._fallback_classification(patient_data)
            
            # Add metadata about retrieval
            if classification:
                classification['vector_db_used'] = vector_db_available
                classification['similar_cases_retrieved'] = len(similar_cases)
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying patient: {e}", exc_info=True)
            return self._fallback_classification(patient_data)
    
    def _create_patient_query(self, patient_data: Dict[str, Any]) -> str:
        """Create a search query from patient data"""
        query_parts = []
        
        # Add demographics
        if patient_data.get('age'):
            query_parts.append(f"Age {patient_data['age']}")
        if patient_data.get('gender'):
            query_parts.append(f"Gender {patient_data['gender']}")
        
        # Add chief complaint
        if patient_data.get('chief_complaint'):
            query_parts.append(patient_data['chief_complaint'])
        
        # Add vital signs
        vital_parts = []
        if patient_data.get('heart_rate'):
            vital_parts.append(f"heart rate {patient_data['heart_rate']}")
        if patient_data.get('respiratory_rate'):
            vital_parts.append(f"respiratory rate {patient_data['respiratory_rate']}")
        if patient_data.get('oxygen_saturation'):
            vital_parts.append(f"oxygen {patient_data['oxygen_saturation']}")
        if patient_data.get('temperature'):
            vital_parts.append(f"temperature {patient_data['temperature']}")
        if patient_data.get('blood_pressure'):
            vital_parts.append(f"blood pressure {patient_data['blood_pressure']}")
        if patient_data.get('pain_level'):
            vital_parts.append(f"pain level {patient_data['pain_level']}")
        
        if vital_parts:
            query_parts.append(", ".join(vital_parts))
        
        return " | ".join(query_parts)
    
    def _llm_classify(
        self,
        patient_data: Dict[str, Any],
        similar_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use LLM for classification with retrieved context
        
        Args:
            patient_data: Patient information
            similar_cases: List of similar cases from vector database (empty list if none)
        
        Returns:
            Classification dictionary
        """
        # Build prompt with context (if available)
        context = ""
        if similar_cases:
            context = self._format_similar_cases(similar_cases)
        else:
            context = "No similar historical cases found. Base classification on patient data and medical knowledge."
        
        prompt = f"""You are an expert emergency department triage nurse. Classify the following patient using ESI (Emergency Severity Index) levels 1-5.

ESI Level Guide:
1 - Immediate - Life-threatening conditions requiring immediate intervention
2 - High Risk - Urgent conditions with high risk of deterioration
3 - Medium - Stable patients requiring evaluation
4 - Lower Medium - Stable patients with minor issues
5 - Minor - Non-urgent conditions

{f"Similar Historical Cases from Database:\n{context}" if similar_cases else "Historical Context:\n" + context}

Current Patient:
Age: {patient_data.get('age', 'Unknown')}
Gender: {patient_data.get('gender', 'Unknown')}
Chief Complaint: {patient_data.get('chief_complaint', 'Not specified')}
Heart Rate: {patient_data.get('heart_rate', 'N/A')} bpm
Respiratory Rate: {patient_data.get('respiratory_rate', 'N/A')} /min
Oxygen Saturation: {patient_data.get('oxygen_saturation', 'N/A')}%
Temperature: {patient_data.get('temperature', 'N/A')}°C
Blood Pressure: {patient_data.get('blood_pressure', 'N/A')}
Pain Level: {patient_data.get('pain_level', 'N/A')}/10

Provide classification in JSON format:
{{
    "triage_level": <1-5>,
    "reasoning": "<brief explanation>",
    "risk_factors": ["<risk factor 1>", "<risk factor 2>"],
    "recommendations": ["<recommendation 1>", "<recommendation 2>"]
}}
"""
        
        try:
            response = self.llm._call(prompt)
            
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                llm_result = json.loads(json_str)
                
                # Validate triage level
                triage_level = int(llm_result.get('triage_level', 3))
                if not (1 <= triage_level <= 5):
                    triage_level = self._rule_based_classify(patient_data, [])['triage_level']
                
                return {
                    'triage_level': triage_level,
                    'esi_level': Config.ESI_LEVELS.get(triage_level, 'Unknown'),
                    'reasoning': llm_result.get('reasoning', 'LLM-based classification'),
                    'risk_factors': llm_result.get('risk_factors', []),
                    'recommendations': llm_result.get('recommendations', []),
                    'confidence': 0.85,  # Higher confidence for LLM
                    'retrieved_cases': len(similar_cases),
                    'classification_method': 'llm_rag'
                }
            else:
                # Fallback if JSON parsing fails
                logger.warning("Could not parse LLM response as JSON, using rule-based fallback")
                return self._rule_based_classify(patient_data, similar_cases)
                
        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            return self._rule_based_classify(patient_data, similar_cases)
    
    def _rule_based_classify(
        self,
        patient_data: Dict[str, Any],
        similar_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Rule-based classification with similarity weighting"""
        # Extract vital signs
        hr = patient_data.get('heart_rate', 0) or 0
        rr = patient_data.get('respiratory_rate', 0) or 0
        o2 = patient_data.get('oxygen_saturation', 100) or 100
        temp = patient_data.get('temperature', 37) or 37
        pain = patient_data.get('pain_level', 0) or 0
        
        # Rule-based triage level determination
        triage_level = 5  # Default to lowest priority
        
        # Level 1 criteria (Critical)
        if (hr > 120 or rr > 30 or o2 < 90 or temp > 39 or pain > 8):
                triage_level = 1
        # Level 2 criteria (Urgent)
        elif (hr > 100 or rr > 25 or o2 < 95 or temp > 38 or pain > 6):
                triage_level = 2
        # Level 3 criteria (Medium)
        elif (hr > 90 or rr > 20 or o2 < 98 or temp > 37.5 or pain > 4):
                triage_level = 3
        # Level 4 criteria (Lower Medium)
        elif (hr > 80 or rr > 18 or o2 < 99 or temp > 37 or pain > 2):
                triage_level = 4
        
        # Adjust based on similar cases if available
        if similar_cases and len(similar_cases) > 0:
            # Calculate weighted average from similar cases
            similar_levels = []
            weights = []
            
            for case in similar_cases:
                metadata = case.get('metadata', {})
                if 'esi_level' in metadata:
                    try:
                        case_level = int(metadata['esi_level'])
                        if 1 <= case_level <= 5:
                            # Weight by similarity (higher similarity = more weight)
                            distance = case.get('distance', 1.0)
                            weight = max(0.1, 1.0 - distance)
                            similar_levels.append(case_level)
                            weights.append(weight)
                    except (ValueError, TypeError):
                        pass
            
            if similar_levels:
                # Weighted average
                weighted_sum = sum(level * weight for level, weight in zip(similar_levels, weights))
                total_weight = sum(weights)
                avg_level = weighted_sum / total_weight
                
                # Combine rule-based and similar case average
                triage_level = int(round((triage_level * 0.6) + (avg_level * 0.4)))
                triage_level = max(1, min(5, triage_level))  # Clamp to 1-5
        
        # Generate reasoning
        reasoning = self._generate_reasoning(patient_data, triage_level, similar_cases)
        
        return {
            'triage_level': triage_level,
            'esi_level': Config.ESI_LEVELS.get(triage_level, 'Unknown'),
            'reasoning': reasoning,
            'risk_factors': self._identify_risk_factors(patient_data, triage_level),
            'recommendations': self._generate_recommendations(patient_data, triage_level),
            'confidence': 0.75 if similar_cases else 0.65,
            'retrieved_cases': len(similar_cases),
            'classification_method': 'rule_based' + ('_similarity_enhanced' if similar_cases else '')
        }
    
    def _format_similar_cases(self, similar_cases: List[Dict[str, Any]]) -> str:
        """Format similar cases for prompt"""
        if not similar_cases:
            return "No similar historical cases found."
        
        formatted = []
        for i, case in enumerate(similar_cases[:5], 1):  # Top 5 cases
            metadata = case.get('metadata', {})
            doc = case.get('document', '')[:200]  # Truncate long documents
            similarity = 1 - case.get('distance', 1.0)
            
            case_info = f"Case {i} (Similarity: {similarity:.2f}): {doc}"
            if 'esi_level' in metadata:
                case_info += f" [ESI Level: {metadata['esi_level']}]"
            formatted.append(case_info)
        
        return "\n".join(formatted)
    
    def _generate_reasoning(
        self,
        patient_data: Dict[str, Any],
        triage_level: int,
        similar_cases: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable reasoning"""
        reasons = []
        
        # Vital sign reasoning
        hr = patient_data.get('heart_rate', 0)
        o2 = patient_data.get('oxygen_saturation', 100)
        
        if hr > 120:
            reasons.append(f"Elevated heart rate ({hr} bpm)")
        if o2 and o2 < 95:
            reasons.append(f"Low oxygen saturation ({o2}%)")
        
        if similar_cases:
            reasons.append(f"Based on {len(similar_cases)} similar historical cases")
        
        if not reasons:
            reasons.append(f"Based on standard triage guidelines (ESI Level {triage_level})")
        
        return "; ".join(reasons)
    
    def _identify_risk_factors(
        self,
        patient_data: Dict[str, Any],
        triage_level: int
    ) -> List[str]:
        """Identify risk factors"""
        risk_factors = []
        
        age = patient_data.get('age', 0)
        hr = patient_data.get('heart_rate', 0)
        o2 = patient_data.get('oxygen_saturation', 100)
        
        if age and age > 65:
            risk_factors.append("Advanced age")
        if hr and hr > 120:
            risk_factors.append("Tachycardia")
        if o2 and o2 < 95:
            risk_factors.append("Hypoxia")
        if triage_level <= 2:
            risk_factors.append("High acuity presentation")
        
        return risk_factors
    
    def _generate_recommendations(
        self,
        patient_data: Dict[str, Any],
        triage_level: int
    ) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []
        
        if triage_level <= 2:
            recommendations.append("Immediate evaluation required")
            recommendations.append("Continuous vital signs monitoring")
        elif triage_level == 3:
            recommendations.append("Evaluation within 1 hour")
            recommendations.append("Periodic vital signs monitoring")
        else:
            recommendations.append("Routine evaluation")
        
        o2 = patient_data.get('oxygen_saturation', 100)
        if o2 and o2 < 95:
            recommendations.append("Consider supplemental oxygen")
        
        return recommendations
    
    def _fallback_classification(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback classification when errors occur"""
        return {
            'triage_level': 3,
            'esi_level': Config.ESI_LEVELS.get(3, 'Unknown'),
            'reasoning': 'Error in classification - default to medium priority',
            'risk_factors': ['Manual review recommended'],
            'recommendations': ['Manual triage assessment required'],
            'confidence': 0.0,
            'retrieved_cases': 0,
            'classification_method': 'fallback'
        }