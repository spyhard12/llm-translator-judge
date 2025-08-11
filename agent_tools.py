import requests
from typing import Dict, List, Any, Optional
import logging
from collections import Counter
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

class TranslationTools:
    """Custom tools to enhance the agentic translation judge's capabilities"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        self.tools = {
            'language_tool_checker': self._languagetool_checker,
            'medical_term_checker': self._medical_term_checker,
            'semantic_similarity_estimator': self._embedding_based_similarity, 
        }
    
    def get_available_tools(self) -> List[str]:
        """Return list of available tools"""
        return list(self.tools.keys())
    
    def execute_tool(self, tool_name: str, context: Dict) -> Dict[str, Any]:
        """Execute a specific tool with the given context"""
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not available"}
        
        try:
            return self.tools[tool_name](context)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
    

    def _embedding_based_similarity(self, context: Dict) -> Dict[str, Any]:
        """Calculate semantic similarity using multilingual embeddings"""
        source_text = context.get('source_text', '')
        translation = context.get('translation', '')
        
        if not source_text or not translation:
            return {"error": "Missing source or translation text"}

        # Encode both texts into embeddings
        emb1 = self.embedding_model.encode(source_text, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(translation, convert_to_tensor=True)

        # Compute cosine similarity
        similarity = util.pytorch_cos_sim(emb1, emb2).item()  # -1 to 1

        # Normalize to 0–10 scale
        normalized_score = round((similarity + 1) / 2 * 10, 2)
        
        print(self._interpret_similarity_score(normalized_score))

        return {
            "semantic_similarity_score": normalized_score,
            "raw_cosine_similarity": round(similarity, 3),
            "interpretation": self._interpret_similarity_score(normalized_score),
            "analysis": f"Semantic similarity between source and translation is {normalized_score}/10"
        }

    def _interpret_similarity_score(self, score: float) -> str:
        """Provide interpretation of similarity score"""
        if score >= 8.5:
            return "Very high semantic similarity - translation preserves meaning very well"
        elif score >= 7.0:
            return "Good semantic similarity - translation captures main meaning"
        elif score >= 5.5:
            return "Moderate semantic similarity - some meaning preserved but may have gaps"
        elif score >= 4.0:
            return "Low semantic similarity - significant meaning differences detected"
        else:
            return "Very low semantic similarity - major meaning loss or errors"

    def _medical_term_checker(self, context: Dict) -> Dict[str, Any]:
        """Check correctness of medical term translations."""
        source = context.get('source_text', '').lower()
        translation = context.get('translation', '').lower()

        # Comprehensive English–Tagalog medical glossary
        medical_glossary = {
            # General Health
            "health": ["kalusugan"], "exercise": ["ehersisyo"], "nutrition": ["nutrisyon"],
            "wellness": ["kagalingan"], "balanced diet": ["balanseng diyeta"],
            "mental health": ["kalusugang pangkaisipan"], "death": ["kamatayan"],
            "sluggish": ["matamlay"], "public health": ["pampublikong kalusugan"],

            # Diseases
            "diseases": ["mga sakit"], "illnesses": ["mga sakit"], "diabetes": ["diyabetes"],
            "hypertension": ["altapresyon"], "asthma": ["hika"], "cancer": ["kanser"],
            "heart disease": ["sakit sa puso"], "stroke": ["istroke"], "tuberculosis": ["tuberkulosis"],
            "pneumonia": ["pulmonya"], "sore eyes": ["pamamaga ng mga mata"],
            "poor eyesight": ["malabong paningin"], "itchy throat": ["makating lalamunan"],
            "hoarse voice": ["namamalat na boses"], "peeling skin": ["nanunuklap na balat"],
            "infected": ["nahawa"], "skin stings": ["mahapdi na balat"], "itchy skin": ["makati ang balat"],
            "heartburn": ["pangangasim ng sikmura"], "stomach cramps": ["paghilab ng tiyan"],

            # Symptoms
            "symptoms": ["mga sintomas"], "pain": ["sakit"], "fever": ["lagnat"], "cough": ["ubo"],
            "shortness of breath": ["hirap sa paghinga"], "nausea": ["pagduduwal"],
            "dizziness": ["pagkahilo"], "swelling": ["pamamaga"], "rash": ["pantal"],

            # Body parts
            "heart": ["puso"], "stomach": ["tiyan"], "brain": ["utak"], "kidney": ["bato"],
            "liver": ["atay"], "lung": ["baga"], "bone": ["buto"], "skin": ["balat"], "blood": ["dugo"],

            # Treatments & healthcare
            "treatment": ["paggamot"], "surgery": ["operasyon"], "prescription": ["reseta"],
            "hospital": ["ospital"], "clinic": ["klinika"], "doctor": ["doktor"], "nurse": ["nars"],
            "patient": ["pasyente"], "pharmacy": ["botika"], "blood thinner": ["pampalabnaw ng dugo"],
            "joint": ["kasukasuan"], "crippled": ["pilay"], "folk illness": ["pasma"],
            "vaccination": ["pagbabakuna"], "emergency": ["hindi inaasahang pangyayari"],
            "tests": ["mga pagsusuri"], "health protocol": ["protokol sa pangkalusugan"],

            # Medications
            "medicine": ["gamot"], "tablet": ["tableta"], "capsule": ["kapsula"], "vaccine": ["bakuna"],
            "antibiotics": ["antibayotiko"], "pain reliever": ["pampawala ng sakit"], "vitamin": ["bitamina"],
            "ointment": ["pamahid"], "cough syrup": ["gamot sa ubo"], "curettage": ["raspa"],
            "crutch": ["saklay"], "laboratory": ["laboratoryo"]
        }

        matches, issues = [], []

        for eng_term, fil_terms in medical_glossary.items():
            if eng_term in source:
                if any(term in translation for term in fil_terms):
                    matches.append((eng_term, fil_terms))
                else:
                    issues.append(f"'{eng_term}' not translated using preferred Filipino terms {fil_terms}")

        score = len(matches) / (len(matches) + len(issues) + 1)

        return {
            "matched_terms": matches,
            "issues": issues,
            "medical_term_accuracy_score": round(score, 2),
            "analysis": "Good medical terminology use" if score > 0.8 else "Needs improvement in medical terminology"
        }
    
    
    def _languagetool_checker(self, context: Dict) -> Dict[str, Any]:
        """Use LanguageTool API to check Filipino grammar/style issues."""
        translation = context.get('translation', '')
        if not translation:
            return {"error": "No translation provided"}

        url = "https://api.languagetoolplus.com/v2/check"
        payload = {
            "text": translation,
            "language": "tl",  # Tagalog/Filipino
            "enabledOnly": "false",
        }
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }

        try:
            response = requests.post(url, data=payload, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            matches = data.get("matches", [])
            issues = []
            for m in matches:
                issues.append({
                    "message": m.get("message"),
                    "shortMessage": m.get("shortMessage"),
                    "suggestions": [r.get("value") for r in m.get("replacements", [])],
                    "context": m.get("context", {}).get("text", ""),
                    "ruleId": m.get("rule", {}).get("id"),
                    "category": m.get("rule", {}).get("category", {}).get("name")
                })
            print(issues)
            return {
                "issue_count": len(issues),
                "issues": issues,
                "analysis": "No major grammar issues found" if not issues else f"Found {len(issues)} grammar/style issues"
            }

        except requests.RequestException as e:
            return {"error": f"LanguageTool API request failed: {str(e)}"}
