import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import openai
from agent_tools import TranslationTools
import textwrap
import os
from dotenv import load_dotenv
import time
import re
import pandas as pd


load_dotenv() 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' 
)

logger = logging.getLogger(__name__)

TOTAL_TOKENS_USED = 0  # Global counter

@dataclass
class EvaluationCriteria:
    """Persistent evaluation criteria stored in memory"""
    accuracy: str = "Does the Filipino translation correctly convey the English source text’s meaning, intent, and details? (0-10)"
    fluency: str = "Is the translation grammatically correct, natural, and idiomatic in Filipino? (0-10)"
    coherence: str = "Does the translation maintain logical flow, context, and structure from the source? (0-10)"
    cultural_appropriateness: str = "Does the translation respect Filipino cultural norms, idioms, and sensitivities (e.g., 'po', 'opo', regional expressions)? (0-10)"
    guideline_adherence: str = "Does the translation follow domain-specific style, terminology, or guidelines? (0-10)"
    completeness: str = "Are all elements of the English source text translated into Filipino without omissions or additions? (0-10)"


@dataclass
class EvaluationResult:
    """Structured output format"""
    overall_score: float
    individual_scores: Dict[str, float]
    detailed_explanations: Dict[str, str]
    thought_process: List[str]
    timestamp: str
    metadata: Dict[str, Any]

class MemoryModule:
    """Manages persistent and temporary information"""
    
    def __init__(self):
        self.persistent_memory = {
            "evaluation_criteria": EvaluationCriteria(),
            "evaluation_history": [],
            "learned_patterns": {}
        }
        self.temporary_memory = {}
    
    def store_persistent(self, key: str, value: Any):
        """Store information that persists across evaluations"""
        self.persistent_memory[key] = value
    
    def store_temporary(self, key: str, value: Any):
        """Store information for current evaluation only"""
        self.temporary_memory[key] = value
    
    def get_persistent(self, key: str) -> Any:
        return self.persistent_memory.get(key)
    
    def get_temporary(self, key: str) -> Any:
        return self.temporary_memory.get(key)
    
    def clear_temporary(self):
        """Clear temporary memory after evaluation"""
        self.temporary_memory.clear()

class PlanningEngine:
    """Guides the LLM's evaluation process through structured reasoning"""
    
    def __init__(self, memory=None):
        self.memory = memory
        self.evaluation_steps = [
            "accuracy_assessment", 
            "fluency_evaluation",
            "coherence_check",
            "cultural_appropriateness_check",
            "guideline_adherence_check",
            "completeness_verification",
            "score_synthesis",
        ]

    
    def get_step_prompt(self, step: str, context: Dict) -> str:
        """Generate specific prompts for each evaluation step"""
        source_text = context.get('source_text', '')
        translation = context.get('translation', '')
        tool_results = context.get('tool_result', [])
        tool_context = ""
        if tool_results:
            tool_context = "\n\nTool Analysis Results:\n"
            for tool_result in tool_results:
                for tool_name, result in tool_result.items():
                    tool_context += f"- {tool_name}: {result}\n"
            tool_context += "\nUse this tool analysis to inform your evaluation.\n"
    

        
        prompts = {
            "accuracy_assessment": f"""
            Evaluate the **accuracy** of the Filipino translation (0–10), focusing on medical precision and meaning preservation.

            Source (English): {source_text}
            Translation (Filipino): {translation}


            Consider:
            - Does the translation fully convey the meaning, intent, and details of the source text?
            - Are all medical terms and health-related concepts correctly translated into culturally accepted Filipino equivalents?
            - Is the meaning medically safe and unambiguous?
            - Are there no distortions, misinterpretations, or false implications?

            Provide your score as "Score: X/10" where X is your numeric rating.
            """,

            "fluency_evaluation": f"""
            Evaluate the **fluency** of the Filipino translation (0–10).

            Source (English): {source_text}
            Translation (Filipino): {translation}
            {tool_context}

            Consider:
            - Is the translation grammatically correct according to Filipino language rules?
            - Does it read naturally and idiomatically to native speakers?
            - Is sentence structure smooth and easy to follow?
            - Does it use natural connectors and phrasing so that healthcare workers or patients can easily understand it?

            Provide your score as "Score: X/10" where X is your numeric rating.
            """,

            "coherence_check": f"""
            Evaluate the **coherence** of the Filipino translation (0–10).

            Source (English): {source_text}
            Translation (Filipino): {translation}
            {tool_context}

            Consider:
            - Does the translation maintain the logical flow and structure of the source text?
            - Are ideas presented in the same logical sequence as in the original?
            - Are the embeddings of the source text and the translation good?
            - Are transitions between sentences and concepts smooth and contextually appropriate?
            - Is the overall message easy to follow without confusion?

            Provide your score as "Score: X/10" where X is your numeric rating.
            """,

            "cultural_appropriateness_check": f"""
            Evaluate the **cultural appropriateness** of the Filipino translation (0–10).

            Source (English): {source_text}
            Translation (Filipino): {translation}

            Consider:
            - Does the translation respect Filipino cultural norms and sensitivities?
            - Does it use respectful language when addressing patients or elders (e.g., "po", "opo") where needed?
            - Are idioms, expressions, or analogies localized appropriately?
            - Are there no culturally insensitive terms or misunderstandings?

            Provide your score as "Score: X/10" where X is your numeric rating.
            """,

            "guideline_adherence_check": f"""
            Evaluate the **guideline adherence** of the Filipino translation (0–10), focusing on the medical domain.

            Source (English): {source_text}
            Translation (Filipino): {translation}
            {tool_context}

            Consider:
            - Does the translation follow established medical terminology in Filipino?
            - Is the style consistent with professional healthcare communication standards?
            - Are abbreviations, units, and drug names handled according to medical guidelines?
            - Are there no deviations from required formatting or domain-specific phrasing?

            NOTE: If there are no medical terms, then just give a perfect score

            Provide your score as "Score: X/10" where X is your numeric rating.
            """,

            "completeness_verification": f"""
            Evaluate the **completeness** of the Filipino translation (0–10).

            Source (English): {source_text}
            Translation (Filipino): {translation}

            Consider:
            - Are all ideas, details, and important elements from the source text translated?
            - Is there no omission of critical medical information?
            - Are there no unnecessary additions that change the intended meaning?
            - Does every sentence from the source have a corresponding translation?

            Provide your score as "Score: X/10" where X is your numeric rating.
            """,

            "score_synthesis": f"""
            Based on your detailed analysis, combine the individual scores into a final overall score.

            IMPORTANT:
            - Final score must be **0–5** (not 10).
            - This is a weighted, holistic judgment, **not** a straight average, make it whole number.

            SCORING LOGIC:
            - Give **5/5 (Excellent)** if:
                - The majority of criteria are 9–10, AND
                - None of the criteria are low, AND
                - There are no critical accuracy or medical safety issues.
                - Give **4/5 (Good)** if:
                - One or two criteria are slightly lower (6–8) but others are strong, OR
                - Quality is high but small non-critical issues exist.
            - Give **3/5 (Good)** if:
                - Multiple criteria are moderate (6–7) OR one is low (5 or below) but accuracy is still acceptable.
            - Give **1–2/5 (Poor)** if:
                - Major accuracy errors, missing content, or unsafe translations.

            - Pair the number with a quality label:
                - 5 = Excellent
                - 3–4 = Good
                - 1–2 = Poor

            Source (English): {source_text}
            Translation (Filipino): {translation}

            Previous analysis results from temporary memory:
            {self._get_previous_scores_summary(context)}

            Provide the final score in the format: "Score: X/5 - [label]" (e.g., "Score: 4/5 - Good"), followed by reasoning.
            """
        }
        return prompts.get(step, f"Analyze the translation pair:\nSource: {source_text}\nTranslation: {translation}")

    def _get_previous_scores_summary(self, context: Dict) -> str:
        """Get a summary of previous step scores for synthesis"""
        summary = []

        criteria_steps = [
            'accuracy_assessment',
            'fluency_evaluation',
            'coherence_check',
            'cultural_appropriateness_check',
            'guideline_adherence_check',
            'completeness_verification'
        ]

        for step in criteria_steps:
            result_text = None

            # 1. Try memory first
            if self.memory:
                step_data = self.memory.get_temporary(step)
                if step_data and 'result' in step_data:
                    result_text = step_data['result']

            # 2. Fallback to context
            if not result_text and step in context:
                result_text = context[step]

            # 3. Extract score
            if result_text:
                score = self._extract_numeric_score(result_text)
                if score is not None:
                    criterion = (
                        step.replace('_assessment', '')
                            .replace('_evaluation', '')
                            .replace('_check', '')
                            .replace('_analysis', '')
                            .replace('_verification', '')
                    )
                    summary.append(f"{criterion}: {score}/10")

        return "; ".join(summary) if summary else "No scores extracted yet"

    def _extract_numeric_score(self, text: str) -> float:
        """Extract numeric score from text response"""
        import re
        
        # Look for patterns like "8/10", "Score: 7", "8.5 out of 10", etc.
        patterns = [
            r'Score:\s*(\d+\.?\d*)/10',
            r'Score:\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)/10',
            r'(\d+\.?\d*)\s*out of 10',
            r'rate.*?(\d+\.?\d*)',
            r'scored?\s*(\d+\.?\d*)',
            r'rating.*?(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*/\s*10'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Take the first valid score found
                    for match in matches:
                        score = float(match)
                        if 0 <= score <= 10:  # Valid score range
                            return score
                except ValueError:
                    continue
        
        return 5.0  # Default score if none found

class AgenticTranslationJudge:
    """Main agentic system orchestrating the translation evaluation"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.memory = MemoryModule()
        self.planner = PlanningEngine(self.memory)
        self.tools = TranslationTools()
        self.thought_log = []
        
        # Initialize the judge persona
        self.system_prompt = """
        You are an expert Filipino translation judge with deep knowledge of both English and Filipino languages.
        You are systematic, unbiased, and thorough in your evaluations. You consider linguistic, cultural, 
        and contextual factors when assessing translation quality.
        
        Your approach is methodical - you analyze translations step by step, use tools when helpful,
        and reflect on your reasoning before providing final judgments.
        
        When asked to score something, always provide a clear numeric whole number score in the format "Score: X/10" 
        where X is a number between 0 and 10 or 0 and 5 depending on the instruction. Follow this with your detailed reasoning.
        
        You communicate your reasoning clearly and provide constructive feedback for improvement.
        """
    
    def _format_multiline(self, text: str, width: int = 100, indent: str = '  ') -> str:
        """
        Wrap paragraphs to `width` and indent each line by `indent`.
        Preserves paragraph breaks (empty line between paragraphs).
        """
        if not text:
            return ''
        # If it's not a simple string (e.g., dict), stringify it nicely
        if not isinstance(text, str):
            try:
                text = json.dumps(text, indent=2, ensure_ascii=False)
            except Exception:
                text = str(text)

        paragraphs = [p.strip() for p in text.split('\n\n')]
        wrapped_paragraphs = []
        for p in paragraphs:
            # collapse internal multiple newlines and linebreaks into single space,
            # but preserve sentences as a paragraph
            lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
            if not lines:
                wrapped_paragraphs.append('')
                continue
            joined = ' '.join(lines)
            filled = textwrap.fill(joined, width=width)
            indented = textwrap.indent(filled, indent)
            wrapped_paragraphs.append(indented)
        return '\n\n'.join(wrapped_paragraphs)

    def pretty_print_evaluation(self,
                                result: 'EvaluationResult',
                                width: int = 100,
                                show_thoughts_preview: bool = True,
                                thought_preview_limit: int = 30,
                                save_readable: bool = True,
                                readable_filepath: str = "translation_evaluation.txt") -> None:
        """
        Nicely print the evaluation to console and optionally save a full readable text file.
        - width: line wrap width for console
        - show_thoughts_preview: if True print first `thought_preview_limit` entries of thought log
        - save_readable: save everything untruncated to `readable_filepath` for later inspection
        """

        # Console printing
        sep = "=" * 80
        print(sep)
        print("=== TRANSLATION EVALUATION RESULTS ===")
        print(f"Overall Score: {result.overall_score}/5\n")
        print("Individual Scores:")
        for crit, score in result.individual_scores.items():
            print(f"  {crit.replace('_',' ').title()}: {score}/10")
        print("\nDetailed Explanations:\n")

        for criterion, explanation in result.detailed_explanations.items():
            title = criterion.replace('_', ' ').title()
            print(f"{title}:")
            formatted = self._format_multiline(explanation, width=width, indent='    ')
            print(formatted)
            print()  # blank line between sections

        # Thought process preview
        if show_thoughts_preview:
            print("Thought Process preview (first entries):")
            for i, entry in enumerate(result.thought_process[:thought_preview_limit], start=1):
                # each entry might be long—format it a bit
                print(f" {i}. {textwrap.shorten(entry, width=200, placeholder='...')}")
            if len(result.thought_process) > thought_preview_limit:
                print(f" ... (total thought steps: {len(result.thought_process)} — full log saved to file)")

        print(sep)

        # Save full human-readable file if requested
        if save_readable:
            try:
                readable_filepath = os.path.join(BASE_DIR, readable_filepath)
                with open(readable_filepath, "w", encoding="utf-8") as f:
                    f.write("=== TRANSLATION EVALUATION RESULTS ===\n\n")
                    f.write(f"Overall Score: {result.overall_score}/5\n\n")
                    f.write("Individual Scores:\n")
                    for crit, score in result.individual_scores.items():
                        f.write(f"  {crit.replace('_',' ').title()}: {score}/10\n")
                    f.write("\nDetailed Explanations:\n\n")
                    for criterion, explanation in result.detailed_explanations.items():
                        f.write(f"{criterion.replace('_',' ').title()}:\n")
                        # write the raw (unwrapped) explanation so nothing is lost
                        if not isinstance(explanation, str):
                            try:
                                f.write(json.dumps(explanation, indent=2, ensure_ascii=False) + "\n\n")
                            except Exception:
                                f.write(str(explanation) + "\n\n")
                        else:
                            f.write(explanation + "\n\n")

                    f.write("Thought Process (full):\n")
                    for i, entry in enumerate(result.thought_process, start=1):
                        f.write(f"{i}. {entry}\n")
                    f.write("\nMetadata:\n")
                    try:
                        f.write(json.dumps(result.metadata, indent=2, ensure_ascii=False))
                    except Exception:
                        f.write(str(result.metadata))
                # optional: also log using the logger
                logger.info(f"Readable evaluation exported to {readable_filepath}")
            except Exception as e:
                logger.error(f"Failed to export readable evaluation: {e}")

    def _llm_call(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make a call to the LLM with thought logging"""
        global TOTAL_TOKENS_USED
        try:
            messages = [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3
            )

            usage = response.usage
            if usage:
                TOTAL_TOKENS_USED += usage.total_tokens
                logger.info(f"Tokens used in this call: {usage.total_tokens}")
            
            result = response.choices[0].message.content
            self.thought_log.append(f"LLM Response: {result[:200]}...")
            return result
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Error in LLM call: {str(e)}"
    
    def _should_use_tool(self, step: str, context: Dict) -> Optional[str]:
        """Decide if a tool should be used for the current step"""
        # Only suggest tools for specific steps where they would be most valuable
        tool_suggestions = {
            'coherence_check': 'semantic_similarity_estimator', 
            'fluency_evaluation': 'language_tool_checker',
            'guideline_adherence_check': 'medical_term_checker' 
        }
        
        suggested_tool = tool_suggestions.get(step)
        if not suggested_tool:
            return None
        
        return suggested_tool
    
    def _execute_step(self, step: str, context: Dict) -> Dict[str, Any]:
        """Execute a single evaluation step with tool integration"""
        self.thought_log.append(f"=== Starting Step: {step} ===")

        tool_to_use = self._should_use_tool(step, context)
        tool_results = []

        # Support multiple tools
        if tool_to_use:
            if not isinstance(tool_to_use, list):
                tool_to_use = [tool_to_use]

            for tool in tool_to_use:
                self.thought_log.append(f"Using tool: {tool}")
                result = self.tools.execute_tool(tool, context)
                tool_results.append({tool: result})
                preview = str(result)[:300] + ("..." if len(str(result)) > 300 else "")
                self.thought_log.append(f"Tool '{tool}' output preview: {preview}")
            context['tool_result'] = tool_results

        # Get step-specific prompt
        step_prompt = self.planner.get_step_prompt(step, context)
        self.thought_log.append(f"Prompt sent to LLM ({len(step_prompt)} chars): {step_prompt[:200]}...")

        # LLM call
        result = self._llm_call(step_prompt)

        self.memory.store_temporary(step, {
            'result': result,
            'tool_used': tool_to_use,
            'tool_result': tool_results
        })

        self.thought_log.append(f"Step '{step}' completed. LLM output (preview): {result[:200]}...")
        self.thought_log.append(f"=== End of Step: {step} ===\n")

        return {
            'step': step,
            'result': result,
            'tool_used': tool_to_use,
            'tool_result': tool_results
        }

    def evaluate_translation(self, source_text: str, translation: str) -> EvaluationResult:
        """Main evaluation orchestration loop"""
        logger.info("Starting translation evaluation")
        
        # Clear temporary memory
        self.memory.clear_temporary()
        self.thought_log.clear()
        
        # Initialize context
        context = {
            'source_text': source_text,
            'translation': translation,
            'evaluation_criteria': self.memory.get_persistent('evaluation_criteria')
        }
        
        # Execute evaluation steps
        step_results = []
        for step in self.planner.evaluation_steps:
            try:
                result = self._execute_step(step, context)
                step_results.append(result)
                self.thought_log.append(f"Completed step: {step}")
            except Exception as e:
                logger.error(f"Error in step {step}: {e}")
                self.thought_log.append(f"Error in step {step}: {str(e)}")
        
        # Extract scores and explanations
        scores = self._extract_scores()
        explanations = self._extract_explanations()
        overall_score = self._calculate_overall_score(scores)
        
        # Create evaluation result
        result = EvaluationResult(
            overall_score=overall_score,
            individual_scores=scores,
            detailed_explanations=explanations,
            thought_process=self.thought_log.copy(),
            timestamp=datetime.now().isoformat(),
            metadata={
                'source_length': len(source_text),
                'translation_length': len(translation),
                'steps_completed': len(step_results),
                'tools_used': [r['tool_used'] for r in step_results if r['tool_used']]
            }
        )
        
        # Store in evaluation history
        history = self.memory.get_persistent('evaluation_history')
        history.append(asdict(result))
        self.memory.store_persistent('evaluation_history', history)
        
        logger.info("Translation evaluation completed")
        return result
    
    def _extract_scores(self) -> Dict[str, float]:
        """Extract individual scores from step results"""
        scores = {}
        
        # Map steps to criteria
        step_to_criterion = {
            'accuracy_assessment': 'accuracy',
            'fluency_evaluation': 'fluency',
            'coherence_check': 'coherence',
            'cultural_appropriateness_check': 'cultural_appropriateness',
            'guideline_adherence_check': 'guideline_adherence',
            'completeness_verification': 'completeness'
        }

        
        for step, criterion in step_to_criterion.items():
            step_data = self.memory.get_temporary(step)
            if step_data and 'result' in step_data:
                result_text = step_data['result']
                score = self._extract_numeric_score(result_text)
                scores[criterion] = score
                self.thought_log.append(f"Extracted {criterion} score: {score}/10 from step {step}")
            else:
                scores[criterion] = 5.0  # Default score if not found
                self.thought_log.append(f"No score found for {criterion}, using default: 5.0/10")
        
        return scores
    
    def _extract_explanations(self) -> Dict[str, str]:
        """Extract detailed explanations for each criterion"""
        explanations = {}
        
        # Map steps to criteria
        step_to_criterion = {
            'accuracy_assessment': 'accuracy',
            'fluency_evaluation': 'fluency',
            'coherence_check': 'coherence',
            'cultural_appropriateness_check': 'cultural_appropriateness',
            'guideline_adherence_check': 'guideline_adherence',
            'completeness_verification': 'completeness'
        }

        
        for step, criterion in step_to_criterion.items():
            step_data = self.memory.get_temporary(step)
            if step_data and 'result' in step_data:
                explanations[criterion] = step_data['result']
            else:
                explanations[criterion] = f"No detailed evaluation found for {criterion}"
        
        score_synthesis = self.memory.get_temporary('score_synthesis')
        if score_synthesis and 'result' in score_synthesis:
            explanations['score_synthesis'] = score_synthesis['result']
        
        return explanations
    
    def _extract_numeric_score(self, text: str) -> float:
        """Extract numeric score from text response"""
        import re
        
        # Look for patterns like "8/10", "Score: 7", "8.5 out of 10", etc.
        patterns = [
            r'Score:\s*(\d+\.?\d*)/10',
            r'Score:\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)/10',
            r'(\d+\.?\d*)\s*out of 10',
            r'rate.*?(\d+\.?\d*)',
            r'scored?\s*(\d+\.?\d*)',
            r'rating.*?(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*/\s*10'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    # Take the first valid score found
                    for match in matches:
                        score = float(match)
                        if 0 <= score <= 10:  # Valid score range
                            return score
                except ValueError:
                    continue
        
        # If no explicit score found, look for qualitative indicators
        text_lower = text.lower()
        if any(word in text_lower for word in ['excellent', 'perfect', 'outstanding']):
            return 9.0
        elif any(word in text_lower for word in ['very good', 'high quality', 'strong']):
            return 8.0
        elif any(word in text_lower for word in ['good', 'adequate', 'satisfactory']):
            return 7.0
        elif any(word in text_lower for word in ['fair', 'acceptable', 'decent']):
            return 6.0
        elif any(word in text_lower for word in ['poor', 'weak', 'problematic']):
            return 4.0
        elif any(word in text_lower for word in ['very poor', 'bad', 'incorrect']):
            return 3.0
        
        return 5.0  # Default score if none found
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Use AI's overall score from score_synthesis step (/5), fallback to weighted avg converted to /5."""
        synthesis_data = self.memory.get_temporary('score_synthesis')
        if synthesis_data and 'result' in synthesis_data:
            ai_score = self._extract_numeric_score(synthesis_data['result'])
            if 0 <= ai_score <= 5:  # Already in /5 format
                return round(ai_score, 2)
        
        weights = {crit: 1/6 for crit in scores.keys()}  # equal weight
        total_score = sum(score * weights[crit] for crit, score in scores.items())
        return round((total_score / 10) * 5, 2)  # convert /10 avg to /5


    
    def export_evaluation(self, result: EvaluationResult, filepath: str):
        """Export evaluation result to JSON file"""
        try:
            filepath = os.path.join(BASE_DIR, filepath)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(result), f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export evaluation: {e}")

def one_trial():
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    judge = AgenticTranslationJudge(OPENAI_API_KEY)

    # Example translation pair
    source_text = """
    The patient was given penicillin to prevent infection.
    """
    
    translation = """
    Ang pasyente ay binigyan ng penicillin upang maiwasan ang impeksyon.
    """
    
    # Evaluate the translation
    start_time = time.time()
    result = judge.evaluate_translation(source_text, translation)
    latency = time.time() - start_time
    
    # Print results
    judge.pretty_print_evaluation(result,
                              width=100,
                              show_thoughts_preview=True,
                              thought_preview_limit=30,
                              save_readable=True,
                              readable_filepath="translation_evaluation.txt")

    judge.export_evaluation(result, "translation_evaluation.json")

    print(f"Latency: {latency:.2f} seconds")
    print(f"Tokens used: {TOTAL_TOKENS_USED}")

def main():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    judge = AgenticTranslationJudge(OPENAI_API_KEY)

    # Load CSV
    csv_path = "test2.csv"
    df = pd.read_csv(csv_path)

    # Output folder
    output_dir = "agentic_results"
    os.makedirs(output_dir, exist_ok=True)

    for idx, row in df.iterrows():
        source_text = row["English"]
        translation = row["Filipino"]

        start_time = time.time()
        result = judge.evaluate_translation(source_text, translation)
        latency = time.time() - start_time


        # File paths for this row
        readable_path = os.path.join(output_dir, f"row_{idx}_evaluation.txt")
        json_path = os.path.join(output_dir, f"row_{idx}_evaluation.json")

        # Save readable text output
        judge.pretty_print_evaluation(
            result,
            width=100,
            show_thoughts_preview=True,
            thought_preview_limit=30,
            save_readable=True,
            readable_filepath=readable_path
        )

        # Save JSON output
        judge.export_evaluation(result, json_path)
        print(f"Processed row {idx} -> {json_path} (Latency: {latency:.2f}s)")
        print(f"Tokens used so far: {TOTAL_TOKENS_USED}")

if __name__ == "__main__":

    # for csv
    main()

    # uncomment for 1 translation pair
    # one_trial()