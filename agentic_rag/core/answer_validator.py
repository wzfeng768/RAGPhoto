"""
Answer Validator
Validates generated answers against retrieved contexts to ensure precision
"""

import re
import json
from typing import Dict, List, Any
from openai import OpenAI

from config.config import config


class AnswerValidator:
    """Validates generated answers against retrieved contexts"""
    
    def __init__(self, llm_client: OpenAI, verbose: bool = False):
        """
        Initialize the answer validator
        
        Args:
            llm_client: OpenAI client for LLM calls
            verbose: Enable verbose logging
        """
        self.llm_client = llm_client
        self.verbose = verbose
    
    def validate_and_correct(
        self,
        question: str,
        answer: str,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Main validation method
        
        Args:
            question: The question being answered
            answer: Generated answer
            contexts: Retrieved contexts
            
        Returns:
            Dictionary with validation results
        """
        # Step 1: Analyze question to identify expected elements
        expected_elements = self._analyze_question_expectations(question)
        
        # Step 2: Extract actual elements from contexts
        actual_elements = self._extract_from_contexts(contexts, expected_elements)
        
        # Step 3: Validate answer contains expected elements
        validation_result = self._validate_answer_completeness(
            answer, expected_elements, actual_elements
        )
        
        # Step 4: Attempt auto-correction if needed
        if validation_result['needs_correction']:
            if self.verbose:
                print(f"   ⚠️  Validation issues: {validation_result['issues']}")
            
            try:
                corrected = self._auto_correct(
                    question, answer, contexts, 
                    validation_result['missing_elements']
                )
                validation_result['corrected_answer'] = corrected
                
                if self.verbose:
                    print(f"   ✅ Auto-correction attempted")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Auto-correction failed: {e}")
                validation_result['corrected_answer'] = answer
        
        return validation_result
    
    def _analyze_question_expectations(self, question: str) -> Dict:
        """
        Identify what type of information is expected
        
        Examples:
        - "What are the PLQYs?" → expects: multiple numerical values
        - "How are crystals fabricated?" → expects: process name with parameters
        - "Compare A and B" → expects: two sets of data
        """
        question_lower = question.lower()
        
        expectations = {
            'type': 'unknown',
            'expects_numbers': False,
            'expects_comparison': False,
            'expects_method': False,
            'number_count': 0
        }
        
        # Detect numerical questions
        if any(word in question_lower for word in ['plqy', 'efficiency', 'pce', '%', 'value', 'how much', 'temperature', 'mobility']):
            expectations['expects_numbers'] = True
            # Check for plural or multiple entities → expect multiple numbers
            if ' and ' in question_lower or ' s ' in question_lower or ' are ' in question_lower:
                expectations['number_count'] = 2  # At least 2
        
        # Detect method questions
        if any(word in question_lower for word in ['how', 'fabricated', 'prepared', 'method', 'process', 'procedure']):
            expectations['expects_method'] = True
        
        # Detect comparison questions
        if any(word in question_lower for word in ['compare', 'difference', 'versus', 'vs', 'better']):
            expectations['expects_comparison'] = True
            expectations['number_count'] = 2  # Expect at least 2 values for comparison
        
        return expectations
    
    def _extract_from_contexts(self, contexts: List[Dict], expectations: Dict) -> Dict:
        """Extract expected elements from contexts"""
        elements = {
            'numerical_values': [],
            'method_names': [],
            'technical_terms': []
        }
        
        for ctx in contexts:
            content = ctx.get('content', '')
            
            if expectations['expects_numbers']:
                # Extract all numbers with units
                number_patterns = [
                    r'\d+\.?\d*\s*%',          # Percentages
                    r'\d+\.?\d*\s*eV',         # Energy values
                    r'\d+\.?\d*\s*°[CF]',      # Temperatures
                    r'\d+\.?\d*\s*nm',         # Wavelengths
                    r'\d+\.?\d*\s*[VvMm][Aa]', # Voltage/Current
                    r'\d+\.?\d*\s*cm[-−]?[²2]', # Areas
                ]
                for pattern in number_patterns:
                    numbers = re.findall(pattern, content, re.IGNORECASE)
                    elements['numerical_values'].extend(numbers)
            
            if expectations['expects_method']:
                # Extract method names (patterns like "X method", "by X", "using X")
                method_patterns = [
                    r'(?:by|using|via)\s+([a-z\-]+(?:\s+[a-z\-]+){0,3})\s+method',
                    r'(?:by|using|via)\s+([a-z\-]+(?:\s+[a-z\-]+){0,2})\s+(?:process|technique|approach)',
                    r'([a-z\-]+\s+vapor-diffusion)',
                    r'([a-z\-]+\s+evaporation)',
                    r'(spin\s+coating)',
                    r'(thermal\s+annealing)'
                ]
                for pattern in method_patterns:
                    methods = re.findall(pattern, content, re.IGNORECASE)
                    elements['method_names'].extend(methods)
        
        # Deduplicate
        elements['numerical_values'] = list(set(elements['numerical_values']))[:20]
        elements['method_names'] = list(set(elements['method_names']))[:10]
        
        return elements
    
    def _validate_answer_completeness(self, answer: str, expectations: Dict, actual_elements: Dict) -> Dict:
        """Check if answer contains expected elements"""
        issues = []
        missing = {}
        
        # Check numerical values
        if expectations['expects_numbers'] and actual_elements['numerical_values']:
            # Extract numbers from answer
            answer_numbers = []
            for pattern in [r'\d+\.?\d*\s*%', r'\d+\.?\d*\s*eV', r'\d+\.?\d*\s*°[CF]', 
                          r'\d+\.?\d*\s*nm', r'\d+\.?\d*\s*[VvMm][Aa]']:
                answer_numbers.extend(re.findall(pattern, answer, re.IGNORECASE))
            
            if not answer_numbers:
                issues.append("Missing numerical values")
                missing['numerical_values'] = actual_elements['numerical_values']
            elif expectations['number_count'] > 0 and len(answer_numbers) < expectations['number_count']:
                issues.append(f"Expected at least {expectations['number_count']} numbers, found {len(answer_numbers)}")
                missing['numerical_values'] = actual_elements['numerical_values']
        
        # Check method names
        if expectations['expects_method'] and actual_elements['method_names']:
            has_method = any(method.lower() in answer.lower() for method in actual_elements['method_names'])
            if not has_method:
                issues.append("Missing specific method name")
                missing['method_names'] = actual_elements['method_names']
        
        needs_correction = len(issues) > 0
        
        return {
            'valid': not needs_correction,
            'needs_correction': needs_correction,
            'issues': issues,
            'missing_elements': missing,
            'confidence': max(0.0, 1.0 - (len(issues) * 0.3))  # Reduce confidence per issue
        }
    
    def _auto_correct(self, question: str, answer: str, contexts: List[Dict], missing_elements: Dict) -> str:
        """Attempt to inject missing precise information into answer"""
        # Format contexts for correction prompt
        context_text = "\n\n".join([
            f"[Context {i+1}]\n{ctx.get('content', '')[:500]}..."
            for i, ctx in enumerate(contexts[:5])
        ])
        
        # Use LLM to regenerate answer with explicit instructions about missing elements
        correction_prompt = f"""The following answer is missing precise information. Please regenerate it to include the exact values and terms from the contexts.

Original Question: {question}

Current Answer (INCOMPLETE):
{answer}

Missing Information:
{json.dumps(missing_elements, indent=2)}

Retrieved Contexts:
{context_text}

TASK: Regenerate the answer to include ALL the missing precise information listed above. Use the EXACT values and terms from the contexts. DO NOT approximate or paraphrase numbers."""

        try:
            response = self.llm_client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a precision correction assistant. Your job is to inject missing exact values and terms into incomplete answers. Always use exact numbers and terms from the source contexts."},
                    {"role": "user", "content": correction_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️  LLM correction failed: {e}")
            return answer  # Fallback to original if correction fails


def test_validator():
    """Test the answer validator"""
    print("Testing Answer Validator...")
    
    # Mock LLM client
    from openai import OpenAI
    client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url)
    
    validator = AnswerValidator(client, verbose=True)
    
    # Test case: Missing precise numbers
    question = "What are the PLQYs of TPA-An and TBA-An?"
    answer = "The PLQYs are close to 100%"  # WRONG - too vague
    contexts = [{
        'content': 'TPA-An and TBA-An single crystals exhibit PLQYs of 98% and 99%, respectively.'
    }]
    
    result = validator.validate_and_correct(question, answer, contexts)
    
    print(f"\nValidation Result:")
    print(f"  Valid: {result['valid']}")
    print(f"  Needs correction: {result['needs_correction']}")
    print(f"  Issues: {result['issues']}")
    print(f"  Confidence: {result['confidence']}")


if __name__ == "__main__":
    test_validator()

