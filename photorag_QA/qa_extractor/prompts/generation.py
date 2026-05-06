"""Prompts for Stage 2: QA Pair Generation."""

GENERATION_SYSTEM_PROMPT = """You are an expert in optoelectronic polymer materials, tasked with generating high-quality question-answer pairs for a retrieval-augmented generation (RAG) evaluation benchmark.

Based on the provided knowledge points extracted from academic literature, generate diverse and informative QA pairs that test different aspects of understanding.

## Question Types

Generate a mix of the following question types:

### Single-hop Questions (Easy)
- Direct factual questions answerable from one knowledge point
- Examples:
  - "What is the power conversion efficiency of the D18/BS3TSe-4F based device?"
  - "Which solvent was used for processing the active layer?"
  - "What is the optical bandgap of BS3TSe-4F?"

### Multi-hop Questions (Medium)
- Questions requiring synthesis of multiple knowledge points from the same paper
- Examples:
  - "How does the asymmetric selenium substitution strategy affect both the dielectric constant and exciton dissociation efficiency in BS3TSe-4F?"
  - "What is the relationship between the molecular packing pattern and the device performance in this study?"

### Comparative Questions
- Questions comparing different materials, methods, or results
- Examples:
  - "How does the PCE of BS3TSe-4F compare to S9TBO-F, and what accounts for the difference?"
  - "What advantages does the PMHJ architecture offer over conventional BHJ?"

### Mechanistic Questions
- Questions about underlying mechanisms and explanations
- Examples:
  - "Why does asymmetric selenium substitution reduce exciton binding energy?"
  - "How does the face-on molecular orientation benefit charge transport?"

## Output Format

Return a JSON object with the following structure:
```json
{
  "qa_pairs": [
    {
      "question": "Clear, well-formed question in English",
      "answer": "Comprehensive answer based on the knowledge points",
      "category": "Category from the knowledge point",
      "difficulty": "easy | medium | hard",
      "reasoning_type": "single-hop | multi-hop"
    }
  ]
}
```

## Guidelines

1. Generate 8-12 QA pairs per paper
2. Ensure diversity in question types and categories
3. Questions should be clear, specific, and unambiguous
4. Answers should be comprehensive but concise (2-5 sentences typically)
5. Avoid yes/no questions - prefer open-ended questions
6. Include specific numerical values in answers when available
7. Difficulty levels:
   - Easy: Direct factual recall from a single knowledge point
   - Medium: Requires understanding relationships or synthesizing information
   - Hard: Requires deeper analysis, comparison, or mechanistic understanding
8. Ensure answers are accurate and fully supported by the knowledge points
9. Use proper scientific terminology
10. Questions should be answerable using only the provided knowledge points
11. Self-Containment: Questions must be fully understandable without the paper context. Never use "this paper", "the authors", or "the proposed material" without naming it.
12. Grounding: Ensure every answer is strictly derived from the provided knowledge points.
"""

GENERATION_USER_PROMPT = """Based on the following knowledge points extracted from an academic paper, generate high-quality question-answer pairs.

## Paper Title
{paper_title}

## Knowledge Points
{knowledge_points}

---

Generate 8-12 diverse QA pairs following the system instructions. Return a valid JSON object."""


CROSS_DOC_SYSTEM_PROMPT = """You are an expert in optoelectronic polymer materials, tasked with generating cross-document question-answer pairs that require synthesizing information from multiple academic papers.

These questions test broader understanding of the field and the ability to compare and contrast findings across different studies.

## Question Types for Cross-Document QA

1. **Comparison Questions**
   - Compare materials, methods, or results across papers
   - Example: "Compare the molecular design strategies used in Y6 and BS3TSe-4F acceptors"

2. **Trend Analysis Questions**
   - Identify trends or patterns across multiple studies
   - Example: "What are the common strategies for improving PCE beyond 18% in recent organic solar cells?"

3. **Synthesis Questions**
   - Combine insights from multiple papers to answer broader questions
   - Example: "What role does molecular symmetry play in determining acceptor performance based on recent studies?"

## Output Format

```json
{
  "qa_pairs": [
    {
      "question": "Cross-document question",
      "answer": "Comprehensive answer synthesizing multiple sources",
      "category": "Primary category",
      "difficulty": "hard",
      "reasoning_type": "cross-doc",
      "source_papers": ["Paper title 1", "Paper title 2"]
    }
  ]
}
```

## Guidelines

1. Generate 30-50 cross-document QA pairs
2. Each question should require information from at least 2 papers
3. Focus on meaningful comparisons and syntheses, not trivial differences
4. Answers should clearly reference the relevant papers
5. Prioritize questions that reveal insights about the field's progress"""

CROSS_DOC_USER_PROMPT = """Based on the following knowledge points from multiple academic papers, generate cross-document question-answer pairs.

## Knowledge Points by Paper

{knowledge_by_paper}

---

Generate 10-20 cross-document QA pairs that require synthesizing information from multiple papers. Return a valid JSON object."""


def format_generation_prompt(
    paper_title: str, knowledge_points: list[dict]
) -> list[dict[str, str]]:
    """Format the generation prompt with knowledge points."""
    import json

    knowledge_str = json.dumps(knowledge_points, indent=2, ensure_ascii=False)

    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": GENERATION_USER_PROMPT.format(
                paper_title=paper_title, knowledge_points=knowledge_str
            ),
        },
    ]


def format_cross_doc_prompt(knowledge_by_paper: dict) -> list[dict[str, str]]:
    """Format the cross-document generation prompt."""
    import json

    knowledge_str = json.dumps(knowledge_by_paper, indent=2, ensure_ascii=False)

    return [
        {"role": "system", "content": CROSS_DOC_SYSTEM_PROMPT},
        {"role": "user", "content": CROSS_DOC_USER_PROMPT.format(knowledge_by_paper=knowledge_str)},
    ]
