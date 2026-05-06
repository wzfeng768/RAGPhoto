"""
Example queries for testing and demonstration
"""

# Simple queries (1 iteration expected)
SIMPLE_QUERIES = [
    "What is perovskite?",
    "Define PCE",
    "Explain band gap",
    "What is photovoltaic effect?",
]

# Medium complexity queries (2 iterations expected)
MEDIUM_QUERIES = [
    "What materials can be fabricated by spin coating?",
    "How does perovskite solar cell work?",
    "What factors affect PCE?",
    "Which materials have high electron mobility?",
]

# Complex queries (2-3 iterations expected)
COMPLEX_QUERIES = [
    "What are the latest improvements in perovskite solar cell efficiency?",
    "How to improve PCE while maintaining long-term stability?",
    "What factors affect the stability of perovskite materials?",
    "Compare organic and inorganic photovoltaic materials",
]

# All example queries
ALL_QUERIES = SIMPLE_QUERIES + MEDIUM_QUERIES + COMPLEX_QUERIES

# Query categories
QUERY_CATEGORIES = {
    'definition': SIMPLE_QUERIES,
    'medium': MEDIUM_QUERIES,
    'complex': COMPLEX_QUERIES
}


def get_example_query(difficulty: str = 'medium', index: int = 0) -> str:
    """
    Get an example query
    
    Args:
        difficulty: 'simple', 'medium', or 'complex'
        index: Index within the category
        
    Returns:
        Query string
    """
    if difficulty == 'simple':
        queries = SIMPLE_QUERIES
    elif difficulty == 'medium':
        queries = MEDIUM_QUERIES
    elif difficulty == 'complex':
        queries = COMPLEX_QUERIES
    else:
        queries = ALL_QUERIES
    
    if 0 <= index < len(queries):
        return queries[index]
    return queries[0]


def get_all_queries():
    """Get all example queries"""
    return ALL_QUERIES

