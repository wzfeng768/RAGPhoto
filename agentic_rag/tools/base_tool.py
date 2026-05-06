"""
Base Tool class for Agentic RAG v2
Defines the interface that all tools must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseTool(ABC):
    """
    Abstract base class for all tools
    
    All tools must implement:
    - execute(): Main tool logic
    - get_description(): Tool description for agent
    """
    
    def __init__(self, name: str = None):
        """
        Initialize tool
        
        Args:
            name: Tool name (defaults to class name)
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute tool logic
        
        Args:
            query: Input query
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with tool results
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get tool description
        
        Returns:
            Human-readable description of what the tool does
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

