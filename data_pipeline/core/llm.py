"""
LLM生成模块
使用OpenAI API进行文本生成和问答
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
from typing import List, Dict, Any, Optional
from config.config import config

class LLMGenerator:
    """LLM生成器"""
    
    def __init__(self):
        """初始化LLM生成器"""
        print("初始化LLM生成器:")
        print(f"  模型: {config.llm_model}")
        print(f"  温度: {config.llm_temperature}")
        print(f"  最大tokens: {config.llm_max_tokens}")
        
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url
        )
        
        # 系统提示词模板
        self.system_prompt = """You are an AI assistant specialized in photovoltaic research and solar cell technology. Your task is to answer questions based on the provided academic documents.

Instructions:
1. Provide accurate, detailed answers based on the given context
2. If the context doesn't contain enough information to answer the question, clearly state "I don't have enough information in the provided context to answer this question accurately"
3. Always cite the source document when providing information
4. Focus on technical details and research findings
5. If multiple sources provide conflicting information, mention both perspectives

Context documents: {context}

Question: {question}

Please provide a comprehensive answer based on the context above."""
    
    def rag_query(self, question: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        基于检索到的文档回答问题
        
        Args:
            question: 用户问题
            context_docs: 检索到的相关文档列表
            
        Returns:
            包含答案和元信息的字典
        """
        try:
            # 构建上下文文本
            context_text = self._build_context_text(context_docs)
            
            # 构建完整提示词
            full_prompt = self.system_prompt.format(
                context=context_text,
                question=question
            )
            
            # 调用LLM生成答案
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
            
            answer = response.choices[0].message.content
            
            return {
                'success': True,
                'answer': answer,
                'question': question,
                'context_docs_count': len(context_docs),
                'tokens_used': response.usage.total_tokens if response.usage else None
            }
            
        except Exception as e:
            print(f"LLM API失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'question': question,
                'answer': "抱歉，生成回答时发生错误。"
            }
    
    def _build_context_text(self, context_docs: List[Dict[str, Any]]) -> str:
        """
        构建上下文文本
        
        Args:
            context_docs: 文档列表
            
        Returns:
            格式化的上下文文本
        """
        if not context_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            # 处理不同的数据格式
            if isinstance(doc, dict):
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                if isinstance(metadata, str):
                    # 如果metadata是字符串，尝试解析或使用默认值
                    source = f'Document {i}'
                else:
                    source = metadata.get('source', f'Document {i}')
            else:
                # 如果doc不是字典，将其视为内容
                content = str(doc)
                source = f'Document {i}'
            
            context_parts.append(f"Document {i} (Source: {source}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def simple_chat(self, message: str) -> str:
        """
        简单聊天功能
        
        Args:
            message: 用户消息
            
        Returns:
            LLM回复
        """
        try:
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "user", "content": message}
                ],
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"LLM聊天失败: {e}")
            return f"抱歉，处理您的消息时发生错误: {e}" 