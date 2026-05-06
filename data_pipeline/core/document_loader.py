"""
文档加载器
用于加载和预处理各种格式的学术文档
"""

import os
import glob
from typing import List, Dict, Any, Optional

class DocumentLoader:
    """文档加载器"""
    
    def __init__(self, base_path: str):
        """
        初始化文档加载器
        
        Args:
            base_path: 文档基础路径
        """
        self.base_path = base_path
        self.supported_extensions = ['.md', '.txt']
    
    def load_all_documents(self) -> List[Dict[str, Any]]:
        """
        加载所有支持的文档
        
        Returns:
            文档列表，每个文档包含content和metadata
        """
        documents = []
        
        if not os.path.exists(self.base_path):
            print(f"警告: 文档路径不存在: {self.base_path}")
            return documents
        
        # 遍历所有支持的文件类型
        for ext in self.supported_extensions:
            pattern = os.path.join(self.base_path, f"**/*{ext}")
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                try:
                    doc = self.load_document(file_path)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    print(f"加载文档失败 {file_path}: {e}")
                    continue
        
        print(f"✓ 成功加载 {len(documents)} 个文档")
        return documents
    
    def load_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        加载单个文件 (用于测试)
        
        Args:
            file_path: 文件路径
            
        Returns:
            包含单个文档的列表
        """
        doc = self.load_document(file_path)
        return [doc] if doc else []
    
    def load_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        加载单个文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档字典或None
        """
        try:
            # 尝试多种编码
            encodings = ['utf-8', 'gbk', 'latin1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"无法读取文件 {file_path}: 编码不支持")
                return None
            
            # 清理内容
            content = self._clean_content(content)
            
            if not content.strip():
                print(f"跳过空文档: {file_path}")
                return None
            
            # 创建元数据
            metadata = {
                'source': file_path,
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'extension': os.path.splitext(file_path)[1],
                'relative_path': os.path.relpath(file_path, self.base_path)
            }
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"加载文档时发生错误 {file_path}: {e}")
            return None
    
    def _clean_content(self, content: str) -> str:
        """
        清理文档内容，包括公式规范化
        
        Args:
            content: 原始内容
            
        Returns:
            清理后的内容
        """
        # 移除多余的空白字符
        content = content.strip()
        
        # 标准化换行符
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # 规范化LaTeX公式（处理带空格的数字格式）
        content = self._normalize_latex_formulas(content)
        
        # 移除多余的空行（保留段落分隔）
        lines = content.split('\n')
        cleaned_lines = []
        empty_line_count = 0
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
                empty_line_count = 0
            else:
                empty_line_count += 1
                if empty_line_count <= 1:  # 最多保留一个空行
                    cleaned_lines.append('')
        
        return '\n'.join(cleaned_lines)
    
    def _normalize_latex_formulas(self, content: str) -> str:
        """
        规范化LaTeX公式，特别是处理带空格的数字格式
        
        Args:
            content: 原始内容
            
        Returns:
            规范化后的内容
        """
        import re
        
        # 处理行内公式中的数字格式：$1 9 . 2 0 \%$ -> $19.20\%$
        # 匹配 $...$ 中的内容，清理数字之间的空格
        def clean_inline_formula(match):
            formula = match.group(1)
            # 清理数字之间的空格，但保留其他LaTeX命令
            # 匹配数字、小数点、百分号之间的空格
            cleaned = re.sub(r'(\d)\s+(\d)', r'\1\2', formula)  # 数字之间
            cleaned = re.sub(r'(\d)\s+(\.)', r'\1\2', cleaned)   # 数字和小数点之间
            cleaned = re.sub(r'(\.)\s+(\d)', r'\1\2', cleaned)   # 小数点和数字之间
            cleaned = re.sub(r'(\d)\s+(\\%)', r'\1\2', cleaned)   # 数字和百分号之间
            cleaned = re.sub(r'(\d)\s+(-)\s+(\d)', r'\1\2\3', cleaned)  # 范围：19 - 20 -> 19-20
            return f'${cleaned}$'
        
        # 处理行内公式
        content = re.sub(r'\$([^$]+)\$', clean_inline_formula, content)
        
        # 处理块级公式 $$...$$（保留原样，但可以添加注释说明）
        # 块级公式通常更复杂，保持原样以便后续处理
        
        return content
    
    def get_document_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        获取文档统计信息
        
        Args:
            documents: 文档列表
            
        Returns:
            统计信息字典
        """
        if not documents:
            return {
                'total_documents': 0,
                'total_characters': 0,
                'avg_document_size': 0,
                'file_types': {}
            }
        
        total_chars = sum(len(doc['content']) for doc in documents)
        
        # 统计文件类型
        file_types = {}
        for doc in documents:
            ext = doc['metadata'].get('extension', 'unknown')
            file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            'total_documents': len(documents),
            'total_characters': total_chars,
            'avg_document_size': total_chars / len(documents),
            'file_types': file_types
        } 