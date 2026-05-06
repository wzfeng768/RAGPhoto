"""
问题加载器
从CSV文件加载评估问题
"""

import pandas as pd
import re
from typing import List, Dict, Any
import os

class QuestionLoader:
    """问题加载器"""
    
    def __init__(self, csv_path: str):
        """
        初始化问题加载器
        
        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
    
    def _parse_paper_reference(self, paper_ref: str) -> Dict[str, str]:
        """
        解析Paper Reference字段，提取摘录内容
        
        Args:
            paper_ref: Paper Reference字符串
            
        Returns:
            解析后的字典，包含document_title, location, excerpt
        """
        if not paper_ref or pd.isna(paper_ref):
            return {'document_title': '', 'location': '', 'excerpt': ''}
        
        paper_ref = str(paper_ref)
        
        # 使用正则表达式解析Paper Reference
        # 格式: "Title: Location, Excerpt: "content""
        
        parsed = {'document_title': '', 'location': '', 'excerpt': ''}
        
        # 提取摘录内容 (在双引号之间)
        excerpt_pattern = r'Excerpt:\s*["""]([^"""]+)["""]'
        excerpt_match = re.search(excerpt_pattern, paper_ref, re.DOTALL)
        if excerpt_match:
            parsed['excerpt'] = excerpt_match.group(1).strip()
        
        # 提取文档标题和位置信息
        # 分割 "title: location, Excerpt: ..."
        parts = paper_ref.split(', Excerpt:')
        if len(parts) > 0:
            title_location = parts[0]
            
            # 进一步分割标题和位置
            if ': ' in title_location:
                title_part, location_part = title_location.split(': ', 1)
                parsed['document_title'] = title_part.strip()
                parsed['location'] = location_part.strip()
            else:
                parsed['document_title'] = title_location.strip()
        
        return parsed
    
    def load_questions(self) -> pd.DataFrame:
        """
        加载问题数据集
        
        Returns:
            包含问题的DataFrame
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"问题文件不存在: {self.csv_path}")
        
        # 尝试多种编码读取CSV
        encodings = ['utf-8', 'gbk', 'latin1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(self.csv_path, encoding=encoding)
                print(f"✓ 使用 {encoding} 编码读取问题文件")
                break
            except (UnicodeDecodeError, pd.errors.EmptyDataError):
                continue
        
        if df is None:
            raise ValueError(f"无法读取问题文件 {self.csv_path}")
        
        print(f"✓ 加载了 {len(df)} 个问题")
        
        # 标准化列名 - 支持多种CSV格式
        column_mapping = {
            # 问题列的各种命名方式
            'Question': 'question',
            'Validation Question': 'question',
            'Research_Question': 'question',  # AI_Research_Questions.csv格式
            'question': 'question',
            # 答案列的各种命名方式
            'Expected Answer': 'expected_answer',
            'Expected_Answer': 'expected_answer',  # AI_Research_Questions.csv格式
            'expected_answer': 'expected_answer',
            'answer': 'expected_answer',
            'Answer': 'expected_answer',
            # 参考文献列
            'Paper Reference (Title + Section + Excerpt)': 'paper_reference',
            'paper_reference': 'paper_reference',
            # ID列
            'Question ID': 'question_id',
            'question_id': 'question_id',
            'ID': 'question_id',  # AI_Research_Questions.csv格式
            # 其他可能的列
            'PDF_Filename': 'pdf_filename',
            'Topic': 'topic'
        }
        
        # 重命名列
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # 确保必需的列存在，但允许部分列为空
        if 'question' not in df.columns:
            raise ValueError("CSV文件缺少必需的列: question")
        
        # 添加缺失的列
        if 'expected_answer' not in df.columns:
            df['expected_answer'] = 'N/A'
        if 'paper_reference' not in df.columns:
            df['paper_reference'] = 'N/A'
        
        # 只删除问题列为空的行
        df = df.dropna(subset=['question'])
        df = df.reset_index(drop=True)
        
        print(f"✓ 清理后的数据集包含 {len(df)} 个有效问题")
        
        return df
    
    def get_sample_questions(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        获取示例问题
        
        Args:
            n: 返回的问题数量
            
        Returns:
            问题列表
        """
        df = self.load_questions()
        sample_df = df.head(n) if len(df) >= n else df
        
        questions = []
        for _, row in sample_df.iterrows():
            # 解析Paper Reference
            paper_ref_parsed = self._parse_paper_reference(row.get('paper_reference', ''))
            
            expected_answer = row.get('expected_answer', 'N/A')
            excerpt = paper_ref_parsed.get('excerpt', '')
            
            # 🔥 关键修正：ground_truth应该是原始摘录内容，而不是期待答案
            # 这样Context Recall才能正确计算检索上下文是否包含原文信息
            ground_truth = excerpt if excerpt else expected_answer
            
            questions.append({
                'question': row['question'],
                'expected_answer': expected_answer,  # LLM期待输出的内容
                'paper_reference': row.get('paper_reference', 'N/A'),
                'paper_reference_parsed': paper_ref_parsed,
                'question_id': row.get('question_id', f"Q{len(questions) + 1}"),
                'ground_truth': ground_truth,  # 原始摘录内容，用于Context Recall计算
                'excerpt': excerpt  # 单独保存摘录内容
            })
        
        return questions
    
    def load_all_questions(self) -> List[Dict[str, Any]]:
        """
        加载所有问题
        
        Returns:
            所有问题列表
        """
        df = self.load_questions()
        
        questions = []
        for _, row in df.iterrows():
            # 解析Paper Reference
            paper_ref_parsed = self._parse_paper_reference(row.get('paper_reference', ''))
            
            expected_answer = row.get('expected_answer', 'N/A')
            excerpt = paper_ref_parsed.get('excerpt', '')
            
            # 🔥 关键修正：ground_truth应该是原始摘录内容，而不是期待答案
            # 这样Context Recall才能正确计算检索上下文是否包含原文信息
            ground_truth = excerpt if excerpt else expected_answer
            
            questions.append({
                'question': row['question'],
                'expected_answer': expected_answer,  # LLM期待输出的内容
                'paper_reference': row.get('paper_reference', 'N/A'),
                'paper_reference_parsed': paper_ref_parsed,
                'question_id': row.get('question_id', f"Q{len(questions) + 1}"),
                'ground_truth': ground_truth,  # 原始摘录内容，用于Context Recall计算
                'excerpt': excerpt  # 单独保存摘录内容
            })
        
        return questions 