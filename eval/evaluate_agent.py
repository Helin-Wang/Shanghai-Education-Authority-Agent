#!/usr/bin/env python3
"""
Comprehensive evaluation script for Shanghai Education Authority Agent
Evaluates agent performance based on:
1. Final answer quality using language metrics
2. Retrieved documents relevance
3. Multi-round conversation analysis
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re
from collections import Counter
import ast
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
from pathlib import Path
import jieba
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
import os
from tqdm import tqdm


class AgentEvaluator:
    """Comprehensive evaluation class for the education authority agent"""
    
    def __init__(self, ground_truth_csv: str, model_path: str = None):
        """
        Initialize evaluator with ground truth data
        
        Args:
            ground_truth_csv: Path to CSV file with ground truth data
            model_path: Path to local m3e-base model (default: app/models/m3e-base)
        """
        self.ground_truth_df = pd.read_csv(ground_truth_csv)
        self.ground_truth_dict = self._create_ground_truth_dict()
        
        # Initialize embedding model from local path
        if model_path is None:
            model_path = "../app/models/m3e-base"
        
        print(f"Loading m3e-base embedding model from {model_path}...")
        try:
            self.embedding_model = SentenceTransformer(model_path)
            print("✓ m3e-base model loaded successfully")
        except Exception as e:
            print(f"Error loading local model: {e}")
            print("Falling back to online model...")
            self.embedding_model = SentenceTransformer('moka-ai/m3e-base')
        
        # Initialize LLM client with SiliconFlow API
        self._initialize_llm()
    
    @staticmethod
    def _normalize_text(text: Any) -> str:
        """Convert arbitrary input to a safe string for downstream processing."""
        if text is None:
            return ''
        # Handle pandas/numpy NaN representations explicitly
        try:
            if pd.isna(text):
                return ''
        except (TypeError, ValueError):
            # pd.isna raises for some iterables; fall back to string conversion
            pass
        if isinstance(text, str):
            return text
        return str(text)
    
    def _initialize_llm(self):
        """Initialize LLM client with SiliconFlow API"""
        try:
            api_key = 'sk-hmqokjrhfszsquludqhbdzftggjriimfelvjjqwzccxnqxmn'
            os.environ["OPENAI_API_BASE"] = 'https://api.siliconflow.cn/v1'
            os.environ["OPENAI_API_KEY"] = api_key
            
            self.llm = ChatOpenAI(
                model='Qwen/Qwen2.5-7B-Instruct', 
                openai_api_key=api_key,
                openai_api_base='https://api.siliconflow.cn/v1',
                streaming=False,  # Disable streaming for evaluation
                temperature=0.1   # Low temperature for consistent evaluation
            )
            self.llm_available = True
            print("✓ LLM client initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM client: {e}")
            self.llm_available = False
            print("Warning: LLM evaluation will be skipped.")
    
    def _create_ground_truth_dict(self) -> Dict[str, Dict]:
        """Create a dictionary mapping questions to ground truth data"""
        gt_dict = {}
        for _, row in self.ground_truth_df.iterrows():
            question = row['问题'].strip()
            gt_dict[question] = {
                'answer': row['答案'],
                'relevant_chunks': ast.literal_eval(row['relevant_chunks']) if row['relevant_chunks'] else [],
                'relevant_chunks_verified': ast.literal_eval(row['relevant_chunks_verified_with_llm']) if row['relevant_chunks_verified_with_llm'] else [],
                'category': row['category'],
                'question_type': row['问题类型']
            }
        return gt_dict
    
    def evaluate_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single result from official_faq_v0.json format
        
        Args:
            result: Single result dictionary
            
        Returns:
            Dictionary with evaluation metrics
        """
        question = result['question']
        agent_answer = result['answer']
        retrieved_docs = result.get('reranked_docs', '').split(',') if result.get('reranked_docs') else []
        
        # Get ground truth
        if question not in self.ground_truth_dict:
            return {'error': f'Question not found in ground truth: {question}'}
        
        gt_data = self.ground_truth_dict[question]
        gt_answer = gt_data['answer']
        gt_chunks = gt_data['relevant_chunks_verified']  # Use LLM-verified chunks as ground truth
        
        evaluation = {
            'question': question,
            'answer_metrics': self._evaluate_answer_quality(agent_answer, gt_answer),
            'retrieval_metrics': self._evaluate_retrieval(retrieved_docs, gt_chunks)
        }
        
        return evaluation
    
    def evaluate_multi_round_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a multi-round conversation result from multi_round_eval.json format
        
        Args:
            result: Single multi-round result dictionary
            
        Returns:
            Dictionary with evaluation metrics
        """
        question = result['question']
        final_answer = result['final_answer']
        conversation_history = result['conversation_history']
        
        # Get ground truth
        if question not in self.ground_truth_dict:
            return {'error': f'Question not found in ground truth: {question}'}
        
        gt_data = self.ground_truth_dict[question]
        gt_answer = gt_data['answer']
        gt_chunks = gt_data['relevant_chunks_verified']  # Use LLM-verified chunks as ground truth
        
        evaluation = {
            'question': question,
            'iterations': result['iterations'],
            'total_time': result['total_time'],
            'stop_reason': result['stop_reason'],
            'final_answer_metrics': self._evaluate_answer_quality(final_answer, gt_answer),
            'round_evaluations': [],
            'retrieval_metrics': {'round_scores': []},
            'conversation_efficiency': 0.0
        }
        
        # Evaluate each round
        total_retrieval_score = 0
        for i, round_data in enumerate(conversation_history):
            retrieved_docs = round_data.get('retrieved_docs_ids', [])
            round_answer = round_data.get('answer', '')
            
            round_eval = {
                'round': i + 1,
                'answer_metrics': self._evaluate_answer_quality(round_answer, gt_answer),
                'retrieval_metrics': self._evaluate_retrieval(retrieved_docs, gt_chunks),
                'needs_clarification': round_data.get('needs_clarification', False),
                'docs_count': round_data.get('docs_count', 0)
            }
            
            evaluation['round_evaluations'].append(round_eval)
        
        # Calculate conversation efficiency (fewer iterations = better)
        evaluation['conversation_efficiency'] = 1.0 / evaluation['iterations'] if evaluation['iterations'] > 0 else 0.0
        
        return evaluation
    
    def _evaluate_answer_quality(self, agent_answer: str, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate answer quality using various language metrics
        
        Args:
            agent_answer: Agent's answer
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary with various quality metrics
        """
        agent_answer = self._normalize_text(agent_answer)
        ground_truth = self._normalize_text(ground_truth)
        
        # Keyword overlap using jieba tokenization
        keyword_metrics = self._calculate_keyword_overlap_jieba(agent_answer, ground_truth)
        
        # Semantic similarity using m3e-base embedding
        semantic_similarity = self._calculate_semantic_similarity_embedding(agent_answer, ground_truth)
        
        # LLM-based evaluation (Qwen2.5-7B-Instruct score)
        llm_score = self._evaluate_with_llm(agent_answer, ground_truth) if self.llm_available else 0.0
        
        return {
            'keyword_precision': keyword_metrics['keyword_precision'],
            'keyword_recall': keyword_metrics['keyword_recall'],
            'keyword_f1': keyword_metrics['keyword_f1'],
            'semantic_similarity': semantic_similarity,
            'llm_score': llm_score
        }
    
    def _evaluate_retrieval(self, retrieved_docs: List[str], ground_truth_chunks: List[str]) -> Dict[str, float]:
        """
        Evaluate retrieval quality
        
        Args:
            retrieved_docs: List of retrieved document IDs
            ground_truth_chunks: List of ground truth relevant chunk IDs
            
        Returns:
            Dictionary with retrieval metrics
        """
        if not ground_truth_chunks:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'overall_score': 0.0
            }
        
        retrieved_set = set(retrieved_docs)
        gt_set = set(ground_truth_chunks)
        
        # Calculate precision, recall, F1
        if retrieved_set:
            precision = len(retrieved_set & gt_set) / len(retrieved_set)
        else:
            precision = 0.0
        
        recall = len(retrieved_set & gt_set) / len(gt_set)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _calculate_keyword_overlap_jieba(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate keyword overlap using jieba tokenization for Chinese text
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Tokenize using jieba
        tokens1 = set(jieba.lcut(text1))
        tokens2 = set(jieba.lcut(text2))
        
        # Remove single characters and common stop words
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        tokens1 = {token for token in tokens1 if len(token) > 1 and token not in stop_words}
        tokens2 = {token for token in tokens2 if len(token) > 1 and token not in stop_words}
        
        if not tokens2:
            return {'keyword_precision': 0.0, 'keyword_recall': 0.0, 'keyword_f1': 0.0}
        
        # Calculate precision, recall, F1
        intersection = tokens1 & tokens2
        
        precision = len(intersection) / len(tokens1) if tokens1 else 0.0
        recall = len(intersection) / len(tokens2)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'keyword_precision': precision,
            'keyword_recall': recall,
            'keyword_f1': f1
        }
    
    def _calculate_semantic_similarity_embedding(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity using m3e-base embedding model
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            # Get embeddings
            embeddings = self.embedding_model.encode([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _evaluate_with_llm(self, agent_answer: str, ground_truth: str) -> float:
        """
        Evaluate answer quality using Qwen2.5-7B-Instruct with improved prompt design
        
        Args:
            agent_answer: Agent's answer
            ground_truth: Ground truth answer
            
        Returns:
            Score between 0 and 1
        """
        try:
            # Improved prompt design based on GPT-4 evaluation practices for open-ended QA
            prompt = f"""你是一个专业的教育领域评估专家。请评估以下回答的质量。

**问题背景**: 这是关于上海教育考试相关的问题回答评估。

**标准答案**:
{ground_truth}

**待评估回答**:
{agent_answer}

**评估标准**:
请从以下4个维度进行评估，每个维度权重相等：

1. **事实准确性** (25%): 回答中的事实信息是否正确，是否与标准答案一致
2. **信息完整性** (25%): 是否涵盖了标准答案中的关键信息点
3. **语言质量** (25%): 表达是否清晰、准确、符合中文表达习惯
4. **相关性** (25%): 回答是否直接、有效地回答了问题

**评分规则**:
- 9-10分: 优秀，完全满足所有标准
- 7-8分: 良好，基本满足标准，有少量不足
- 5-6分: 一般，部分满足标准，有明显不足
- 3-4分: 较差，基本不满足标准
- 1-2分: 很差，完全不满足标准
- 0分: 无意义或错误回答

**输出要求**:
请只输出一个0-10之间的整数分数，不要包含任何其他文字、解释或标点符号。

分数:"""
            
            response = self.llm.invoke(prompt)
            score_text = response.content.strip()
            
            # Extract numeric score more robustly
            score_match = re.search(r'\b(\d+)\b', score_text)
            if score_match:
                score = float(score_match.group(1))
            else:
                # Fallback: look for any number in the text
                numbers = re.findall(r'\d+', score_text)
                score = float(numbers[0]) if numbers else 0.0
            
            # Normalize to 0-1 range
            normalized_score = min(max(score / 10.0, 0.0), 1.0)
            
            return normalized_score
            
        except Exception as e:
            print(f"Error in LLM evaluation: {e}")
            return 0.0
    
    
    def generate_evaluation_report(self, single_results: List[Dict] = None, 
                                multi_round_results: List[Dict] = None,
                                output_file: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            single_results: List of single result evaluations
            multi_round_results: List of multi-round result evaluations
            output_file: Optional output file path
            
        Returns:
            Comprehensive evaluation report
        """
        report = {
            'evaluation_summary': {},
            'single_result_evaluations': single_results or [],
            'multi_round_evaluations': multi_round_results or [],
            'aggregated_metrics': {}
        }
        
        # Aggregate metrics for single results
        if single_results:
            # Extract answer scores (using LLM score as primary metric)
            answer_scores = [r['answer_metrics']['llm_score'] for r in single_results if 'answer_metrics' in r]
            # Extract retrieval scores (using F1 as primary metric)
            retrieval_scores = [r['retrieval_metrics']['f1'] for r in single_results if 'retrieval_metrics' in r]
            
            report['aggregated_metrics']['single_results'] = {
                'count': len(single_results),
                'answer_metrics': {
                    'average_llm_score': np.mean(answer_scores) if answer_scores else 0,
                    'std_llm_score': np.std(answer_scores) if answer_scores else 0,
                    'min_llm_score': np.min(answer_scores) if answer_scores else 0,
                    'max_llm_score': np.max(answer_scores) if answer_scores else 0
                },
                'retrieval_metrics': {
                    'average_f1': np.mean(retrieval_scores) if retrieval_scores else 0,
                    'std_f1': np.std(retrieval_scores) if retrieval_scores else 0,
                    'min_f1': np.min(retrieval_scores) if retrieval_scores else 0,
                    'max_f1': np.max(retrieval_scores) if retrieval_scores else 0
                }
            }
        
        # Aggregate metrics for multi-round results
        if multi_round_results:
            # Extract answer scores (using LLM score as primary metric)
            answer_scores = [r['final_answer_metrics']['llm_score'] for r in multi_round_results if 'final_answer_metrics' in r]
            # Extract retrieval scores (using F1 as primary metric)
            retrieval_scores = []
            for r in multi_round_results:
                if 'round_evaluations' in r and r['round_evaluations']:
                    round_f1_scores = [round_eval['retrieval_metrics']['f1'] for round_eval in r['round_evaluations']]
                    retrieval_scores.extend(round_f1_scores)
            
            iterations = [r['iterations'] for r in multi_round_results if 'iterations' in r]
            times = [r['total_time'] for r in multi_round_results if 'total_time' in r]
            
            report['aggregated_metrics']['multi_round_results'] = {
                'count': len(multi_round_results),
                'answer_metrics': {
                    'average_llm_score': np.mean(answer_scores) if answer_scores else 0,
                    'std_llm_score': np.std(answer_scores) if answer_scores else 0,
                    'min_llm_score': np.min(answer_scores) if answer_scores else 0,
                    'max_llm_score': np.max(answer_scores) if answer_scores else 0
                },
                'retrieval_metrics': {
                    'average_f1': np.mean(retrieval_scores) if retrieval_scores else 0,
                    'std_f1': np.std(retrieval_scores) if retrieval_scores else 0,
                    'min_f1': np.min(retrieval_scores) if retrieval_scores else 0,
                    'max_f1': np.max(retrieval_scores) if retrieval_scores else 0
                },
                'conversation_metrics': {
                    'average_iterations': np.mean(iterations) if iterations else 0,
                    'average_time': np.mean(times) if times else 0,
                    'conversation_efficiency': np.mean([r['conversation_efficiency'] for r in multi_round_results if 'conversation_efficiency' in r]) if multi_round_results else 0
                }
            }
        
        # Overall summary
        all_answer_scores = []
        all_retrieval_scores = []
        
        if single_results:
            all_answer_scores.extend([r['answer_metrics']['llm_score'] for r in single_results if 'answer_metrics' in r])
            all_retrieval_scores.extend([r['retrieval_metrics']['f1'] for r in single_results if 'retrieval_metrics' in r])
        
        if multi_round_results:
            all_answer_scores.extend([r['final_answer_metrics']['llm_score'] for r in multi_round_results if 'final_answer_metrics' in r])
            for r in multi_round_results:
                if 'round_evaluations' in r and r['round_evaluations']:
                    round_f1_scores = [round_eval['retrieval_metrics']['f1'] for round_eval in r['round_evaluations']]
                    all_retrieval_scores.extend(round_f1_scores)
        
        report['evaluation_summary'] = {
            'total_evaluations': len(single_results or []) + len(multi_round_results or []),
            'answer_metrics': {
                'average_llm_score': np.mean(all_answer_scores) if all_answer_scores else 0,
                'std_llm_score': np.std(all_answer_scores) if all_answer_scores else 0,
                'performance_level': self._get_performance_level(np.mean(all_answer_scores) if all_answer_scores else 0)
            },
            'retrieval_metrics': {
                'average_f1': np.mean(all_retrieval_scores) if all_retrieval_scores else 0,
                'std_f1': np.std(all_retrieval_scores) if all_retrieval_scores else 0,
                'performance_level': self._get_performance_level(np.mean(all_retrieval_scores) if all_retrieval_scores else 0)
            }
        }
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        
        return report
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level based on score"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Shanghai Education Authority Agent')
    parser.add_argument('--ground_truth', required=True, help='Path to ground truth CSV file')
    parser.add_argument('--single_results', help='Path to single results JSON file')
    parser.add_argument('--multi_round_results', help='Path to multi-round results JSON file')
    parser.add_argument('--output', help='Output file path for evaluation report')
    parser.add_argument('--model_path', help='Path to local m3e-base model (default: app/models/m3e-base)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AgentEvaluator(args.ground_truth, args.model_path)
    
    single_evaluations = []
    multi_round_evaluations = []
    
    # Evaluate single results
    if args.single_results:
        with open(args.single_results, 'r', encoding='utf-8') as f:
            single_results = json.load(f)
        
        for result in tqdm(single_results, desc="Evaluating single results"):
            evaluation = evaluator.evaluate_single_result(result)
            single_evaluations.append(evaluation)
    
    # Evaluate multi-round results
    if args.multi_round_results:
        with open(args.multi_round_results, 'r', encoding='utf-8') as f:
            multi_round_results = json.load(f)
        
        for result in tqdm(multi_round_results, desc="Evaluating multi-round results"):
            evaluation = evaluator.evaluate_multi_round_result(result)
            multi_round_evaluations.append(evaluation)
    
    # Generate report
    report = evaluator.generate_evaluation_report(
        single_evaluations, 
        multi_round_evaluations, 
        args.output
    )
    
    # Print summary
    print("Evaluation Summary:")
    print(f"Total evaluations: {report['evaluation_summary']['total_evaluations']}")
    
    print(f"\nAnswer Quality Metrics:")
    answer_metrics = report['evaluation_summary']['answer_metrics']
    print(f"  Average LLM Score: {answer_metrics['average_llm_score']:.3f}")
    print(f"  Std LLM Score: {answer_metrics['std_llm_score']:.3f}")
    print(f"  Performance Level: {answer_metrics['performance_level']}")
    
    print(f"\nRetrieval Quality Metrics:")
    retrieval_metrics = report['evaluation_summary']['retrieval_metrics']
    print(f"  Average F1 Score: {retrieval_metrics['average_f1']:.3f}")
    print(f"  Std F1 Score: {retrieval_metrics['std_f1']:.3f}")
    print(f"  Performance Level: {retrieval_metrics['performance_level']}")
    
    if args.output:
        print(f"\nDetailed report saved to: {args.output}")
