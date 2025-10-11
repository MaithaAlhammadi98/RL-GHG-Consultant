"""
Enhanced Reward Function for GHG RL Agent
Multi-component reward that measures:
1. Answer quality (accuracy, relevance)
2. Retrieval quality (did we get relevant chunks?)
3. Action selection quality (did we choose the right retrieval strategy?)
4. Answer grounding (does answer use retrieved chunks?)
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
import re


def calculate_enhanced_reward(
    question: str,
    action: str,
    chunks: List[str],
    agent_answer: str,
    judge_score: float,
    judge_verdict: str,
    expected_action: Optional[str] = None,
    question_category: Optional[str] = None
) -> Dict[str, float]:
    """
    Calculate multi-component reward for RL agent
    
    Args:
        question: User question
        action: Action taken by agent (broad, legal_only, etc.)
        chunks: Retrieved document chunks
        agent_answer: Generated answer
        judge_score: Score from OpenAI judge (0.0-1.0)
        judge_verdict: Verdict from judge (thumbs_up/thumbs_down)
        expected_action: Expected optimal action (if known)
        question_category: Category of question (legal, financial, company, etc.)
    
    Returns:
        Dict with component rewards and total
    """
    rewards = {}
    
    # Component 1: Answer Quality (from judge) - Weight: 50%
    rewards['answer_quality'] = _answer_quality_reward(judge_score, judge_verdict)
    
    # Component 2: Retrieval Quality - Weight: 20%
    rewards['retrieval_quality'] = _retrieval_quality_reward(question, chunks)
    
    # Component 3: Action Selection Quality - Weight: 15%
    rewards['action_quality'] = _action_selection_reward(
        action, expected_action, question_category
    )
    
    # Component 4: Answer Grounding - Weight: 15%
    rewards['answer_grounding'] = _answer_grounding_reward(chunks, agent_answer)
    
    # Total weighted reward
    rewards['total'] = (
        0.50 * rewards['answer_quality'] +
        0.20 * rewards['retrieval_quality'] +
        0.15 * rewards['action_quality'] +
        0.15 * rewards['answer_grounding']
    )
    
    return rewards


def _answer_quality_reward(judge_score: float, judge_verdict: str) -> float:
    """
    Reward based on answer quality from judge
    
    Scale:
    - 0.90-1.00: +1.0 (excellent)
    - 0.75-0.89: +0.7 (good)
    - 0.60-0.74: +0.3 (acceptable)
    - 0.40-0.59:  0.0 (neutral)
    - 0.20-0.39: -0.5 (poor)
    - 0.00-0.19: -1.0 (very poor)
    """
    if judge_score >= 0.90:
        return 1.0
    elif judge_score >= 0.75:
        return 0.7
    elif judge_score >= 0.60:
        return 0.3
    elif judge_score >= 0.40:
        return 0.0
    elif judge_score >= 0.20:
        return -0.5
    else:
        return -1.0


def _retrieval_quality_reward(question: str, chunks: List[str]) -> float:
    """
    Reward based on retrieval quality
    
    Measures:
    - Did we retrieve any chunks? (basic check)
    - Are chunks relevant to question? (keyword overlap)
    - Do chunks contain domain-specific terms? (GHG-related)
    """
    if not chunks:
        return -1.0  # No retrieval is bad
    
    # Extract keywords from question
    question_lower = question.lower()
    question_keywords = _extract_keywords(question_lower)
    
    # Check relevance of chunks
    relevance_scores = []
    for chunk in chunks[:4]:  # Check top 4 chunks
        chunk_lower = chunk.lower()
        
        # Keyword overlap
        overlap = sum(1 for kw in question_keywords if kw in chunk_lower)
        relevance = overlap / max(len(question_keywords), 1)
        relevance_scores.append(relevance)
        
        # Bonus for GHG-specific terms
        ghg_terms = [
            'emissions', 'greenhouse', 'carbon', 'methane', 'co2', 'scope',
            'ghg', 'climate', 'reporting', 'calculation', 'protocol'
        ]
        has_ghg_terms = any(term in chunk_lower for term in ghg_terms)
        if has_ghg_terms:
            relevance_scores[-1] += 0.2
    
    # Average relevance
    avg_relevance = sum(relevance_scores) / len(relevance_scores)
    
    # Convert to reward scale
    if avg_relevance >= 0.7:
        return 1.0
    elif avg_relevance >= 0.5:
        return 0.5
    elif avg_relevance >= 0.3:
        return 0.0
    else:
        return -0.5


def _action_selection_reward(
    action: str,
    expected_action: Optional[str],
    question_category: Optional[str]
) -> float:
    """
    Reward based on action selection quality
    
    If expected_action is provided, check if we selected it.
    Otherwise, infer expected action from question_category.
    """
    # If we have expected action, use it
    if expected_action:
        return 1.0 if action == expected_action else -0.5
    
    # Otherwise, infer from category
    if question_category:
        inferred_action = _infer_action_from_category(question_category)
        if inferred_action:
            return 0.5 if action == inferred_action else -0.3
    
    # No information to judge action quality
    return 0.0


def _infer_action_from_category(category: str) -> Optional[str]:
    """Infer expected action from question category"""
    category_lower = category.lower()
    
    if 'legal' in category_lower or 'regulatory' in category_lower:
        return 'legal_only'
    elif 'financial' in category_lower or 'esg' in category_lower:
        return 'financial_only'
    elif 'company' in category_lower or 'corporate' in category_lower:
        return 'company_only'
    elif 'technical' in category_lower or 'general' in category_lower:
        return 'broad'
    
    return None


def _answer_grounding_reward(chunks: List[str], agent_answer: str) -> float:
    """
    Reward based on how well answer is grounded in retrieved chunks
    
    Measures:
    - Does answer contain information from chunks?
    - Are there hallucinations (answer content not in chunks)?
    """
    if not chunks or not agent_answer:
        return 0.0
    
    answer_lower = agent_answer.lower()
    
    # Check if answer uses chunk information
    # Look for numerical values, specific terms, citations
    
    # Extract key phrases from chunks (simplified)
    chunk_terms = set()
    for chunk in chunks:
        chunk_lower = chunk.lower()
        # Extract potential key terms (simple heuristic)
        words = re.findall(r'\b\w{4,}\b', chunk_lower)
        chunk_terms.update(words[:50])  # Top 50 words per chunk
    
    # Check overlap between answer and chunks
    answer_words = set(re.findall(r'\b\w{4,}\b', answer_lower))
    
    if not answer_words:
        return 0.0
    
    overlap = len(answer_words & chunk_terms)
    overlap_ratio = overlap / len(answer_words)
    
    # Reward scale
    if overlap_ratio >= 0.6:
        return 1.0  # Well grounded
    elif overlap_ratio >= 0.4:
        return 0.5  # Moderately grounded
    elif overlap_ratio >= 0.2:
        return 0.0  # Minimally grounded
    else:
        return -0.5  # Poorly grounded (potential hallucination)


def _extract_keywords(text: str) -> List[str]:
    """Extract keywords from text (simple version)"""
    # Remove common stop words
    stop_words = {
        'the', 'is', 'are', 'what', 'how', 'when', 'where', 'who', 'which',
        'do', 'does', 'did', 'can', 'should', 'would', 'could', 'a', 'an',
        'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'about', 'as', 'that', 'this', 'these', 'those'
    }
    
    words = re.findall(r'\b\w{3,}\b', text.lower())
    keywords = [w for w in words if w not in stop_words]
    return keywords


def simple_reward(judge_score: float, judge_verdict: str) -> float:
    """
    Simple backward-compatible reward function
    (for comparison with old system)
    """
    if judge_verdict == "thumbs_up":
        return 1.0
    elif judge_verdict == "thumbs_down":
        return -1.0
    else:
        return 0.0


# Backward compatibility with old reward.py
def feedback_reward(tag: str) -> float:
    """Legacy function for backward compatibility"""
    t = (tag or "").lower()
    if t in ("up", "", "thumbs_up", "good", "helpful", "positive"):
        return 1.0
    if t in ("down", "", "thumbs_down", "bad", "not_helpful", "negative"):
        return -1.0
    return 0.0


def scale_reward(score: float, min_r: float = -1.0, max_r: float = 1.0) -> float:
    """Legacy function for backward compatibility"""
    score = float(score)
    return max(min(score, max_r), min_r)


if __name__ == "__main__":
    # Test the reward function
    print("Testing Enhanced Reward Function")
    print("=" * 60)
    
    # Test case 1: Good answer with good retrieval
    test_chunks = [
        "Scope 1 emissions are direct emissions from owned or controlled sources.",
        "Scope 2 emissions are indirect emissions from purchased electricity.",
        "These categories are defined in the GHG Protocol."
    ]
    test_answer = "Scope 1 emissions are direct emissions while Scope 2 are from purchased electricity."
    
    rewards = calculate_enhanced_reward(
        question="What are Scope 1 and Scope 2 emissions?",
        action="broad",
        chunks=test_chunks,
        agent_answer=test_answer,
        judge_score=0.85,
        judge_verdict="thumbs_up",
        expected_action="broad",
        question_category="general"
    )
    
    print("Test Case 1: Good answer")
    for component, value in rewards.items():
        print(f"  {component}: {value:.3f}")
    
    print("\n" + "=" * 60)
    
    # Test case 2: Wrong action but good answer
    rewards2 = calculate_enhanced_reward(
        question="What are the reporting requirements under 40 CFR Part 98?",
        action="broad",
        chunks=test_chunks,
        agent_answer=test_answer,
        judge_score=0.40,
        judge_verdict="thumbs_down",
        expected_action="legal_only",
        question_category="legal"
    )
    
    print("Test Case 2: Wrong action selection")
    for component, value in rewards2.items():
        print(f"  {component}: {value:.3f}")

