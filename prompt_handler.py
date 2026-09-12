"""
prompt_handler.py
-----------------
Builds structured, parameterized prompts for question generation.
Separated from LLM communication for clean single-responsibility design.
"""

from dataclasses import dataclass
from typing import List


SYSTEM_PROMPT = (
    "You are an expert academic tutor. Your job is to generate clear, accurate study "
    "questions from course material. Always format your output as a numbered list. "
    "For multiple choice questions, include 4 options labeled A–D and mark the correct answer. "
    "For short answer questions, include a brief model answer after each question."
)

@dataclass
class QuestionPromptParams:
    """
    all parameters needed to construct a question-generated prompt
    """
    content: str
    num_questions: int
    difficulty: str
    question_types: List[str]
    course_name: str


class PromptBuilder:
    """Constructs LLM prompts from QuestionPromptParams"""  
    @staticmethod
    def build_question_prompt(params: QuestionPromptParams) -> str:
        """Builds a fully-formed question-generation prompt"""
        types_str = " and ".join(params.question_types)
        prompt= (
            f"Course: {params.course_name}\n"
            f"Difficulty: {params.difficulty.upper()}\n\n"
            f"--- COURSE MATERIAL ---\n{params.content}\n"
            f"--- END OF MATERIAL ---\n\n"
            f"Generate {params.num_questions} {types_str} questions based strictly on the "
            f"material above. Difficulty level: {params.difficulty}. "
            f"Number each question and follow the format instructions."
        )
        return prompt

    @staticmethod
    def get_system_prompt() -> str:
        return SYSTEM_PROMPT