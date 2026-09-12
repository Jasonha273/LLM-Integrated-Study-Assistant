"""
main pipeline: reads YAML config file and 
generates questions from local llm
 based on the provided context and parameters.
"""

import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from llm_client import LLMClient, LLMClientError, LLMConfig
from prompt_handler import PromptBuilder, QuestionPromptParams



#Data Models
@dataclass
class CourseConfig:
    course_name: str
    file: str
    num_questions: int
    difficulty: str
    question_types: List[str]


@dataclass
class GenerationResult:
    """Holds the output for one course's question generation run."""
    course_name: str
    questions: str
    success: bool
    error: Optional[str] = None

#Pipeline Functions

class QuestionGenerator:
    """
    The full pipeline:
    1.Load the YAML config file
    2.Read each course document
    3.Build the prompt for the LLM
    4.Call the LLM via the LLMClient to generate questions
    5.Return the structured results
    """

    def __init__(
        self,
        config_path: str = "courses/config.yaml",
        courses_dir: str = "courses",
        llm_config: Optional[LLMConfig] = None,
    ):
        self.config_path = Path(config_path)
        self.courses_dir = Path(courses_dir)
        self.llm = LLMClient(llm_config or LLMConfig())
        self.prompt_builder = PromptBuilder()

    #config loading
    def load_config(self) -> List[CourseConfig]:
        """Load the YAML config file and return a list of CourseConfig objects."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_data = yaml.safe_load(f)
        courses = []
        for course in config_data.get("courses", []):
            courses.append(
                CourseConfig(
                    course_name=course["name"],
                    file=course["file"],
                    num_questions = course.get("num_questions", 5),
                    difficulty = course.get("difficulty", "medium"),
                    question_types = course.get("question_types", ["multiple_choice"]),
                )
            )
        return courses

    #document loading
    def load_document(self, filename: str) -> str:
        path = self.courses_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Course document not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    #Core Generation
    def generate_for_course(self, course: CourseConfig) -> GenerationResult:
        """Runs the full pipeline for a single course"""
        print(f"\n{'─'*50}")
        print(f"  Course : {course.course_name}")
        print(f"  File   : {course.file}")
        print(f"  Qs     : {course.num_questions}  |  Difficulty: {course.difficulty}")
        print(f"{'─'*50}")

        try:
            content= self.load_document(course.file)
            params = QuestionPromptParams(
                content=content,
                num_questions = course.num_questions,
                difficulty=course.difficulty,
                question_types=course.question_types,
                course_name=course.course_name,
            )
            prompt = self.prompt_builder.build_question_prompt(params)
            system = self.prompt_builder.get_system_prompt()
 
            print("  Querying LLM...")
            questions = self.llm.generate(prompt, system_prompt=system)

            print("Questions attained.")
            print(questions)

            return GenerationResult(
                course_name=course.course_name,
                questions=questions,
                success=True,
            )            
            
        except (FileNotFoundError, LLMClientError) as e:
            print(f"  ✗ Error: {e}")
            return GenerationResult(
                course_name=course.course_name,
                questions="",
                success=False,
                error=str(e),
            )

    def run(self) -> List[GenerationResult]:
        """Load config and run generation for all courses."""
        print("║    LLM Study Assistant Pipeline  ║")

        if not self.llm.is_available():
            print("\nOllama Server not detected.")
            print("   Start with: ollama serve")
            print("   Then pull a model:  ollama pull llama3.2\n")
        courses= self.load_config()

        print(f"\nLoaded {len(courses)} course(s) from config")

        results = [self.generate_for_course(c) for c in courses]
        passed = sum(1 for r in results if r.success)
        print(f"\n{'═'*50}")
        print(f"  Done: {passed}/{len(results)} courses processed successfully.")
        print(f"{'═'*50}\n")
        return results

    def save_results(self, results: List[GenerationResult], output_file: str = "output.json"):
        """pushes data to a JSON file for later review"""
        data= [
            {
                "course":r.course_name,
                "success":r.success,
                "questions":r.questions,
                "error":r.error,
            }
            for r in results
        ]
        with open(output_file, "w", encoding ="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Results Saved to {output_file}")

#Entry Point

if __name__ == "__main__":
    generator = QuestionGenerator(
        config_path = "courses/config.yaml",
        courses_dir = "courses",
        llm_config = LLMConfig(
            model = "llama3.2",
            temperature= 0.7,
        ),
    )
    results= generator.run()
    generator.save_results(results)