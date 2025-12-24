import os
from src.schemas.models import GradingResult

class ReportGenerator:
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_markdown_report(self, result: GradingResult, filename: str = None) -> str:
        """
        Generates a Markdown report from the grading result and saves it.
        """
        if not filename:
            filename = f"Grading_Report_{result.student_id}.md"
        
        file_path = os.path.join(self.output_dir, filename)
        
        markdown_content = f"""# Grading Report
**Student ID:** {result.student_id}
**Date:** {os.path.basename(file_path)}
**Score:** {result.total_score} / {result.total_max_score} ({(result.total_score/result.total_max_score)*100:.1f}%)

---

## General Comments
{result.general_comments}

---

## Detailed Breakdown

"""
        for item in result.grades:
            markdown_content += f"### Question {item.question_id}\n"
            markdown_content += f"**Score:** {item.score} / {item.max_score}\n\n"
            markdown_content += f"**Reasoning:**\n{item.reasoning}\n\n"
            markdown_content += f"**Feedback:**\n{item.feedback}\n\n"
            markdown_content += "---\n\n"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        return file_path

report_generator = ReportGenerator()
