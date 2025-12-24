from typing import List, Optional
from pydantic import BaseModel, Field

class GradeItem(BaseModel):
    question_id: str = Field(..., description="The ID of the question being graded")
    score: float = Field(..., description="The score awarded for this question")
    max_score: float = Field(..., description="The maximum possible score for this question")
    reasoning: str = Field(..., description="Explanation for the score, citing specific errors or correct points")
    feedback: str = Field(..., description="Constructive feedback for the student")

class GradingResult(BaseModel):
    student_id: str = Field(..., description="Identifier for the student submission")
    grades: List[GradeItem] = Field(..., description="List of grades for each question")
    total_score: float = Field(..., description="Total score awarded")
    total_max_score: float = Field(..., description="Total possible score")
    general_comments: str = Field(..., description="Overall comments on the submission")

