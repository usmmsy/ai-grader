from typing import List, Union

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from src.config import config
from src.rag.vector_store import vector_store_manager
from src.schemas.models import GradingResult

class GraderAgent:
    def __init__(self):
        # self.llm = ChatOpenAI(model=config.MODEL_NAME, api_key=config.OPENAI_API_KEY, temperature=0)
        self.llm = ChatOllama(model="qwen3-vl:8b", temperature=0, num_gpu=99)
        self.parser = PydanticOutputParser(pydantic_object=GradingResult)
        self.rubric_collection = "rubrics"

    def _format_submission(self, submission: Union[str, List[Document]]) -> str:
        if isinstance(submission, list):
            return "\n\n".join([doc.page_content for doc in submission])
        return submission

    def grade_submission(self, submission: Union[str, List[Document]], student_id: str = "Unknown") -> GradingResult:
        submission_text = self._format_submission(submission)
        
        print("Retrieving rubric context...")
        # Strategy: We query the vector store for the most relevant rubric sections.
        # Since the submission might cover multiple questions, we ideally want the whole relevant rubric.
        # For now, we query with the first chunk of the submission to get the general context/assignment match.
        # In a more advanced version, we would extract "Question X" headers and query for each.
        query_text = submission_text[:2000] 
        results = vector_store_manager.query(self.rubric_collection, query_text, k=5)
        
        if not results:
            print("WARNING: No rubric context found in the vector database.")
            print("Please ensure you have run the 'ingest' command with the rubric file first.")
            # We can either return an error or proceed (but grading will likely be poor)
            # For now, we proceed but the LLM will know context is missing.
            context_text = "NO RUBRIC FOUND. PLEASE GRADE BASED ON GENERAL KNOWLEDGE."
        else:
            context_text = "\n\n".join([doc.page_content for doc in results])
        
        # Construct Prompt
        prompt = ChatPromptTemplate.from_template(
            """
            You are an expert academic grader for undergraduate mathematics.
            
            Your task is to grade a student's homework submission based STRICTLY on the provided rubric and solution context.
            
            ### RUBRIC & SOLUTION CONTEXT:
            {context}
            
            ### STUDENT SUBMISSION:
            {submission}
            
            ### INSTRUCTIONS:
            1. Identify each question answered in the submission.
            2. Compare the student's work against the solution/rubric.
            3. Assign a score and provide detailed reasoning.
            4. If the rubric doesn't explicitly cover a specific error, use your general domain knowledge but be lenient.
            5. Output the result in the specified JSON format.
            
            {format_instructions}
            
            Student ID: {student_id}
            """
        )
        
        # Invoke Chain
        chain = prompt | self.llm | self.parser
        
        print("Grading submission...")
        result = chain.invoke({
            "context": context_text,
            "submission": submission_text,
            "format_instructions": self.parser.get_format_instructions(),
            "student_id": student_id
        })
        
        return result

grader_agent = GraderAgent()
