import sys
import os
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.config import config
from src.agents.extractor import rubric_extractor
from src.agents.grader import grader_agent
from src.rag.ingestion import ingestion_service
from src.utils.reporter import report_generator

def main():
    parser = argparse.ArgumentParser(description="AI Homework Grader System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a rubric file")
    ingest_parser.add_argument("file_path", help="Path to the rubric PDF or text file")

    # Grade Command
    grade_parser = subparsers.add_parser("grade", help="Grade a student submission")
    grade_parser.add_argument("file_path", help="Path to the student submission PDF")
    grade_parser.add_argument("--student_id", default="Student_001", help="ID of the student")

    # Query Command (for testing)
    query_parser = subparsers.add_parser("query", help="Query the rubric knowledge base")
    query_parser.add_argument("text", help="Query text")

    args = parser.parse_args()

    print("AI Grader System Initialized")
    print(f"Using Model: {config.MODEL_NAME}")
    print(f"Database Directory: {config.CHROMA_DB_DIR}")

    if args.command == "ingest":
        rubric_extractor.ingest_rubric(args.file_path)
    
    elif args.command == "grade":
        print(f"Processing submission: {args.file_path}")
        # 1. Extract Submission Data (using Vision-First by default for PDFs)
        submission_docs = ingestion_service.process_and_chunk(args.file_path)
        
        # 2. Run Grader Agent
        result = grader_agent.grade_submission(submission_docs, student_id=args.student_id)
        
        # 3. Generate Report
        report_path = report_generator.generate_markdown_report(result)
        
        # 4. Output Result
        print("\n=== GRADING RESULT ===")
        print(f"Student: {result.student_id}")
        print(f"Total Score: {result.total_score}/{result.total_max_score}")
        print(f"Report saved to: {report_path}")

    elif args.command == "query":
        results = rubric_extractor.query_rubric(args.text)
        for i, doc in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(doc.page_content)
    else:
        parser.print_help()
    
if __name__ == "__main__":
    main()
