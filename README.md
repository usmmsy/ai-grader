# AI Homework Grader

A multi-agent system for grading undergraduate mathematics homework using RAG (Retrieval Augmented Generation) and Vision-LLM capabilities.

## Features

*   **Vision-First Ingestion:** Renders PDFs as images to accurately capture math formulas, graphs, and tables using GPT-4o.
*   **RAG Grading:** Retrieves relevant rubric sections to grade student submissions.
*   **Structured Reporting:** Generates detailed Markdown reports with scores, reasoning, and feedback.

## Setup

1.  **Environment:**
    ```bash
    conda create -n aigrader python=3.11
    conda activate aigrader
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    *   Rename `.env.example` to `.env`.
    *   Add your `OPENAI_API_KEY`.

## Usage

### 1. Ingest a Rubric
Load your answer key or grading rubric into the vector database.
```bash
python src/main.py ingest "path/to/rubric.pdf"
```

### 2. Grade a Submission
Grade a student's homework PDF. The system will read the PDF (including graphs/math), compare it to the rubric, and generate a report.
```bash
python src/main.py grade "path/to/student_submission.pdf" --student_id "Student_001"
```

### 3. View Results
Reports are saved to `data/results/Grading_Report_{Student_ID}.md`.

## Architecture

*   **`src/rag/ingestion.py`**: Handles PDF processing (Vision/OCR) and chunking.
*   **`src/agents/grader.py`**: The core logic that queries the DB and prompts the LLM.
*   **`src/utils/vision.py`**: Interface for GPT-4o vision capabilities.

