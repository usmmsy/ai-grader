from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader, UnstructuredWordDocumentLoader
# from unstructured.partition.pdf import partition_pdf  <-- Moved to lazy import to avoid ONNX crashes
from src.utils.vision import vision_helper
import os
import pypdfium2 as pdfium

class IngestionService:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.images_dir = os.path.join(os.getcwd(), "data", "extracted_images")
        os.makedirs(self.images_dir, exist_ok=True)

    def load_file(self, file_path: str, use_vision: bool = True) -> List[Document]:
        """
        Loads a file (PDF or Text) and returns a list of Documents.
        
        Args:
            file_path: Path to the file.
            use_vision: If True, renders PDF pages as images and uses LLM for transcription.
                        If False, uses Unstructured for traditional extraction.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.endswith(".pdf"):
            if use_vision:
                return self._process_pdf_vision(file_path)
            else:
                return self._process_pdf_unstructured(file_path)
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            loader = TextLoader(file_path)
            return loader.load()
        elif file_path.endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
            return loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _process_pdf_vision(self, file_path: str) -> List[Document]:
        """
        Vision-First Extraction:
        1. Render each PDF page as an image.
        2. Send image to LLM for full Markdown transcription.
        """
        print(f"Processing PDF with Vision-First approach: {file_path}")
        documents = []
        
        pdf = pdfium.PdfDocument(file_path)
        n_pages = len(pdf)
        
        for i in range(n_pages):
            print(f"  - Transcribing page {i+1}/{n_pages}...")
            page = pdf[i]
            # Render page to bitmap
            bitmap = page.render(scale=2) # Scale 2 for better resolution
            pil_image = bitmap.to_pil()
            
            # Save temp image
            temp_img_path = os.path.join(self.images_dir, f"temp_page_{i}.jpg")
            pil_image.save(temp_img_path, format="JPEG")
            
            # Transcribe
            transcription = vision_helper.transcribe_page(temp_img_path)
            
            # Create Document
            metadata = {"source": os.path.basename(file_path), "page": i+1}
            documents.append(Document(page_content=transcription, metadata=metadata))
            
            # Cleanup
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
        return documents

    def _process_pdf_unstructured(self, file_path: str) -> List[Document]:
        """
        Traditional Extraction using Unstructured (OCR + Layout Analysis).
        """
        print(f"Partitioning PDF with Unstructured: {file_path}")
        
        try:
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(
                filename=file_path,
                extract_images_in_pdf=True,
                infer_table_structure=True,
                chunking_strategy="by_title",
                max_characters=2000,
                new_after_n_chars=1500,
                combine_text_under_n_chars=1000,
                image_output_dir_path=self.images_dir,
            )
        except ImportError:
            print("Warning: 'unstructured' dependencies missing. Falling back to basic loader.")
            loader = UnstructuredPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            print(f"Error in unstructured partition: {e}. Falling back to basic loader.")
            loader = UnstructuredPDFLoader(file_path)
            return loader.load()

        documents = []
        for element in elements:
            metadata = element.metadata.to_dict()
            metadata["source"] = os.path.basename(file_path)
            
            if "Table" in str(type(element)):
                text_content = element.text
                if element.metadata.text_as_html:
                    text_content += f"\n\nHTML Representation:\n{element.metadata.text_as_html}"
                documents.append(Document(page_content=text_content, metadata=metadata))

            elif "Image" in str(type(element)):
                image_path = element.metadata.image_path
                if image_path and os.path.exists(image_path):
                    description = vision_helper.summarize_image(image_path)
                    content = f"[IMAGE DESCRIPTION]: {description}"
                    documents.append(Document(page_content=content, metadata=metadata))
            
            else:
                documents.append(Document(page_content=str(element), metadata=metadata))

        return documents

    def process_and_chunk(self, file_path: str) -> List[Document]:
        """
        Loads a file and splits it into chunks.
        """
        # We default to Vision for PDFs as it's more robust for math/graphs
        raw_documents = self.load_file(file_path, use_vision=True)
        
        # If the documents are already page-sized chunks from Vision, we might not need aggressive splitting.
        # But for consistency, we still run them through the splitter if they are very long.
        chunks = self.text_splitter.split_documents(raw_documents)
        
        return chunks

ingestion_service = IngestionService()
