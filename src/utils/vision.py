import base64
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.config import config

class VisionHelper:
    def __init__(self):
        # Use local Qwen3-VL via Ollama for extraction (Cost: $0)
        # num_gpu=99 ensures all layers are offloaded to the GPU (RTX 4060 8GB)
        # self.llm = ChatOllama(model="qwen3-vl:4b", temperature=0, num_gpu=99)
        self.llm = ChatOllama(model="qwen3-vl:8b", temperature=0, num_gpu=99)
        
        # Keep GPT-4o available if needed for complex reasoning later, 
        # but for now we default to the local model for vision tasks.
        # self.llm_backup = ChatOpenAI(model="gpt-4o", api_key=config.OPENAI_API_KEY)

    def encode_image(self, image_path: str) -> str:
        """Encodes an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def summarize_image(self, image_path: str, context: str = "math homework") -> str:
        """
        Generates a semantic description of an image (graph, diagram, etc.).
        """
        base64_image = self.encode_image(image_path)
        
        prompt = f"""
        Analyze this image from a {context} context. 
        If it is a graph, describe the axes, the shape of the curve, key points (intercepts, vertices), and any labels.
        If it is a table, summarize the data trends.
        If it is a formula, transcribe it into LaTeX format.
        Provide a detailed, factual description that allows someone to understand the image without seeing it.
        """

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        response = self.llm.invoke([message])
        return response.content

    def transcribe_page(self, image_path: str) -> str:
        """
        Transcribes a full page image into Markdown, preserving math and describing visuals.
        """
        base64_image = self.encode_image(image_path)
        
        prompt = """
        You are an expert data extractor for a math grading system.
        Transcribe the content of this page into clean Markdown.
        
        Rules:
        1. Text: Transcribe text exactly as it appears.
        2. Math: Transcribe ALL math formulas and symbols into LaTeX format (wrapped in $ or $$).
        3. Visuals: If you see a graph, diagram, or table, insert a description in blockquotes like:
           > [Graph Description: A parabola opening upwards with vertex at (0,0)...]
        4. Structure: Preserve headers and lists.
        """

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        response = self.llm.invoke([message])
        return response.content

vision_helper = VisionHelper()
