import math

from transformers import AutoTokenizer, pipeline


class ModelStandard:
    """Standard model for summarisation
    Uses a pre-trained model and the huggingface pipeline method"""

    def __init__(self, model_name: str):
        """Initialise the model. Set the pipeline counter to 0"""
        self.model_name = model_name
        self.pipeline_counter = 0

    def get_model_name(self):
        return self.model_name

    def generate_model(self, model_name_or_path: str, device: str):
        """Generate a model from the given model name or path"""
        llm = pipeline("summarization", model=model_name_or_path, device=device)
        return llm

    def perform_inference(
        self, llm: pipeline, text: str, min_length: int, max_length: int
    ) -> str:
        """Perform inference on the given text"""
        self.pipeline_counter += 1

        # Run LLM on input text
        result = llm(
            text,
            min_length=math.ceil(min_length),
            max_length=math.floor(max_length),
        )
        return result[0]["summary_text"]

    def get_tokenizer(self):
        """Get tokenizer for model"""
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def get_pipeline_counter(self):
        """Return the pipeline counter"""
        return self.pipeline_counter
