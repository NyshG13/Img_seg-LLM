# qwen_analyzer.py

from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

class QwenVLAnalyzer:
    """
    A helper class to analyze images of objects using a Qwen-VL model.
    """
    def __init__(self, model_id="Qwen/Qwen2.5-VL-3B-Instruct", device="cuda"):
        """
        Initializes the Qwen-VL model and processor.

        Args:
            model_id (str): The HuggingFace model ID for Qwen-VL.
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype="auto", 
            device_map="auto",
            trust_remote_code=True
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        self.prompt_template = """You are a scientific AI assistant on a Mars crash site investigation. Analyze the provided image of an object detected by a rover.

Provide a detailed report focusing on:
1.  **Object Identification:** What is the object? If uncertain, provide the best hypothesis (e.g., "metallic debris," "structural component," "natural rock formation").
2.  **Condition Assessment:** Describe its condition (e.g., "damaged," "intact," "weathered," "burnt").
3.  **Potential Origin & Function:** Where might it have come from? What could have been its purpose?
4.  **Hazard & Usability Analysis:** Is this object a potential hazard for the rover? Could any part of it be salvaged or useful?

Keep your analysis professional, technical, and concise for mission control."""
        print(f"[QwenVLAnalyzer] Initialized model {model_id} on device {self.device}.")

    def analyze_object(self, object_image: Image.Image) -> str:
        """
        Generates a description for a single object image.

        Args:
            object_image (Image.Image): A PIL Image of the cropped object.

        Returns:
            str: The generated textual description of the object.
        """
        if not isinstance(object_image, Image.Image):
            raise TypeError("object_image must be a PIL.Image.Image")
            
        inputs = self.processor(
            text=self.prompt_template, 
            images=object_image, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=1024)
        
        response = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        
        # Clean up the response by removing the prompt part
        cleaned_response = response.replace(self.prompt_template, "").strip()
        
        return cleaned_response