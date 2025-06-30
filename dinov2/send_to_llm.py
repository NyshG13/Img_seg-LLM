import os
import json
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from fpdf import FPDF

# os.makedirs("json_output", exist_ok=True)
os.makedirs("weird_objects_json", exist_ok=True)

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"  # or "Qwen/Qwen1.5-7B-Chat"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")

prompt_template = """You are a scientific AI assistant stationed on Mars, assisting a crash site investigation mission. Your current task is to analyze an image captured by the Husarion Panther robot at the Marsyard crash site.

                                      Carefully examine the terrain, structures, possible debris, or anomalies in the image. Then provide a detailed report describing:

                                      1. The environment (terrain, obstacles, lighting, surface texture)

                                      2. Any evidence of the crashed expedition (e.g., wreckage, parts, signs of impact)

                                      3. Any objects of interest or potential hazards for the robots

                                      4. Hypotheses about what may have happened based on the scene. If there was a crash, elaborate how it could have happened.

                                      5. Which of the itmes found can be used further and which are harmful.

                                      Assume your audience is mission control and scientific investigators on Earth. Use technical yet readable language. Keep the tone professional, observational, and concise."""

image_folder = "weird_object"
os.makedirs(image_folder, exist_ok=True) 

# image_folder = "weird_object"

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        img_path = os.path.join(image_folder, filename)
        image = Image.open(img_path).convert("RGB")

        # Encode and run through Qwen
        inputs = processor(text=prompt_template, images=image, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=1024)
        response = processor.batch_decode(output, skip_special_tokens=True)[0]

        # Save JSON
        json_data = {
            "image": filename,
            "description": response.strip()
        }

        json_filename = os.path.splitext(filename)[0] + ".json"
        with open(os.path.join("weird_objects_json", json_filename), "w") as f:
            json.dump(json_data, f, indent=4)

        print(f"✅ Processed {filename}")


#to save as pdf 
pdf.add_page()
pdf.set_font("Times", "B", 18)
pdf.cell(0, 10, "Marsyard Rover Anomaly Report", ln=True, align="C")
pdf.set_font("Times", "", 12)
pdf.ln(10)
pdf.multi_cell(0, 10, "This report summarizes all anomalous objects detected by the rover in the Marsyard test environment. Each page includes the image and its LLM-generated analysis.")

json_folder = "weird_objects_json"
img_folder = "weird_objects"

for json_file in sorted(os.listdir(json_folder)):
    if json_file.endswith(".json"):
        with open(os.path.join(json_folder, json_file)) as f:
            data = json.load(f)

        img_filename = data["image"]
        description = data["description"]

        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Object: {img_filename}", ln=True)

        # Add image
        img_path = os.path.join(img_folder, img_filename)
        img = Image.open(img_path)
        img.save("temp_image.jpg")  # convert in case of incompatible format
        pdf.image("temp_image.jpg", x=30, y=30, w=150)
        pdf.ln(100)

        # Add description
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, f"LLM Description:\n{description}")
        pdf.ln(5)

pdf.output("marsyard_anomaly_report.pdf")
print("✅ Report generated as marsyard_anomaly_report.pdf")