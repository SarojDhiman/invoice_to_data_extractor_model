from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import textract
import os
import time

# Set the device
# DEVICE = "cuda"
DEVICE = "cpu"
# DEVICE = "cuda:0"
# Load the Pix2Struct model and processor
model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-docvqa-large").to(DEVICE)
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-docvqa-large")

def generate(img, questions):
    inputs = processor(images=[img for _ in range(len(questions))],
                      text=questions, return_tensors="pt").to(DEVICE)
    predictions = model.generate(**inputs, max_new_tokens=256)
    return zip(questions, processor.batch_decode(predictions, skip_special_tokens=True))

def convert_pdf_to_images(filename):
    images = convert_from_path(filename)
    return images

def extract_text_from_image(image):
    # Use Textract to extract text from the image
    text = textract.process(image)
    return text.decode("utf-8")

def process_pdf_to_image_and_text(pdf_filename, questions):
    # Convert PDF to images
    images = convert_pdf_to_images(pdf_filename)
    completions_filename = "out.txt"  # Provide the desired output text file name
    
    all_answers = []

    for page_no, image in enumerate(images, 1):
        image_filename = f"img_page_{page_no}"
        rawtext_filename = f"raw_text_page_{page_no}"

        # Save the image
        image.save(f"{image_filename}.png")
        completions = generate(image, questions)

        # Print the completions to the terminal
        page_answers = []
        for question, answer in completions:
            page_answers.append({
                "Question": question,
                "Answer": answer
            })
            print(f"Page {page_no}")
            print(f"Question: {question}")
            print(f"Answer: {answer}\n")

        all_answers.append(page_answers)

        # Extract text from the image using Textract
        extracted_text = extract_text_from_image(f"{image_filename}.png")

        # Write the extracted text to a text file
        with open(f"{rawtext_filename}.txt", "w") as output_file:
            output_file.write(extracted_text)

        # Write the completions to a text file
        with open(completions_filename, "a") as output_file:
            for question, answer in zip(questions, page_answers):
                output_file.write(f"Page {page_no}\nQuestion: {question}\nAnswer: {answer}\n\n")

    return all_answers

pdf_filename = "new_02.pdf"
questions = [
    "what is the company address?",
    "what is the date of?",
    "what is the total?",
    "what is billed address?"]

start = time.time()
all_page_answers = process_pdf_to_image_and_text(pdf_filename, questions)
print("Time taken:", (time.time() - start) / 60)

