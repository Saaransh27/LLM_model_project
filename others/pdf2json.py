import os
import PyPDF2
import jsonlines 

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

def convert_pdf_to_json(pdf_path, output_json_path):
    filename = os.path.basename(pdf_path)
    with open(pdf_path, "rb") as f:
        text = extract_text_from_pdf(f)
        json_data = {
            "filename": filename,
            "text": text
        }
    with jsonlines.open(output_json_path, mode='w') as writer:
        writer.write(json_data)

if __name__ == "__main__":
    
    folder_path = r'C:\Users\LENEVO\Downloads\articles'
    
    output_folder = r'D:\LLM_model_project\dataset'
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            output_json_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.json")
            convert_pdf_to_json(pdf_path, output_json_path)