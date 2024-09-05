import chromadb
from flask import Flask, render_template, request
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import uuid
import cv2 as cv
import numpy as np

import torch
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import cv2 as cv1
import pytesseract
import base64
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import uuid
from langchain.schema.document import Document

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain.llms import HuggingFaceHub
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads/'


      

# (1) Define functions for loading images and performing OCR

def pytes(image):
    try:
        # Perform OCR using pytesseract
        result = pytesseract.image_to_string(image)
        return result
    except Exception as e:
        print(f"Error occurred during OCR: {e}")
        return None

def encode_image(image_path):
    try:
        # Encode the image as base64
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error occurred during image encoding: {e}")
        return None

def split_text_into_chunks(text):
    try:
        # Split the text into chunks
        chunk_size = 500
        chunk_overlap = 100
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        return chunks
    except Exception as e:
        print(f"Error occurred during text splitting: {e}")
        return []



def extract_images_from_pdf(pdf_path, output_folder):
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                images = doc[page_num].get_images(full=True)
                for img_idx, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_data = base_image["image"]
                    
                    # Save image bytes to file
                    img_path = os.path.join(output_folder, f"Image_{uuid.uuid4()}.png")
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_data)
    except Exception as e:
        print(f"Error occurred during PDF processing: {e}")


# Load the LLM model and create the Chroma vector database
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}  # If using GPU memory, set device to 'cuda'; otherwise, 'cpu'
encode_kwargs = {'normalize_embeddings': True}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
embedding_function = OpenCLIPEmbeddingFunction()

from chromadb.utils.data_loaders import ImageLoader
image_loader = ImageLoader()

path = "uploads/"
vectorstore = Chroma(collection_name="multi_modal_rag",
                     embedding_function=embedding_model,
                     persist_directory=path + 'new_Chroma_vectorstore')

# Load the vectore base
retriever = vectorstore.as_retriever()

# Load the LLM
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature": 1, "max_new_tokens": 1024, "max_length": 8000, 'context_length': 8000})

# Implement the RAG chain
def plt_img_base64(img_base64):
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    return image_html

def split_image_text_types(docs):
    ''' Split base64-encoded images and texts '''
    b64 = []
    text = []
    for doc in docs:
        temp_text = doc.page_content
        if temp_text not in text:
          text.append(temp_text)
        try:
            temp_image = doc.metadata['image_source']
            if temp_image not in b64:
              b64.append(temp_image)
        except Exception as e:
            continue
    return {
            "images": b64,
            "texts": text
           }

def prompt_func(dict):
    format_texts = "\n".join(dict["context"]['texts'])
    return [
        SystemMessage(content="You're a helpful Question Answering assistant"),
        HumanMessage(
            content=[
                {"type": "text", "text": f'''
Context = {format_texts}

Give full answer for question using above information. If you know the answer more detaily apart from the Context, please provide that answer too.

Question is {dict["question"]}
'''}
                ]
            ),
        AIMessage(content='provide the answer here')
        ]

chain = (
    {"context": retriever | RunnableLambda(split_image_text_types), "question": RunnablePassthrough()}
    | RunnableLambda(prompt_func)
    | llm
)


docs = retriever.get_relevant_documents(
    "pettah road"
)
docs



# (2) Route for file upload and processing
@app.route('/upload_pdf', methods=[ 'POST'])
def upload_file():
    if request.method == 'POST':
        # Handle file upload and processing here
       # question = request.form['question']
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            # Save the uploaded file to a temporary location
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            try:
                uploaded_file.save(file_path)
            except Exception as e:
                print(f"Error occurred while saving file: {e}")
                return "Error occurred while saving file."

            # Process the uploaded PDF file
            try:
                with fitz.open(file_path) as doc:
                    text = ""
                    for page in doc:
                        temp_text = page.get_text()
                        temp_list = temp_text.split('\n')
                        temp_text = '\n'.join(temp_list[4:])
                        text += temp_text

                    # Split text into chunks
                    splitted_texts = split_text_into_chunks(text)
            except Exception as e:
                print(f"Error occurred during PDF processing: {e}")
                return "Error occurred during PDF processing."
            
            
            
            # Embed text into vector store
            doc_ids = [str(uuid.uuid4()) for _ in splitted_texts]
            doc_texts = [
                        Document(page_content=s, metadata={"id": doc_ids[i]})
                        for i, s in enumerate(splitted_texts)
                    ]
            vectorstore.add_documents(doc_texts)


           # Extract images from the uploaded PDF
            output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'Images')
            os.makedirs(output_folder, exist_ok=True)
            extract_images_from_pdf(file_path, output_folder)

                
                


            
            # Load images, perform OCR, and convert to base64
            image_summaries = []
            img_base64_list = []

            # Adjust the path to the uploads directory
            path = "uploads/"

            for img_path in sorted(os.listdir(os.path.join(path, "Images"))):
                try:
                    img = cv.imread(os.path.join(path, "Images", img_path))
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    image_summaries.append(pytes(img))
                    base64_image = encode_image(os.path.join(path, "Images", img_path))
                    img_base64_list.append(base64_image)
                except Exception as e:
                    print(f"Error occurred during image processing: {e}")

                
                
                img_ids = [str(uuid.uuid4()) for _ in img_base64_list]
                image_doc = [
                Document(page_content=s, metadata={"id": img_ids[i],
                                       "image_source" : img_base64_list[i]})
                           for i, s in enumerate(image_summaries)
                            ]

                vectorstore.add_documents(image_doc)
            return jsonify({"image_summaries": image_summaries
                               }) 
                
            # Return results or render a template with the results
            
            #docs = split_image_text_types(docs)
            #llm_result = chain.invoke(question)
            #llm_result1= llm_result.split('}]\nAI: provide the answer here\n')[1]
            #matching_images =  [encode_image(os.path.join(path, "Images", img_path)) for img_path in sorted(os.listdir(os.path.join(path, "Images")))]
            #matching_images=docs["images"]   
           # return render_template('results.html', llm_result1=llm_result1,  matching_images=matching_images, image_summaries=image_summaries)



            # Return results or render a template with the results
    return " file uploaded"



# (2) Route for question answering and processing
@app.route('/question', methods=[ 'POST'])
def question():

    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            docs = retriever.get_relevant_documents(question)
            docs = split_image_text_types(docs)
            llm_result = chain.invoke(question)
            
            # Split llm_result only if it's not empty
            if llm_result:
                llm_result_split = llm_result.split('}]\nAI: provide the answer here\n')
                if len(llm_result_split) > 1:
                    llm_result1 = llm_result_split[1]
                else:
                    llm_result1 = "Answer not found"
            else:
                llm_result1 = "Answer not found"

            matching_images=docs["images"]  

            return jsonify({
                "llm_result": llm_result1,
                "matching_images": matching_images,
            })

        return jsonify({"error": "No question entered"})

    return jsonify({"error": "Invalid request method"})




if __name__ == '__main__':
    app.run(debug=False)
