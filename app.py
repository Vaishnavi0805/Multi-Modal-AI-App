import streamlit as st
import openai
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
import torch
import IPython.display as display
from PIL import Image

# OpenAI API key

def transcribe_audio(openai_api_key,audio_file):
    openai.api_key =openai_api_key
    st.audio(audio_file, format="audio/wav")
    st.write("Transcription:")
    
    # Convert audio to text using Whisper ASR
   
    transcript = openai.Audio.transcribe(
        file = audio_file,
        model = "whisper-1",
        response_format="text",
        language="en"
    )
    st.write(transcript)



def generate_image_caption(image_file):
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    processor = AutoProcessor.from_pretrained("microsoft/git-base")
    cache_dir = "E:\\alphaai"
    model_name = "microsoft/git-base"
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # Process image and generate caption
    image = Image.open(image_file)
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values

    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    st.write(generated_caption)

def generate_text(openai_api_key,prompt):
    openai.api_key =openai_api_key
    # Generate text using OpenAI's GPT-3.5
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": '''You're a content creator with a specialization in crafting engaging, topic-specific content. Your primary goal is to generate content that is not only relevant to the topic but also captivating for readers. Your approach involves linking the topic to real-life examples or relatable analogies to ensure the content resonates with readers. Whether you're crafting a narrative, a descriptive blog post, an informative article, or any other format, your aim is to provide valuable insights and information while adhering to the chosen topic.
        
        Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.'''},
        {"role": "user", "content": prompt}
    ]
    )
    st.write("Generated Text:")   
    st.write(completion.choices[0].message['content'])

def main():
    st.title("Multi-Modal AI App")
 
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Audio Transcription", "Image Captioning", "Text Generation"])
    
    if page == "Home":
        st.write("Welcome to the Multi-Modal AI App!")
        st.write("Use the navigation bar on the left to access different functionalities.")
    elif page == "Audio Transcription":
        st.header("Audio Transcription using Whisper")
        openai_api_key = st.text_input("Enter your OPENAI API Key")
        
        if  openai_api_key:
            audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
            
            if audio_file is not None:
                transcribe_audio(openai_api_key,audio_file)
    
    elif page == "Image Captioning":
        st.header("Image Captioning using Open Source Model")
        image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg","jfif"])
        
        if image_file is not None:
            generate_image_caption(image_file)
    
    elif page == "Text Generation":
        st.header("Text Generation using OpenAI Model")
        openai_api_key = st.text_input("Enter your OPENAI API Key")
        
        if  openai_api_key:
            prompt = st.text_area("Enter a text prompt")
            
            if prompt:
                generate_text(openai_api_key,prompt)

if __name__ == "__main__":
    main()
