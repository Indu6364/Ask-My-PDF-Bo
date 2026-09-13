import google.generativeai as genai

genai.configure(API_KEY = "your_actual_api_key")

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)