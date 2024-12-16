from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

@app.get('/')
def welcome():
    return {"message": "Bienvenido a la API de Jonay Temino"}

@app.get('/saluda')
def root(name: str): 
    return {'Message': f'Hola {name}, bienvenido!'}

@app.get('/edad')
def age(birth: int): 
    return {'Message': f'Tu edad es de {2024-birth} años.'}

@app.get('/qa')
def question_answering(question: str, context: str): 
    qa_pipeline = pipeline('question-answering')
    answer = qa_pipeline(question=question, context=context)
    return {'answer': answer['answer'], 'score': answer['score']}

@app.get('/translate')
def translate(query: str): 
    translation_pipeline = pipeline('translation_en_to_fr')
    translation = translation_pipeline(query)
    return {'translation': translation[0]['translation_text']}