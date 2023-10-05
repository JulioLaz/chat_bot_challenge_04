import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import jellyfish
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
from utils import tratamiento_texto, df_dialogo,normalizar,lista_frases_normalizadas,lista_frases, df_dialogo
import warnings
warnings.filterwarnings("ignore")
import nltk
nltk.download('punkt')

nlp = spacy.load('es_core_news_md')


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish
import tkinter as tk
from tkinter import scrolledtext
from time import time
from tkinter import Label, PhotoImage


def dialogo(user_response):
    # Tratamiento de texto
    user_response = tratamiento_texto(user_response)  # Tratando el texto
    user_response = re.sub(r"[^\w\s]", '', user_response)  # Elimina signos de puntuación

    df = df_dialogo.copy()

    vectorizer = TfidfVectorizer()

    for idx, row in df.iterrows():

        dialogos_numero = vectorizer.fit_transform(df_dialogo['dialogo'])
        respuesta_numero = vectorizer.transform([user_response])
        cos_sim = cosine_similarity(dialogos_numero[idx], respuesta_numero)[0][0]

        df.at[idx, 'interseccion'] = len(set(user_response.split()) & set(row['dialogo'].split()))/len(user_response.split())
        df.at[idx, 'similarity'] = cos_sim
        df.at[idx, 'jaro_winkler'] = jellyfish.jaro_winkler_similarity(user_response, row['dialogo'])

        # Calcular la probabilidad:
        df.at[idx, 'probabilidad'] = max(df.at[idx, 'interseccion'], df.at[idx, 'similarity'], df.at[idx, 'jaro_winkler'])

    df.sort_values(by=['probabilidad', 'jaro_winkler'], inplace=True, ascending=False)

    # Probabilidad máxima:
    probabilidad = round(df['probabilidad'].head(1).values[0],2)

   #  #Guardar df:
   #  guardar_dataframe(df)

    if probabilidad >= 0.93:
        print('Respuesta encontrada por el método de comparación de textos - Probabilidad: ', probabilidad)
        print('Respuesta : ',  df['respuesta'].head(1).values[0])
        respuesta = df['respuesta'].head(1).values[0]
    else:
        respuesta = ''
    return respuesta

#Cargar tu modelo entrenado aqui(recuerda siempre cargar el modelo y el vectorizer o tokenizer usado en el entrenamiento del modelo):
ruta_modelo = 'modelo_transformers'
Modelo_TF = BertForSequenceClassification.from_pretrained(ruta_modelo)
tokenizer_TF = BertTokenizer.from_pretrained(ruta_modelo)

#Función para dialogar utilizando el modelo de Machine Learning:
def clasificacion_modelo(pregunta):
  frase = normalizar(pregunta)
  frase = ' '.join(str(elemento) for elemento in frase)
  tokens = tokenizer_TF.encode_plus(
      frase,
      add_special_tokens=True,
      max_length=128,
      padding='max_length',
      truncation=True,
      return_tensors='pt'
  )
  input_ids = tokens['input_ids']
  attention_mask = tokens['attention_mask']

  with torch.no_grad():
      outputs = Modelo_TF(input_ids, attention_mask)

  etiquetas_predichas = torch.argmax(outputs.logits, dim=1)
  etiquetas_decodificadas = etiquetas_predichas.tolist()

  diccionario = {3: 'Continuacion', 10: 'Nombre', 2: 'Contacto', 13: 'Saludos', 14: 'Sentimiento', 9: 'Identidad', 15: 'Usuario', 6: 'ElProfeAlejo', 1: 'Aprendizaje', 0: 'Agradecimiento', 5: 'Edad', 4: 'Despedida', 11: 'Origen', 12: 'Otros', 7: 'Error', 8: 'Funcion'}
  llave_buscada = etiquetas_decodificadas[0]
  clase_encontrada = diccionario[llave_buscada]

  #Buscar respuesta más parecida en la clase encontrada
  df = df_dialogo[df_dialogo['tipo'] == clase_encontrada]
  df.reset_index(inplace=True)
  vectorizer = TfidfVectorizer()
  dialogos_num = vectorizer.fit_transform(df['dialogo'])
  pregunta_num = vectorizer.transform([tratamiento_texto(pregunta)])
  similarity_scores = cosine_similarity(dialogos_num, pregunta_num)
  indice_pregunta_proxima = similarity_scores.argmax()

  if max(similarity_scores)>0.5 and clase_encontrada not in ['Otros']:
    print('Respuesta encontrada por el modelo Transformers - tipo:',clase_encontrada)
    print('Respuesta: ', df['respuesta'][indice_pregunta_proxima])

    respuesta = df['respuesta'][indice_pregunta_proxima]
  else:
    respuesta = ''
  return respuesta

# Función para devolver la respuesta de los documentos
def respuesta_documento(pregunta):
  pregunta = normalizar(pregunta)
  def contar_coincidencias(frase):
    return sum(1 for elemento in pregunta if elemento in frase)

  diccionario = {valor: posicion for posicion, valor in enumerate(lista_frases_normalizadas)}
  lista = sorted(list(diccionario.keys()), key=contar_coincidencias, reverse=True)[:100]

  lista.append(''.join(pregunta))

  TfidfVec = TfidfVectorizer(tokenizer=normalizar)
  tfidf = TfidfVec.fit_transform(lista)

  vals = cosine_similarity(tfidf[-1], tfidf)
  idx = vals.argsort()[0][-2]
  flat = vals.flatten()
  flat.sort()
  req_tfidf = round(flat[-2],2)
  if req_tfidf>=0.22:
    print('Respuesta encontrada por el método TfidfVectorizer - Probabilidad:', req_tfidf)
    print('Respuesta: ', lista_frases[diccionario[lista[idx]]])

    respuesta = lista_frases[diccionario[lista[idx]]]
  else:
    respuesta = ''
  return respuesta

# Función para devolver una respuesta final buscada en todos los métodos disponibles
def respuesta_chatbot(pregunta):
  respuesta = dialogo(pregunta)
  if respuesta != '':
    return respuesta
  else:
    respuesta = clasificacion_modelo(pregunta)
    if respuesta != '':
      return respuesta
    else:
      preg=pregunta
      respuesta = respuesta_documento(preg)
      if respuesta != '':
        return respuesta
      else:
        return 'Respuesta no encontrada'
      
def conversar():
    t0 = time()
    while True:
        pregunta = input("Ingresa tu pregunta ('salir' para finalizar): ")
        if pregunta.lower() == 'salir':
            break
        respuesta_chatbot(pregunta)
    tf = round(time()-t0,1)
    print('tiempo:',tf)
    return respuesta_chatbot(pregunta)
# conversar()

window = tk.Tk()
window.title("Chatbot")

# Create a text widget to display the conversation
conversation_display = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=70, height=6)
conversation_display.grid(column=0, row=0, padx=10, pady=10, columnspan=2)

# Create a Label widget for the user input
user_input_label = Label(window, text="⛔")
user_input_label.config(font=("Helvetica", 16), fg="red")
user_input_label.grid(column=0, row=1, padx=10, pady=10, sticky='w')  # Use 'sticky' to align it to the left (west)

# Create an entry widget for user input
user_input = tk.Entry(window, width=60)
user_input.grid(column=0, row=1, padx=20, pady=10)

# Function to handle user input and display responses
def handle_user_input(event=None):
    user_query = user_input.get()
    user_input.delete(0, tk.END)

    if user_query.lower() == 'salir':
        window.destroy()
        return

    response = respuesta_chatbot(user_query)

    conversation_display.tag_configure("user", foreground="blue", font=("Helvetica", 13))
    conversation_display.tag_configure("chatbot", foreground="green", font=("Helvetica", 13))
    conversation_display.tag_configure("bot", foreground="green", font=("Helvetica", 16))
    
    conversation_display.insert(tk.END, f"Usuario: {user_query}\n", "user")
    conversation_display.insert(tk.END, f"Chatbot: \n {response}", "chatbot")
    conversation_display.insert(tk.END, f"✅\n", "bot")
    
    conversation_display.see(tk.END) 

user_input.bind('<Return>', handle_user_input)

send_button = tk.Button(window, text="Enviar", command=handle_user_input)
send_button.grid(column=1, row=1, padx=10, pady=10)

conversation_display.tag_configure("inicio", foreground="red", font=("Helvetica", 12))
conversation_display.insert(tk.END, "¡Hola! Soy un chatbot. Puedes preguntarme lo que quieras!\n",'inicio')

window.mainloop()