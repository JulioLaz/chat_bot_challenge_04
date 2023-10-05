import pandas as pd
import re, os, pickle
import spacy
import jellyfish
from docx import Document
import csv
import nltk
nltk.download('punkt')
nlp = spacy.load('es_core_news_md')

#importar verbos:
import pickle

archivo_pickle_verbos = "verbos/lista_verbos.pickle"
archivo_pickle_verbos_irregulares = "verbos/verbos_irregulares.pickle"

# Importar la lista_verbos:
with open(archivo_pickle_verbos, "rb") as verbos:
        lista_verbos = pickle.load(verbos)

# Importar el diccionario:
with open(archivo_pickle_verbos_irregulares, "rb") as verbos_irregulares:
        lista_verbos_irregulares = pickle.load(verbos_irregulares)

# print(lista_verbos)
# print('---------------------------------')
# print(lista_verbos_irregulares)

#Función para encontrar la raiz de las palabras
def raiz(palabra):
    max_similitud = 0.0
    for verbo in lista_verbos:
        similitud = jellyfish.jaro_similarity(palabra, verbo)
        if similitud > max_similitud:
            max_similitud = similitud
            palabra_encontrada = verbo

    if max_similitud >= 0.93:
        return palabra_encontrada
    else:
        return palabra

def tratamiento_texto(texto):
  trans = str.maketrans('áéíóú','aeiou')
  texto = texto.lower()
  texto = texto.translate(trans)
  texto = re.sub(r"[^\w\s+\-*/]", '', texto)
  texto = " ".join(texto.split())
  # print('texto de tratamiento_texto: ',texto)
  return texto

#Función para reemplazar el final de una palabra por 'r'
def reemplazar_terminacion(frase):
    terminaciones = ["es", "me", "as", "ste", "te"]
    palabras = frase.split()

    for i, palabra in enumerate(palabras):
        for terminacion in terminaciones:
            if palabra.endswith(terminacion) and len(palabra) > len(terminacion):
                palabras[i] = palabra[:-len(terminacion)] + "r"
                break

    nueva_frase = " ".join(palabras)
    return nueva_frase

#Función para adicionar o eliminar tokens
def revisar_tokens(texto, tokens):
  texto=tratamiento_texto(texto)
  if len(tokens)==0:
    if any(name in texto for name in ['cientifico de datos', 'data scientist']):
        tokens.append('datascientist')
    if any(name in texto for name in ['elprofealejo', 'el profe alejo', 'profe alejo', 'profealejo']):
        tokens.append('elprofealejo')
    if any(name in texto for name in ['ciencia de datos', 'data science']):
        tokens.append('datascience')
  else:
    elementos_a_eliminar = ["cual", "que", "quien", "cuanto", "cuando", "como"]
    if 'hablame' in texto and 'hablar' in tokens: elementos_a_eliminar.append('hablar')
    elif 'cuentame' in texto and 'contar' in tokens: elementos_a_eliminar.append('contar')
    elif 'hago' in texto and 'hacer' in tokens: elementos_a_eliminar.append('hacer')
    elif 'entiendes' in texto and 'entender' in tokens: elementos_a_eliminar.append('entender')
    elif 'sabes' in texto and 'saber' in tokens: elementos_a_eliminar.append('saber')
    tokens = [token for token in tokens if token not in elementos_a_eliminar]
  return tokens

#Función para devolver los tokens normalizados del texto
def normalizar(texto):
    tokens = []
    tokens = revisar_tokens(texto, tokens)
    doc = nlp(texto)
    for t in doc:
        # Obtener el lemma
        lemma = lista_verbos_irregulares.get(t.text, t.lemma_)
        # lemma=lista_verbos_irregulares.get(t.text, t.lemma_.split()[0])

        # Verificar si lemma es una cadena no vacía
        if lemma and isinstance(lemma, str):
            lemma = re.sub(r'[^\w\s+\-*/]', '', lemma)

            if t.pos_ in ('VERB','PROPN','PRON','NOUN','AUX','SCONJ','ADJ','ADV','NUM') or lemma in lista_verbos:
                if t.pos_ == 'VERB':
                    lemma = reemplazar_terminacion(lemma)
                    tokens.append(raiz(tratamiento_texto(lemma)))
                else:
                    tokens.append(tratamiento_texto(lemma))

    tokens = list(dict.fromkeys(tokens))
    tokens = list(filter(None, tokens))
    tokens = revisar_tokens(texto, tokens)
    tokens_str = str(tokens)
    # print('tokens: ',tokens_str)
    return tokens_str

#CARGAR DOCUMENTOS:

#Importando bases de dialogo fluído
txt_folder_path = 'dialogos'
lista_documentos=[x for x in os.listdir(txt_folder_path) if x.endswith(".txt")]
lista_dialogos, lista_dialogos_respuesta, lista_tipo_dialogo = [],[],[]
for idx in range(len(lista_documentos)):
  f=open(txt_folder_path+'/'+lista_documentos[idx], 'r', encoding='utf-8', errors='ignore')
  estado = True
  for line in f.read().split('\n'):
    if estado:
      line_tratado = tratamiento_texto(line)
      lista_dialogos.append(line_tratado)
      lista_tipo_dialogo.append(lista_documentos[idx][:-4])
    else:
      lista_dialogos_respuesta.append(line)
    estado=not estado

#Creando Dataframe de diálogos
datos = {'dialogo':lista_dialogos,'respuesta':lista_dialogos_respuesta,'tipo':lista_tipo_dialogo,'interseccion':0,'jaro_winkler':0,'probabilidad':0}
df_dialogo = pd.DataFrame(datos)
df_dialogo = df_dialogo.drop_duplicates(keep='first')
df_dialogo.reset_index(drop=True, inplace=True)

#Importando bases csv
txt_folder_path = 'documentos'
lista_documentos=[x for x in os.listdir(txt_folder_path) if x.endswith(".csv")]
documento_csv = ''
for i in range(len(lista_documentos)):
  with open(txt_folder_path+'/'+lista_documentos[i], "r", encoding="utf-8") as csv_txt:
    csv_text = csv.reader(csv_txt)
    for fila in csv_text:
      if fila[-1]!='frase':
        documento_csv += fila[-1]

#Importando bases docx
lista_documentos=[x for x in os.listdir(txt_folder_path) if x.endswith(".docx")]
documento_docx = ''
for i in range(len(lista_documentos)):
  for t in Document(txt_folder_path+'/'+lista_documentos[i]).paragraphs:
    documento_docx += t.text.replace('*','\n\n*')+ "\n"
    # documento_docx += t.text.replace('*','\n\n*').replace('-','\n-')
    # documento_docx += documento_docx + "\n"

#Importando bases txt
lista_documentos=[x for x in os.listdir(txt_folder_path) if x.endswith(".txt")]
documento_txt = ''
for i in range(len(lista_documentos)):
  with open(txt_folder_path+'/'+lista_documentos[i], "r", encoding="utf-8") as txt:
    txt_new = txt.read()
    for i in txt_new:
      documento_txt += i
# print('documento_csv: ',documento_csv)
# print('------------------------------------------------------------------------------')
# print('documento_docx: ', documento_docx)
# print('------------------------------------------------------------------------------')
# print('documento_txt: ', documento_txt)

documento = documento_txt+documento_docx+documento_csv
lista_frases = nltk.sent_tokenize(documento,'spanish')
lista_frases_normalizadas = [''.join(str(normalizar(x))) for x in lista_frases]
