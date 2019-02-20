import tensorflow as tf
import sys
from numpy import array
from numpy import *

import os


import numpy as np
import matplotlib.pyplot as plt


#Variable para la ruta al directorio
path = '/tf_files/starwars/P_3_prueba'
 
#Lista vacia para incluir los ficheros
lstFiles = []
listatumor = []
listasintumor = [1, 2, 3, 4]
 
#Lista con todos los ficheros del directorio:
lstDir = os.walk(path)   #os.walk()Lista directorios y ficheros
 
 
#Crea una lista de los ficheros jpg que existen en el directorio y los incluye a la lista.
 
for root, dirs, files in lstDir:
    for fichero in files:
        (nombreFichero, extension) = os.path.splitext(fichero)
        if(extension == ".jpg"):
            lstFiles.append(nombreFichero+extension)
            #print (nombreFichero+extension)
cs=sorted(lstFiles)
print(cs)  
sorted(lstFiles)
print(lstFiles)            
#print ('LISTADO FINALIZADO')
print "longitud de la lista = ", len(lstFiles)
num=len(lstFiles)
# pruebas con listas 
lista = []

# se crea una variable para guardar los archivos recibidos 

image_path = sys.argv[1]

# sacaramos la imagen del modelo recien guardado 
label_lines = [line.rstrip() for line 
                     in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

  # lo ponemos en la variables graph_def y lo analizamos
with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      _ = tf.import_graph_def(graph_def, name='')


for i in range(len(lstFiles)):
  print i
  image_path = path+"/"+lstFiles[i]
  print image_path
  # leer los archivos de la etiqueta
  image_data = tf.gfile.FastGFile(image_path, 'rb').read()



  # se crea una variable del tentrenar la imagen. 
  with tf.Session() as sess:
    #Alimente image_data como entrada al grafico y obtenga la primera prediccion
    #usamos la funcion softmax en nuestros datos de imagen de entrada y generamo
    #  una matriz de prediccion ejecutando la sesion 

      softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
      predictions = sess.run(softmax_tensor, \
               {'DecodeJpeg/contents:0': image_data})
    
      # Ordenar para mostrar las etiquetas de la primera prediccin en orden de confianza
      top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
      for node_id in top_k:
          human_string = label_lines[node_id]
          score = predictions[0][node_id]
          print('%s (score = %.5f)' % (human_string, score))
          if human_string == "tumor":
           lista.append(score)

print lista


plt.plot(lista)
plt.savefig('Paciente_tres.png')


