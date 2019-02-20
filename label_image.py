import tensorflow as tf
import sys
from numpy import array
from numpy import *

# se crea una variable para guardar los archivos recibidos 

image_path = sys.argv[1]

# leer el archivo 
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# sacaramos la imagen del modelo recien guardado 
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

# lo ponemos en la variables graph_def y lo analizamos
with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
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
