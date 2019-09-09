#Load required libraries
import tensorflow as tf
import tensorflow_hub as hub
from sklearn import preprocessing
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
matplotlib.style.use('ggplot')

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

#def elmo_vectors(x):
embeddings = elmo(["i am very very happy", "This is pretty disappoinitng"], signature="default", as_dict=True)["elmo"]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    x=sess.run(tf.reduce_mean(embeddings,1))

#print the embedding vectors for input text
#print(x)

#compute cosine similarity
sim = cosine_similarity(x)
print(sim)


#text_similarity_diagramm
#output of your first part
cosine = [sim[0]]

#set constants
r = 1
d = 2 * r * (1 - cosine[0][1])

#draw circles
circle1=plt.Circle((0, 0), r, alpha=.5)
circle2=plt.Circle((d, 0), r, alpha=.5)

#set axis limits
plt.ylim([-1.1, 1.1])
plt.xlim([-1.1, 1.1 + d])
fig = plt.gcf()
fig.gca().add_artist(circle1)
fig.gca().add_artist(circle2)
fig.savefig('text_similarity_diagramm.png')
