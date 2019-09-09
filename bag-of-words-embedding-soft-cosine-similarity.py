#import require libraries
import gensim
from gensim.matutils import softcossim 
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess

#print(gensim.__version__)

# Download the FastText model
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

#Define the documents
rev1 = "Overall the app is very good. I always use it. But after the Android 10 update on the Google pixel 2 xl, I'm not able to chat with other contacts while I'm on video call with picture to picture mode like before. Emailed support and it's like they didn't even try to understand what the problem was and just replied with random links resolutions. Not sure if it was Android update issue or what. Still have the issue."

rev2 = "This would be a much better app if it wasn't such a pain in the arse to update contacts. How hard does it have to be? *Edit*. A year on, and still a pain. Trying to get a contact who is on WhatsApp into the contact list just doesn't happen. You can refresh until you're blye in the face. It still wont add them."

rev3 = "A Feature request: so my whatsapp media is not saved to my phone gallery because most of them are memes , greetings,news but sometimes it's Important photos/videos which i want to save in gallery and there is no option to save this photo to gallery . Please add this feature it would be very helpfull"

rev4 = "The what's new is a lie. It's said the same thing for multiple updates. Why the secretive updates? What are you doing to our devices? // 20 Aug another secretive update with false changelog // 3 Sep *another* update delivered with a FALSE update. This is unaceptable. This is digital rape."

rev5 = "Hii..... WhatsApp My model is motog4 plus ....not show fingerprint lock option why please answer dedo bhai in Hindi????"

rev6 = "We need to have improvements on the quality phone conversations. No smooth flow of conversation, and too many reconnecting instances in one chat, or Tel conversation."

documents = [rev1, rev2, rev3, rev4, rev5, rev6]


#Prepare a dictionary and a corpus.
dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])

#Prepare the similarity matrix
similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0, nonzero_limit=100)

#Embeddings - Convert the sentences into bag-of-words vectors.
sent_1 = dictionary.doc2bow(simple_preprocess(rev1))
sent_2 = dictionary.doc2bow(simple_preprocess(rev2))
sent_3 = dictionary.doc2bow(simple_preprocess(rev3))
sent_4 = dictionary.doc2bow(simple_preprocess(rev4))
sent_5 = dictionary.doc2bow(simple_preprocess(rev5))
sent_6 = dictionary.doc2bow(simple_preprocess(rev6))

sentences = [sent_1, sent_2, sent_3, sent_4, sent_5, sent_6]


#Compute soft cosine similarity of 2 documents
#print(softcossim(sent_1, sent_2, similarity_matrix))

#Compute soft cosine similarity matrix
import numpy as np
import pandas as pd

def soft_cosine_similarity_matrix(sentences):
    len_array = np.arange(len(sentences))
    xx, yy = np.meshgrid(len_array, len_array)
    cossim_mat = pd.DataFrame([[round(softcossim(sentences[i],sentences[j], similarity_matrix) ,2) for i, j in zip(x,y)] for y, x in zip(xx, yy)])
    return cossim_mat

#print(documents)
soft_cosine_similarity_matrix(sentences)

#        0	      1	      2	      3	      4	      5
#   0	1.00	0.46	0.41	0.87	0.60	0.82
#   1	0.46	1.00	0.59	0.40	0.41	0.24
#   2	0.41	0.59	1.00	0.35	0.50	0.29
#   3	0.87	0.40	0.35	1.00	0.52	0.71
#   4	0.60	0.41	0.50	0.52	1.00	0.58
#   5	0.82	0.24	0.29	0.71	0.58	1.00
