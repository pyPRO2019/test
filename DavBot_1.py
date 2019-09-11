import pandas as pd
import nltk
import random
import numpy as np
import string


## Reading the data set 
f=open('chatbot.txt','r',errors='ignore')
raw=f.read()

## Convert the text to lower case
raw=raw.lower()

nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only

sent_token=nltk.sent_tokenize(raw)## converting to a a list of sentences
word_token=nltk.word_tokenize(raw) ### Convering to a list of words


lemmer=nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
                     
                     
                     
         ## Key Words Matching
greeting_input=("hello", "hi", "greetings", "sup", "what's up","hey ")
greeting_responses=["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_input:
            return random.choice(greeting_responses)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response (user_response):
    robo_response=''
    sent_token.append(user_response)
    
    TfidfVec= TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf=TfidfVec.fit_transform(sent_token)
    vals=cosine_similarity(tfidf[-1],tfidf)
    idx=vals.argsort()[0][-2]
    flat= vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if (req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you. Please try using other words"
        return robo_response
    else:
        robo_response=robo_response+ sent_token[idx]
        return robo_response
    
    
    
flag=True
print("DavBot : Helloo!  My name is DavBot. I will answer your queries about Chatbots. If you want to exit, just type Bye! ")

while (flag==True):
    user_response=input()
    user_response=user_response.lower()
    if (user_response !='bye'):
        if (user_response=='thanks' or user_response=='thank you'):
            flag=False
            print ('DavBot: You are welcome.     :)')
        else:
            if (greeting(user_response)!=None):
                print ("DavBot: " + greeting(user_response))
            else:
                print("DavBot: ",end='')
                print(response(user_response))
                sent_token.remove(user_response)
    else:
        flag=False
        print("DavBot:  Bye! It was nice chating with you")
        print("DavBot:  Take Care...")
          