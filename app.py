# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:10:57 2024

@author: Sai
"""
import streamlit as st

def chatbot(qn):
    
    import string
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from nltk.corpus import stopwords
    import nltk
    from nltk.stem import WordNetLemmatizer
    import random
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")
    
    nltk.download('popular', quiet=True)  # Downloads packages
    f = open('GIM.txt', 'r', errors='ignore')
    raw = f.read()
    raw = raw.lower()
    
    sent_tokens = nltk.sent_tokenize(raw)
    lemmer = WordNetLemmatizer()
    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]
    
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    
    def LemNormalize(text):
        tokens = nltk.word_tokenize(text.lower().translate(remove_punct_dict))
        tokens = [lemmer.lemmatize(token) for token in tokens if token not in ENGLISH_STOP_WORDS]
        return ' '.join(tokens)
    
    def response(user_response):
        robot_response = ''
        sent_tokens.append(user_response)
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sent_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]
        if req_tfidf == 0:
            robot_response = robot_response + "I think I need to read more about that..."
            return robot_response
        else:
            robot_response = robot_response + sent_tokens[idx]
            return robot_response
    
    GREETING_INPUTS = ("namastey", "namaskaram", "hello", "hi", "whats up", "hey")
    GREETING_RESPONSES = ["namastey", "namaskaram", "hello", "hi", "whats up", "hey"]
    
    def greeting(sentence):
        for word in sentence.split():
            if word.lower() in GREETING_INPUTS:
                return random.choice(GREETING_RESPONSES)
    
    user_response = qn
    user_response=user_response.lower()
    res = response(user_response)
    
    return res
    

from streamlit_chat import message

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))

def main():

    import streamlit as st
    import time
    
    st.title("GIM AdmissionAssistant")
    intro_message = (
        "Hello there! How can I assist you with your GIM admission journey? "
        "Feel free to ask any questions you may have about the admission process, requirements, or any other related queries."
    )
    
    # Display the introduction message
    st.markdown(intro_message)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("How may I help you today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = chatbot(prompt)
            
            # Simulate stream of response with milliseconds delay
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})    
    


if __name__ == "__main__":
    main()
