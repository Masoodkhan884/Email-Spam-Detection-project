import streamlit as st
import pickle
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')


st.set_page_config(
    page_title="Email Detection",
    page_icon="spam.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

ps=PorterStemmer()

# side bar
from PIL import Image
# Sidebar content
with st.sidebar:
    # Logo insertion
    image = Image.open('spam.png')
    st.image(image,width=200, caption=None,use_column_width=False)

    # Personal information
    st.title("Made By: \n MASOOD KHAN")

    # Contact Us button (for email)
    st.markdown(
        '<a href="mailto:masoodkhanse884@gmail.com" style="text-decoration:none;"><button style="background-color:#4CAF50; color:white; padding:10px 20px; border:none; border-radius:5px;">Contact Us</button></a>',
        unsafe_allow_html=True
    )
    # GitHub Button with link opening in a new tab
    st.title("Go To my GitHub")
    st.markdown(
        '<a href="https://github.com/Masoodkhan884" target="_blank" style="text-decoration:none;"><button style="background-color:#4CAF50; color:white; padding:10px 20px; border:none; border-radius:5px;">GitHub</button></a>',
        unsafe_allow_html=True
    )




# Injecting custom CSS for the sidebar
CSS_STYLE=f"""
    <style>
    /* Style the sidebar */
    .st-emotion-cache-6qob1r{{
        position: relative;
        height: 100%;
        width: 100%;
        overflow: overlay;
        BACKGROUND: #000000;
    }}

    .st-emotion-cache-h4xjwg {{
    display: none;
}}
    
.st-emotion-cache-1jicfl2 {{
    width: 100%;
    padding: 3rem 4rem 4rem;
    min-width: auto;
    max-width: initial;
}}


.st-emotion-cache-qcpnpn {{
    border: 1px solid rgba(250, 250, 250, 0.2);
    border-radius: 0.5rem;
    padding: calc(-1px + 1rem);

    /* Add box shadow */
     box-shadow: 2px 4px 3px rgb(239 232 232 / 39%);
}}




    </style>
    """



st.markdown(CSS_STYLE,unsafe_allow_html=True)




def Transform_text(text):
    text=text.lower()
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:] # to copy string 
    y.clear() # clear y
    for i in text:
          #Remove stop words and Punctuation from the sentence 
        if i not in stopwords.words('english') and i not in string.punctuation: 
            y.append(i)

    text=y[:] # to copy string 
    y.clear() # clear y
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y) # To join all the words which is return
# Loading of pickle data   
Tfidf = pickle.load(open('Vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Title of the app

st.markdown(
            """
            <h1 style='text-align: center;'>Emails/SMS Spam Classifier</h1>
            """,
            unsafe_allow_html=True
            )

# st.markdown(
#                 """
#                 <hr style="border: none; height: 2px; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); box-shadow: 10px 4px 8px rgba(3, 225, 129, 0.2);" />
#                 """,
#                 unsafe_allow_html=True
#             )   


st.markdown(
        """
                <hr style="border: none; height: 1px;width: 60%; background: linear-gradient(90deg, rgba(216,82,82,1) 13%, rgba(237,242,6,1) 57%, rgba(226,0,255,1) 93%); margin: 0 auto;" />
                """,
                unsafe_allow_html=True)
                




box=st.form("input")

# input the text
input_Mail = box.text_input("Enter your Mails ")

submit=box.form_submit_button("Predict")

# button for performing prediction
if submit:
    # preprocessing
    Transform_Mails=Transform_text(input_Mail)
    # Vectorizing the text data
    Transform_Mails=Tfidf.transform([Transform_Mails])
    # prediction
    prediction=model.predict(Transform_Mails)
    # display

    if prediction==1:
        st.error('Spam')
    else:
        st.success('Not Spam')
    


st.markdown(
    """
    <style>
.st-emotion-cache-15hul6a {
    display: inline-flex;
    -webkit-box-align: center;
    align-items: center;
    -webkit-box-pack: center;
    justify-content: center;
    font-weight: 400;
    padding: 0.25rem 0.75rem;
    border-radius: 0.5rem;
    min-height: 2.5rem;
    margin: 0px;
    line-height: 1.6;
    color: inherit;
    width: auto;
    user-select: none;
    background-color: rgb(32 168 34);
    border: 1px solid rgba(250, 250, 250, 0.2);
}
    </style>
    """,
    unsafe_allow_html=True
)


