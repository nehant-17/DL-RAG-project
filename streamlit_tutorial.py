import streamlit as st 
import pandas as pd 
import numpy as np

st.title("Hello, Streamlit!")
st.write(":streamlit: This is your first streamlit app")
st.text("Lets get started")


name = st.text_input("Enter your name : ")
if st.button("Greet"):
    st.success(f"Hello, {name}")


#Displaying Data and Charts 
df = pd.DataFrame(np.random.randn(10,2), columns=['A','B'])
st.line_chart(df)
st.bar_chart(df)

#File uploading and caching topics 
upload_file = st.file_uploader("upload a csv", type='csv')
if upload_file:
    df = pd.read_csv(upload_file)
    st.dataframe(df)


st.title("text and Markdown Demo")
st.header("This is a header")
st.subheader("This is a subheader")
st.markdown("**Bold**, *Italic*, 'Code', [Link](https://streamlit.io)")
st.code("for i in range(5): print(i)", language="python")    