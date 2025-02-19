import streamlit as st
from routellm.controller import Controller
import os

st.title("RouteLLM Query Router")

os.environ["LITELLM_LOG"] = "DEBUG"

# Initialize RouteLLM controller
@st.cache_resource
def init_controller():
    return Controller(
        routers=["mf"],
        strong_model="gpt-4o",
        weak_model="gpt-3.5-turbo",
        config={
            "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"}
        }
    )

controller = init_controller()

# Input prompt
prompt = st.text_area("Enter your prompt:", height=100)

if st.button("Send Query"):
    if prompt:
        try:
            response = controller.chat.completions.create(
                model="router-mf-0.11593",
                messages=[{"role": "user", "content": prompt}]
            )
            st.write("Response:")
            st.write(response.choices[0].message.content)
            st.write(f"Selected Model: {response.choices[0].message.metadata.get('selected_model', 'Unknown')}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a prompt.")
