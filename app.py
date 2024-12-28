import requests


def get_external_ip():
    response = requests.get("https://api64.ipify.org?format=json")
    if response.status_code == 200:
        data = response.json()
        return data.get("ip")
    else:
        return "Unknown"


external_ip = get_external_ip()
import streamlit as st

st.write("External IP:", external_ip)

from src.streamlit_app import StreamlitApp


def main():
    """Initialize and run the Streamlit application."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
