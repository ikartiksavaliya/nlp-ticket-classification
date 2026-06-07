"""
Streamlit Web Application for Customer Support Ticket Classification.
Integrates the TicketInferencePipeline to classify user-submitted ticket texts.
Provides real-time visualization of predictions, confidence levels, and OOV word tracking.
"""

import streamlit as st

def main():
    """
    Main function to construct and execute the Streamlit application.
    Displays:
    - Sidebar configuration for model selection.
    - Main input text area for the raw ticket description.
    - Classification triggers and status notifications.
    - Visual indicators (bar charts) of class probabilities.
    - Preprocessing walkthrough showing how the raw text was normalized.
    - OOV (Out-of-vocabulary) word diagnostics for transparency.
    """
    st.set_page_config(
        page_title="Ticket Classification System",
        page_icon="🎫",
        layout="centered"
    )
    
    st.title("🎫 Customer Support Ticket Classifier")
    st.markdown(
        """
        This application classifies incoming customer support tickets into product/issue categories
        using a production-grade traditional NLP and Machine Learning pipeline.
        """
    )
    
    # 1. Sidebar - Load Artifacts information
    st.sidebar.header("Pipeline Configuration")
    
    # 2. Main Input Section
    st.subheader("Enter Ticket Details")
    ticket_text = st.text_area(
        "Paste the raw support ticket text here:",
        height=150,
        placeholder="Example: My account was suspended but I was charged for the premium subscription last night..."
    )
    
    # 3. Predict Button and Pipeline Outputs
    if st.button("Classify Ticket"):
        if ticket_text.strip() == "":
            st.warning("Please enter some ticket text to classify.")
        else:
            with st.spinner("Processing ticket..."):
                # TODO: Initialize TicketInferencePipeline
                # TODO: Run pipeline.predict_single(ticket_text)
                # TODO: Display predictions, confidence scores, probability charts, and OOV stats
                pass

if __name__ == "__main__":
    main()
