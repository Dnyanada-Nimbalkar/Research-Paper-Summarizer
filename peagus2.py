import streamlit as st
import os
import PyPDF2
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from io import StringIO
from rouge_score import rouge_scorer

# Add texts to add in the web-app
url_pegasus = 'https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html'
st.markdown("# PDF summarization using Pegasus")
st.markdown(
    '''PhD scholars tackle a challenging task: summarizing scientific papers. This requires deep understanding and rewriting skills, known as abstractive summarization, a tough nut in NLP. Thanks to seq2seq learning and models like ELMo and BERT, this task became more manageable.
    Google’s [PEGASUS](%s) further improved the state-of-the-art (SOTA) 
results for abstractive summarization, in particular with low resources. To be more specific, unlike previous models, PEGASUS enables us to achieve
 close to SOTA results with 1,000 examples, rather than tens of thousands of training data. The model uses Transformers Encoder-Decoder architecture. 
 The encoder outputs masked tokens while the decoder 
  generates Gap sentences. ''' % url_pegasus)

st.header('''Upload any PDF (less than 200MB) using the Browse file button to get its abstractive summary within minutes. 
          Note: Upload one PDF at a time. The speed of processing depends on the PDF's text length''')

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload a file")

if uploaded_file:
    # If file uploaded successfully, print "Saved file"
    with open(os.path.join(uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Saved File")

st.markdown("# Summary")

# Load tokenizer
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

# Load model
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

def summarize_text(text):
    # Split text into smaller chunks
    max_length = 1024
    sentences = text.split('. ')
    current_chunk = ""
    chunks = []
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Summarize each chunk and combine summaries
    summaries = []
    for chunk in chunks:
        tokens = tokenizer(chunk, truncation=True, padding="longest", return_tensors="pt")
        summary = model.generate(**tokens)
        decoded_output = tokenizer.decode(summary[0], skip_special_tokens=True)
        summaries.append(decoded_output)

    return " ".join(summaries)

def compute_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

if uploaded_file:
    # Load the PDF reader object
    fhandle = open(uploaded_file.name, 'rb')
    pdfReader = PyPDF2.PdfReader(fhandle)

    # Add the number of page filter on the app
    values = st.slider(
        'Summarize page number between:',
        min_value=1, max_value=len(pdfReader.pages),
        value=(1, len(pdfReader.pages)),
        step=1
    )

    # Create a status loader
    with st.spinner('Preparing summary. Please wait...'):
        page_wise_text = []
        for x in range(values[0] - 1, values[1]):
            # Extract text from PDF
            page = pdfReader.pages[x]
            text = page.extract_text()

            if text:
                # Summarize the text
                detailed_summary = summarize_text(text)

                # Update summary on the list
                page_wise_text.append((text, detailed_summary))

                # Print the summary
                st.markdown("### Page number: {}".format(x + 1))
                st.text_area(label="", value=detailed_summary, height=200, key=x + 1)
                st.markdown("")
            else:
                st.warning(f"Page {x + 1} contains no text.")

        # Compute and display ROUGE scores
        st.markdown("# ROUGE Scores")
        for idx, (reference, summary) in enumerate(page_wise_text):
            rouge_scores = compute_rouge(reference, summary)
            st.markdown(f"### Page {idx + 1}")
            st.text_area(label="ROUGE-1", value=str(rouge_scores['rouge1']), height=100, key=f"rouge1_{idx}")
            st.text_area(label="ROUGE-2", value=str(rouge_scores['rouge2']), height=100, key=f"rouge2_{idx}")
            st.text_area(label="ROUGE-L", value=str(rouge_scores['rougeL']), height=100, key=f"rougeL_{idx}")
