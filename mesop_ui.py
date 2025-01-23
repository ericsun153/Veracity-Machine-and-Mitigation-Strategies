import os
import time
import io
import mesop as me
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import content_types
from collections.abc import Iterable
import chromadb
import serpapi
import PyPDF2
import json
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score, classification_report
from prediction_engine import PredictionEngine
from utils import *

############################################ global settings ############################################
# Remember to set your API key here
os.environ['GOOGLE_API_KEY'] = 'AIzaSyArMll8tFUpS2tHcPr6di-hb7jWnZubU80'

# Initialize API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("my_collection", metadata={"hnsw:space": "cosine"})

# Instantiate the GenAI model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
obj_funcs = [repetition_analysis, origin_tracing, evidence_verification, omission_checks, exaggeration_analysis, target_audience_assessment]
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",
    generation_config=generation_config,
    tools=obj_funcs,
)

def tool_config_from_mode(mode: str, fns: Iterable[str] = ()):
    """Create a tool config with the specified function calling mode."""
    return content_types.to_tool_config(
        {"function_calling_config": {"mode": mode, "allowed_function_names": fns}}
    )

chat_session = model.start_chat(history=[])
tool_config = tool_config_from_mode("auto")

# Global variables
_prediction_engine = None
_related_search_results = None
############################################ mesop app ############################################
@me.stateclass
class State:
    input: str = ""
    output: str = ""
    in_progress: bool = False
    chat_window: str = ""
    search_output: str = ""
    db_input: str = ""
    db_output: str = ""
    file: me.UploadedFile | None = None
    news_text: str = ""
    initialized: bool = False
    in_building: bool = False
    training_status: str = ""
    factuality_factors: list[str]
    use_pred_model: bool = True
    use_gen_model: bool = True
    use_search_online: bool = True
    use_function_calling: bool = True
    prompt_type: str = "FCoT"

# Event handler for checkboxes
def on_change_checkbox(e: me.CheckboxChangeEvent):
    state = me.state(State)
    if e.key == "pred_model":
        state.use_pred_model = e.checked
        print("Predictive Model: " + str(state.use_pred_model))
    elif e.key == "gen_model":
        state.use_gen_model = e.checked
        print("Generative Model: " + str(state.use_gen_model))
    elif e.key == "search_online":
        state.use_search_online = e.checked
        print("Search Online: " + str(state.use_search_online))
    elif e.key == "function_calling":
        state.use_function_calling = e.checked
        print("Function Calling: " + str(state.use_function_calling))

# Event handler for the button toggle
def on_change_button_toggle(e: me.ButtonToggleChangeEvent):
    state = me.state(State)
    state.prompt_type = e.values
    print("Current prompting method: " + str(state.prompt_type))

# Event handler for the build machine button
def on_click_build_machine(e: me.ClickEvent):
    state = me.state(State)
    if e.key == "build_machine":
        state.in_building = True
        print("Building...")
        yield
        
        # Initialize the PredictionEngine
        yield from train_predictive()

        # Complete the build
        state.in_building = False
        state.initialized = True
        print("Build complete.")
        yield

# Master page
@me.page(path="/")
def page():
    with me.box(
        style=me.Style(
            background="#fff",
            min_height="calc(100% - 48px)",
            padding=me.Padding(bottom=16),
        )
    ):
        with me.box(
            style=me.Style(
                width="min(1080px, 100%)",
                margin=me.Margin.symmetric(horizontal="auto"),
                padding=me.Padding.symmetric(horizontal=16),
            )
        ):
            header_text()
            toggles()
            build_machine_button()
            
            if me.state(State).initialized:
                me.divider()
                news_input()
                display_news_article()
                chat_window()
                search_output()
                chat_input()
                output()
            footer()

############################################ begin page elements ############################################
def header_text():
    with me.box(style=me.Style(padding=me.Padding(top=64, bottom=36))):
        me.text(
            "Veracity Machine",
            style=me.Style(
                font_size=36,
                font_weight=700,
                background="linear-gradient(90deg, #4285F4, #AA5CDB, #DB4437) text",
                color="transparent"
            )
        )
    with me.box(style=me.Style(padding=me.Padding(bottom=16))):
        me.text(
            "Welcome to the Veracity Machine, a tool for analyzing the veracity of news articles.",
            style=me.Style(font_size=16)
        )

# Toggles for selecting the features to include in the analysis
def toggles():
    state = me.state(State)
    with me.box(style=me.Style(padding=me.Padding(bottom=16), margin=me.Margin(top=32))):
        me.text("Customize the features you would like to include in your machine. Recommend default options.")

    with me.box(style=me.Style(padding=me.Padding(bottom=16))):
        me.text("1. Choose the models to be used:")
        me.checkbox(
            "Predictive Model (Random Forest Classifier)", 
            checked=state.use_pred_model, 
            on_change=on_change_checkbox,
            key='pred_model',
            style=me.Style(margin=me.Margin(right=64))
        )
        me.checkbox(
            "Generative Model (gemini-1.5-pro-002)", 
            checked=state.use_gen_model, 
            on_change=on_change_checkbox,
            key='gen_model',
        )

    with me.box(style=me.Style(padding=me.Padding(bottom=16))):
        me.text("2. Choose the prompting method:")
        me.button_toggle(
            value=state.prompt_type,
            buttons=[
            me.ButtonToggleButton(label="Normal Prompting", value="normal"),
            me.ButtonToggleButton(label="CoT Prompting", value="CoT"),
            me.ButtonToggleButton(label="FCoT Prompting", value="FCoT"),
            ],
            multiple=False, # only one button can be selected at a time
            hide_selection_indicator=False,
            on_change=on_change_button_toggle,
            style=me.Style(margin=me.Margin(top=16, bottom=16)),
        )

    with me.box(style=me.Style(padding=me.Padding(bottom=16))):
        me.text("3. Factuality factors (don't have toggle capability in PredAI):")
        state.factuality_factors = ["Echo Chamber", "Education", "Event Coverage", "Frequency Heuristic", "Location", "Malicious Account", "Misleading Intention", "News Coverage"]
        for ff in state.factuality_factors:
            me.checkbox(
                ff,
                checked=True,
                disabled=True, # disabled until predAI can handle this
                style=me.Style(margin=me.Margin(right=64))
            )

    with me.box(style=me.Style(padding=me.Padding(bottom=16))):
        me.text("4. Choose additional functions:")
        me.checkbox(
            "Search Online",
            checked=state.use_search_online,
            on_change=on_change_checkbox,
            key='search_online',
            style=me.Style(margin=me.Margin(right=64))
        )
        me.checkbox(
            "Function Calling",
            checked=state.use_function_calling,
            on_change=on_change_checkbox,
            key='function_calling',
        )

def build_machine_button():
    state = me.state(State)
    with me.box(style=me.Style(padding=me.Padding(top=16), display="flex", justify_content="center")):
        me.button(
            "Build" if not state.in_building else "Building...",
            type="flat",
            disabled=state.in_building,
            on_click=on_click_build_machine,
            key='build_machine',
            style=me.Style(width=256)
        )

    with me.box(style=me.Style(padding=me.Padding(top=16, bottom=16), display="flex", justify_content="center")):
        me.text(state.training_status)

# to train the predictive model
def train_predictive():
    global _prediction_engine
    state = me.state(State)
    try:
        state.training_status = "Initializing prediction engine..."
        yield
        
        # Initialize prediction engine
        _prediction_engine = PredictionEngine()
        state.training_status = "Loading dataset and preparing models..."
        yield
        
        # Start the training process
        _prediction_engine.load_dataset_and_prepare_models()
        state.training_status = "Training complete!"
        yield
    except Exception as e:
        state.training_error = f"Training failed: {str(e)}"

############################################ begin model elements ############################################
# ------------------------------------------ Choose News article ------------------------------------------
# Options for inputting the news article
def news_input():
    state = me.state(State)
    with me.box(style=me.Style(padding=me.Padding(top=16), display="flex", justify_content="center")):
        me.text("Input the news article you would like to analyze:", style=me.Style(font_weight="bold"))
    with me.box(style=me.Style(padding=me.Padding(top=16), display="flex", justify_content="center", align_items="center")):
        # URL input method
        with me.box(style=me.Style(padding=me.Padding.all(15))):
            me.input(
                label="URL for News Article",
                placeholder="Paste the URL here and press Enter",
                type="url",
                on_enter=handle_url,
                style=me.Style(width=400)
            )

        me.text("or", style=me.Style(font_weight="bold"))

        # Upload method from PDF
        with me.box(style=me.Style(padding=me.Padding.all(15))):
            me.uploader(
                label="Upload PDF",
                accepted_file_types=["application/pdf"],
                on_upload=handle_upload,
                type="flat",
                color="primary",
                style=me.Style(font_weight="bold"),
            )

# Event handler for URL input
def handle_url(e: me.InputEnterEvent):
    state = me.state(State)
    # Helper function for checking validity of URL
    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    # Helper function for scraping news text from URL
    def scrape_news_text(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        news_text = '\n'.join([para.get_text() for para in paragraphs])
        # Extract title
        title = soup.find('h1').get_text() if soup.find('h1') else "No title found"
        # Extract authors
        author_text = ""
        content = response.content.decode('utf-8')
        by_index = content.find("By ")
        if by_index != -1:
            author_text = content[by_index:by_index + 100]
            
        return f"Title: {title}\n\nAuthors: {author_text}\n\n{news_text}"

    if is_valid_url(e.value):
        try:
            state.news_text = scrape_news_text(e.value)
        except Exception as ex:
            state.news_text = f"Failed to scrape the website: {str(ex)}"
    else:
        state.news_text = "Not valid URL"

# Event handler for file upload
def handle_upload(e: me.UploadEvent):
    state = me.state(State)
    state.file = e.file
    if state.file and state.file.mime_type == 'application/pdf':
        # Extract text from the PDF after upload
        extract_text_from_pdf(state.file)

# Extract text from the uploaded PDF file
def extract_text_from_pdf(file: me.UploadedFile):
    """Extracts text from the uploaded PDF file and stores it in the state."""
    state = me.state(State)
    # Wrap the bytes content in a BytesIO object
    pdf_file = io.BytesIO(file.getvalue())
    # Initialize the PDF reader
    pdf_reader = PyPDF2.PdfReader(pdf_file)  
    extracted_text = ""
    # Extract text from each page
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        extracted_text += page.extract_text()
    state.news_text = extracted_text  # Store extracted PDF text in state

# Display the news input
def display_news_article():
    state = me.state(State)
    if state.news_text:
        with me.box(style=me.Style(padding=me.Padding.all(10), border=me.Border.all(me.BorderSide(width=3, color="black", style="double")), border_radius=10, overflow="auto", max_height=400)):
            me.text("News Article:")
            me.text(state.news_text, type="body-1")  # Display the entire text


# ------------------------------------------ Search Functionality -------------------------------------------
# Input window for user questions
def chat_window():
    state = me.state(State)
    with me.box(style=me.Style(padding=me.Padding(top=32))):
        me.text("Search Online for Related Information:", style=me.Style(font_weight="bold"))
    with me.box(style=me.Style(padding=me.Padding.all(8), background="white", display="flex", width="100%", border=me.Border.all(me.BorderSide(width=0, style="solid", color="black")), border_radius=12, box_shadow="0 10px 20px #0000000a, 0 2px 6px #0000000a, 0 0 1px #0000000a", margin=me.Margin(top=16))):
        with me.box(style=me.Style(flex_grow=1)):
            me.native_textarea(
                value=state.chat_window,
                autosize=True,
                min_rows=4,
                placeholder="Enter your prompt or query. Search results will be automatically included.",
                style=me.Style(
                    padding=me.Padding(top=16, left=16),
                    background="white",
                    outline="none",
                    width="100%",
                    overflow_y="auto",
                    border=me.Border.all(
                        me.BorderSide(style="none"),
                    ),
                ),
                on_blur=search_textarea_on_blur,
            )
        with me.content_button(type="icon", on_click=click_chat_send):
            me.icon("send")

# Retains the user's input in the chat window
def search_textarea_on_blur(e: me.InputBlurEvent):
    state = me.state(State)
    state.chat_window = e.value

# Initiate the search process 
def click_chat_send(e: me.ClickEvent):
    global _related_search_results
    state = me.state(State)
    if not state.chat_window.strip():
        return
    
    state.in_progress = True  # Start spinner for progress indication
    user_prompt = state.chat_window
    state.search_output = "Searching for results...\n"
    yield

    # Conduct the web search
    search_results = web_search(user_prompt)
    
    # Format the search results for display
    if search_results:
        _related_search_results = f"Search Results for '{user_prompt}':\n\n" + format_search_results_pretty(search_results)
        state.search_output = _related_search_results
    else:
        state.search_output = "No results found for your query."

    state.in_progress = False
    yield

# Function to recursively make objects serializable
def extract_serializable_data(obj):
    """Recursively convert non-serializable objects to serializable types."""
    if isinstance(obj, dict):
        # Convert non-serializable keys to strings
        return {str(key): extract_serializable_data(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [extract_serializable_data(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Convert objects with __dict__ attribute to dictionaries
        return extract_serializable_data(vars(obj))
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj  # Leave primitive types as-is
    else:
        return str(obj)  # Convert unsupported types to strings

def format_search_results_pretty(search_results):
    """Formats search results as a pretty JSON string."""
    try:
        if hasattr(search_results, "raw_response"):
            search_dict = search_results.raw_response  # Raw response if available
        else:
            search_dict = extract_serializable_data(vars(search_results))  # Fallback: Use vars()

        # Pretty-print the JSON string
        pretty_json = json.dumps(search_dict, indent=4, ensure_ascii=False)
        return pretty_json

    except Exception as e:
        print(f"Error serializing search results: {e}")

def web_search(user_prompt):
    # Conduct a Google Custom Search query
    params = {
        "q": user_prompt,
        "location": "San Diego, California, United States",
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": "749dd5738c6934b68bb86b199202f6c1c9328db2638a4abb74502695b1bda013"
        }
    search = {}
    # search = serpapi.search(params) # Uncomment this line to enable the search
    return search

def search_output():
    state = me.state(State)
    if state.search_output:
        with me.box(
            style=me.Style(
                background="#F0F4F9",
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin(top=36),
            )
        ):
            me.text(state.search_output)

# -------------------------------------------------------- Prompting ------------------------------------------------------------
# User input for GenAI prompting
def chat_input():
    state = me.state(State)
    with me.box(
        style=me.Style(
            padding=me.Padding.all(8),
            background="white",
            display="flex",
            width="100%",
            border=me.Border.all(
                me.BorderSide(width=0, style="solid", color="black")
            ),
            border_radius=12,
            box_shadow="0 10px 20px #0000000a, 0 2px 6px #0000000a, 0 0 1px #0000000a",
        )
    ):
        with me.box(
            style=me.Style(
                flex_grow=1,
            )
        ):
            me.native_textarea(
                value=state.input,
                autosize=True,
                min_rows=4,
                placeholder="Enter your customized prompt, or will do default FCOT if empty.",
                style=me.Style(
                    padding=me.Padding(top=16, left=16),
                    background="white",
                    outline="none",
                    width="100%",
                    overflow_y="auto",
                    border=me.Border.all(
                        me.BorderSide(style="none"),
                    ),
                ),
                on_blur=textarea_on_blur,
            )
        with me.content_button(type="icon", on_click=click_send):
            me.icon("send")

def textarea_on_blur(e: me.InputBlurEvent):
    state = me.state(State)
    state.input = e.value

def click_send(e: me.ClickEvent):
    global _prediction_engine
    global _related_search_results
    state = me.state(State)
    # Kickstart veracity analysis workflow
    if not state.input.strip():  # Check if input is empty or contains only whitespace
        state.input = "Default input text here."  # Default FCT prompt if no input is provided
        return
    
    state.in_progress = True
    input_text = state.input

    '''TODO: Potential implementation of chunking here (By statement)'''

    predict_score = _prediction_engine.predict_new_example(convert_statement_to_series(state.news_text))['overall']

    # top_100_statements = get_top_100_statements(input_text)
    fct_prompt = generate_fct_prompt(state.news_text, predict_score)
    combined_input = combine_pdf_and_prompt(fct_prompt, state.news_text)  # Combine prompt with PDF text
    top_100_statements = get_top_100_statements(combined_input)

    for chunk in call_api(combined_input):
        state.output += chunk
        yield

    state.output += '\n\n The probability of the statement truthness:\n\n' + str(top_100_statements) + '\n\n'

    state.in_progress = False
    yield

def convert_statement_to_series(statement):
    '''
    TODO: Will need to extract speaker info later on
    '''
    if not isinstance(statement, str):
        return pd.Series(['','', '', '', '', '', '','','0.0','0.0','0.0','0.0','0.0', '', ''])
    subject = ''
    speaker = ''
    speaker_title = ''
    state = ''
    party_aff = ''
    context = ''
    return pd.Series(['','',subject, statement, speaker, speaker_title, state, party_aff,'0.0','0.0','0.0','0.0','0.0', context, ''])

# Fractal COT & Function Call
# Define the complex objective functions
frequency_heuristic = [
    {"description": "Micro Factor 1: Repetition Analysis", "details": "Analyzing wider coverage helps assess consensus. If multiple independent sources confirm manipulation, it strengthens the claim. Even widespread agreement about deceptive editing wouldn't automatically justify government action against CBS. The First Amendment protects against content-based restrictions."},
    {"description": "Micro Factor 2: Origin Tracing", "details": "Confirmed sources are critical. Discrepancies between reporting and original sources raise red flags."},
    {"description": "Micro Factor 3: Evidence Verification", "details": "Expert analysis is the most crucial element. Expert opinions on video manipulation are essential. Expert testimony would be necessary for any legal action alleging manipulation, though such a case would face significant First Amendment hurdles."}
]

misleading_intentions = [
    {"description": "Micro Factor 1: Omission Checks", "details": "Assess the omissions' impact. Did they distort the message? Did they create a demonstrably false representation? (Proving this is difficult)."},
    {"description": "Micro Factor 2: Exaggeration Analysis", "details": "Evaluate the 'scandal' claim. Does the evidence support it, or is it hyperbole? Does the situation, even if accurately reported, justify calls for license revocation under existing legal and constitutional frameworks?"},
    {"description": "Micro Factor 3: Target Audience Assessment", "details": "Analyze audience manipulation. Identify targeting tactics (language, framing). While such tactics can be ethically questionable, they are generally protected speech unless they involve provable falsehoods and meet the very high legal bar for defamation or incitement."}
]

def generate_fct_prompt(input_text, predict_score, iterations=3, regular_CoT=False):
    # Only use regular CoT prompting for testing
    if regular_CoT:
        ffs = ['Frequency Heuristic', 'Misleading Intentions']
        prompt = f'Use {iterations} iterations to check the veracity score of this news article. Factors to consider are {ffs}. In each, determine what you missed in the previous iteration. Also put the result from RAG into consideration/rerank.'
        prompt += f'\n\n RAG:\n Here, out of six potential labels (true, mostly-true, half-true, barely-true, false, pants-fire), this is the truthfulness label predicted using a classifier model: {predict_score}.\n These are the top 100 related statement in LiarPLUS dataset that related to this news article: {get_top_100_statements(input_text)}.'
        prompt += "\nProvide a percentage score and explanation for each iteration and its microfactors.\n\n"
        prompt += "Final Evaluation: Return an exact numeric veracity score for the text, and provide a matching label out of these six [true, mostly-true, half-true, barely-true, false, pants-fire]"
        return prompt
    
    # Fractal Chain of Thought Prompting with Objective Functions
    prompt = f'Use {iterations} iterations to check the veracity score of this news article. In each, determine what you missed in the previous iteration based on your evaluation of the objective functions. Also put the result from RAG into consideration/rerank.'
    prompt += f'\n\n RAG:\n Here, out of six potential labels (true, mostly-true, half-true, barely-true, false, pants-fire), this is the truthfulness label predicted using a classifier model: {predict_score}.\n These are the top 100 related statement in LiarPLUS dataset that related to this news article: {get_top_100_statements(input_text)}.'
    prompt += f'Here is some additional information that has been searched from internet: {_related_search_results}.'
    for i in range(1, iterations + 1):
        prompt += f"Iteration {i}: Evaluate the text based on the following objectives and also on microfactors, give explanations on each of these:\n"
        prompt += "\nFactuality Factor 1: Frequency Heuristic:\n"
        for fh in frequency_heuristic:
            prompt += f"{fh['description']}: {fh['details']}\n"
        prompt += "\nFactuality Factor 2: Misleading Intentions:\n"
        for mi in misleading_intentions:
            prompt += f"{mi['description']}: {mi['details']}\n"
        prompt += "\nProvide a percentage score and explanation for each objective function ranking and its microfactors.\n\n"
    prompt += "Final Evaluation: Return an exact numeric veracity score for the text, and provide a matching label out of these six [true, mostly-true, half-true, barely-true, false, pants-fire]"
    return prompt

def get_top_100_statements(user_input):
    # Query ChromaDB for top 100 similar inputs based on cosine similarity
    results = collection.query(
        query_texts=[user_input],
        n_results=100,
        include=["documents"],
        where = {"source": {"$in": ["train", "test", "validate"]}}
    )

    #store and return the number of each 100 statements in dictionary
    statement_dic = {}
    for i in range(100):
        try:
            statement = results['documents'][0][i].split(', ')[2]
        except:
            statement = "placeholder statement"
        if statement not in statement_dic:
            statement_dic[statement] = 1
        else:
            statement_dic[statement] += 1   
    return statement_dic

def combine_pdf_and_prompt(prompt: str, news_text: str) -> str:
    """Combines the user's prompt with the extracted PDF text."""
    if not news_text:
        return prompt
    return f"Prompt: {prompt}\n\nPDF Content: {news_text}"  # Combine entire text

def chunk_news_text(news_text: str) -> list[str]:
    if not news_text:
        return []
        
    prompt = """Split the following text into logical chunks (max 10 chunks) of reasonable size. 
    Preserve complete paragraphs and maintain context. Return ONLY the chunks as a numbered list, with no additional text.
    Format each chunk like:
    1. [chunk text]
    2. [chunk text]
    etc.

    Text to split:
    {text}
    """

    response = chat_session.send_message(prompt.format(text=news_text))
    chunks_text = response.text.strip()
    
    # Split on numbered lines and clean up
    chunks = []
    for line in chunks_text.split('\n'):
        # Skip empty lines
        if not line.strip():
            continue
        # Remove the number prefix and clean whitespace
        chunk = line.split('.', 1)[-1].strip()
        if chunk:
            chunks.append(chunk)
            
    return chunks
   
    
# Sends API call to GenAI model with user input
def call_api(input_text):
    context = " "
    # Add context to the prompt
    full_prompt = f"Context: {context}\n\nUser: {input_text}"
    response = chat_session.send_message(full_prompt, tool_config=tool_config)
    # time.sleep(0.5)
    # yield response.candidates[0].content.parts[0].text
    yield response.parts[0].text

# Display output from GenAI model
def output():
    state = me.state(State)
    if state.output or state.in_progress:
        with me.box(
            style=me.Style(
                background="#F0F4F9",
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin(top=36),
            )
        ):
            if state.output:
                me.markdown(state.output)
            if state.in_progress:
                with me.box(style=me.Style(margin=me.Margin(top=16))):
                    me.progress_spinner()    

############################################ end page elements ############################################

def footer():
    with me.box(style=me.Style(padding=me.Padding(top=16), display="flex", justify_content="center", align_items="center")):
        me.text(
            "DSC180 -- GenAI For Good",
            style=me.Style(font_size=12,color="#777",)
        )
    
############################################ miscellaneous not used ############################################
# Manual add function for database (would be deprecated when dataset processing is fully automized)
def db_input():
    state = me.state(State)
    with me.box(
        style=me.Style(
            padding=me.Padding.all(8),
            background="white",
            display="flex",
            width="100%",
            border=me.Border.all(
                me.BorderSide(width=0, style="solid", color="black")
            ),
            border_radius=12,
            box_shadow="0 10px 20px #0000000a, 0 2px 6px #0000000a, 0 0 1px #0000000a",
            margin=me.Margin(top=36),
        )
    ):
        with me.box(
            style=me.Style(
                flex_grow=1,
            )
        ):
            me.native_textarea(
                value=state.db_input,
                autosize=True,
                min_rows=4,
                placeholder="Enter statement to be added to database",
                style=me.Style(
                    padding=me.Padding(top=16, left=16),
                    background="white",
                    outline="none",
                    width="100%",
                    overflow_y="auto",
                    border=me.Border.all(
                        me.BorderSide(style="none"),
                    ),
                ),
                on_blur=db_textarea_on_blur,
            )
        with me.content_button(type="icon", on_click=click_add_to_db):
            me.icon("add")

def db_textarea_on_blur(e: me.InputBlurEvent):
    state = me.state(State)
    state.db_input = e.value

def click_add_to_db(e: me.ClickEvent):
    state = me.state(State)
    if not state.db_input:
        return

    collection.add(
                documents=[state.db_input],         
                metadatas=[{"source": 'manual', "row_index": 'none'}],            
                ids=[f"doc_{int(time.time())}"]   
            )
    
    state.db_output = f"Manually added to database: {state.db_input[:50]}..."
    state.db_input = ""
    yield

# Display database state for added vector
def db_output():
    state = me.state(State)
    if state.db_output:
        with me.box(
            style=me.Style(
                background="#F0F4F9",
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin(top=36),
            )
        ):
            me.text(state.db_output)

#converting lierplus dataset and store in datebase
def convert_store_lp_data():
    train_data = pd.read_csv('data/train2.tsv', sep='\t',header=None, dtype=str)
    test_data = pd.read_csv('data/test2.tsv', sep='\t', header=None, dtype=str)
    validate_data = pd.read_csv('data/val2.tsv', sep='\t',header=None, dtype=str)

    datasets = [
        {"data": train_data, "source": "train"},
        {"data": test_data, "source": "test"},
        {"data": validate_data, "source": "validate"}
    ]
   
    for dataset in datasets:
        source = dataset["source"]
        data = dataset["data"]
     # Iterate over each row, combining it into a paragraph and processing it
        for idx, row in data.iterrows():
            # Combine row data into a single string (statement + metadata)
            statement = ', '.join(row.astype(str))

            # Store statement and metadata in ChromaDB
            collection.add(
                documents=[statement],         
                metadatas=[{"source": source, "row_index": idx}],            
                ids=[f"{source}_doc_{idx}"]   
            )

    print("All data has been successfully processed and stored in ChromaDB.")


#converting predictive ai generated dataset and store in datebase
def convert_store_predai_data():
    train_data = pd.read_csv('PredictiveAI/train_data_full.tsv', sep='\t',header=None, dtype=str)
    test_data = pd.read_csv('PredictiveAI/test_data_full.tsv', sep='\t', header=None, dtype=str)
    validate_data = pd.read_csv('PredictiveAI/val_data_full.tsv', sep='\t',header=None, dtype=str)
    micro_factors = pd.read_csv('PredictiveAI/average_scores.tsv', sep='\t',header=None, dtype=str)

    datasets = [
        {"data": train_data, "source": "train"},
        {"data": test_data, "source": "test"},
        {"data": validate_data, "source": "validate"},
        {"data": micro_factors, "source": "factors"}
    ]
   
    for dataset in datasets:
        source = dataset["source"]
        data = dataset["data"]
     # Iterate over each row, combining it into a paragraph and processing it
        for idx, row in data.iterrows():
            # Combine row data into a single string (statement + metadata)
            statement = ', '.join(row.astype(str))

            # Store statement and metadata in ChromaDB
            collection.add(
                documents=[statement],         
                metadatas=[{"source": source, "row_index": idx}],            
                ids=[f"{source}_doc_{idx}"]   
            )

    print("All PredAI data has been successfully processed and stored in ChromaDB.")

def obtain_model_accuracy(test_size=20):
    global _prediction_engine
    """Doesn't work when function calling is activated, test_size=20 is when Gemini actually gives useful responses"""
    # Load test statements
    test = pd.read_csv("../data/test2.tsv", sep="\t", header=None, dtype=str).drop(columns=[0]).sample(n=test_size, random_state=15)
    test_statements = [datapoint[1][3] for datapoint in test.iterrows()]

    print(test.iloc[:,1].to_list())

    predictions = []
    rag_retrievals = []
    _prediction_engine.load_dataset_and_prepare_models()
    for statement in test.iterrows():
        predict_score = _prediction_engine.predict_new_example(statement[1])['overall'][0]
        predictions.append(predict_score)
        rag_100 = get_top_100_statements(statement[1][3])
        rag_retrievals.append(rag_100)

    prompt = 'For each of these news statements, use 3 iterations to determine the veracity within the statement. In each iteration, determine what you missed in the previous iteration based on your evaluation of the objective functions. Also put the result from RAG into consideration/rerank.'
    for i in range(1, 4):
        prompt += f"Iteration {i}: Evaluate the text based on the following objectives and also on microfactors:\n"
        prompt += "\nFactuality Factor 1: Frequency Heuristic:\n"
        for fh in frequency_heuristic:
            prompt += f"{fh['description']}: {fh['details']}\n"
        prompt += "\nFactuality Factor 2: Misleading Intentions:\n"
        for mi in misleading_intentions:
            prompt += f"{mi['description']}: {mi['details']}\n"
        prompt += "\nDo not provide any explanation and only give the final output.\n\n"
    prompt += "Final output: For each of the statements, return an exact numeric veracity score for the text, and provide a matching label out of these six [true, mostly-true, half-true, barely-true, false, pants-fire]. Return the labels within <> markers.\n"
    prompt += f"Rule: you have to return the exact number of outputs({test_size}) as the number of inputs. Make sure this rule is followed by adding an index to each output."
    prompt += f'\n\n RAG:\n Here, out of six potential labels (true, mostly-true, half-true, barely-true, false, pants-fire), these are the respective truthfulness labels predicted using a classifier model: {predictions}\n RAG: These are the top 100 related statements in LiarPLUS dataset that related to each of the corresponding news statement: {rag_retrievals}'

    combined_input = f"Prompt: {prompt}\n\n Content: {test_statements}"
    
    output = ''
    for chunk in call_api(combined_input):
        output += chunk

    print(output)
    # Extract label from response
    try:
        labels = [l.lower() for l in re.findall(r'<([\w-]+)>', output)]
    except Exception as e:
        print(f"Error extracting label: {str(e)}")

    print(labels)
    test_accuracy = accuracy_score(test.iloc[:,1].to_list(), labels)
    report = classification_report(test.iloc[:,1].to_list(), labels)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print('=====================\n\n')
    print(report)

# # Verify the data count in ChromaDB
# doc_count = collection.count()
# print(f"Total documents stored in ChromaDB: {doc_count}")
