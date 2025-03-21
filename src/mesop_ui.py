import os
import time
import io
import mesop as me
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import content_types
from collections.abc import Iterable
from collections import defaultdict
import chromadb
import PyPDF2
import http.client
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
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBBuM8yb72NEARwvEJLr_ZSGPKQYBn0tSQ'

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
    model_name="gemini-2.0-flash-001",
    generation_config=generation_config,
    tools=obj_funcs,
)

def tool_config_from_model(mode: str, fns: Iterable[str] = ()):
    """Create a tool config with the specified function calling mode."""
    return content_types.to_tool_config(
        {"function_calling_config": {"mode": mode, "allowed_function_names": fns}}
    )

chat_session = model.start_chat(history=[])

# Global variables
_prediction_engine = None
_related_search_results = None
# _function_calling_outputs = None
############################################ mesop app ############################################
@me.stateclass
class State:
    input: str = ""
    output: str = ""
    in_progress: bool = False
    search_input_text: str = ""
    search_display: str = ""
    db_input: str = ""
    db_output: str = ""
    file: me.UploadedFile | None = None
    news_title: str = ""
    news_author: str = ""
    nes_source: str = ""
    news_text: str = ""
    initialized: bool = False
    in_building: bool = False
    training_status: str = ""
    factuality_factors: list[str]
    use_pred_model: bool = True
    use_gen_model: bool = True
    use_search_online: bool = True
    use_function_calling: bool = True
    use_rag: bool = True
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
    elif e.key == "rag":
        state.use_rag = e.checked
        print("Retrieval-augmented Generation: " + str(state.use_rag))

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
                search_input()
                chat_input()
                search_display()
                output()
            footer()

############################################ begin page elements ############################################
def header_text():
    state = me.state(State)
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
            "Welcome to the Veracity Machine, a tool for analyzing the veracity of news articles. This tool uses a combination of predictive models, generative models, and external search functionality to analyze news articles for veracity.",
            style=me.Style(font_size=16)
        )

    # API key input
    with me.box(style=me.Style(padding=me.Padding(bottom=16))):
        me.text(
            "Enter your Google API key below, your API key needs to support Gemini 2.0 flash. If you don't have an API key, you can use the default key provided.",
        )
        me.input(
            label="API key (leave empty if use default)", 
            appearance="outline", 
            on_blur=apikey_on_blur,
            placeholder="",
            style=me.Style(width=400, padding=me.Padding(top=8))
        )

    # FF table
    with me.box(style=me.Style(padding=me.Padding(bottom=16))):
        me.text("Factuality factors:")
        state.factuality_factors = ["Echo Chamber", "Education", "Event Coverage", "Frequency Heuristic", "Location", "Malicious Account", "Misleading Intention", "News Coverage"]
        me.markdown(" | ".join(state.factuality_factors))

def apikey_on_blur(e: me.InputBlurEvent):
    if e.value:
        genai.configure(api_key=e.value)
    else:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

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
            "Generative Model (gemini-2.0-flash-001)", 
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
        me.text("3. Choose additional functions:")
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
            style=me.Style(margin=me.Margin(right=64))
        )
        me.checkbox(
            "Retrieval-augmented Generation",
            checked=state.use_rag,
            on_change=on_change_checkbox,
            key='rag',
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
    
    with me.box(style=me.Style(display="flex", justify_content="center")):
        me.textarea(
            label="Or paste the news text here:",
            placeholder="Paste and Press Enter",        
            shortcuts={
                me.Shortcut(key="enter"): handle_manual_input,
            },
            style=me.Style(width=800, height=200)
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
            
        return f"Title: {title}\n\n{news_text}"

    if is_valid_url(e.value):
        try:
            state.news_text = preprocess_news_text(scrape_news_text(e.value))
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
    state.news_text = preprocess_news_text(extracted_text)  # Store extracted PDF text in state

def handle_manual_input(e: me.InputEnterEvent):
    state = me.state(State)
    state.news_text = preprocess_news_text(e.value)

# Parse preprocessing outputs
def parse_tagged_string(text):
    tag_pattern = re.compile(r'<\s*([a-zA-Z0-9\s]+)\s*>([\s\S]*?)<\/\s*\1\s*>', re.DOTALL)
    parsed_data = defaultdict(list)
    
    for match in tag_pattern.finditer(text):
        tag, content = match.groups()
        content = content.strip()

        parsed_data[tag].append(content)
    print(parsed_data.keys())
    return {key: value if len(value) > 1 else value[0] for key, value in parsed_data.items()}

# Preprocess the news text by chunking it into logical sections and extracting meta data
def preprocess_news_text(news_text):
    state = me.state(State)
    if not news_text:
        return ""
        
    prompt = f"""Preprocess the following news article for later analysis. 
    First, extract a list of metadata elements <list> [title, author, publication source] </list> and enclose the elements with their respective tags, for example <publication source> CNN </publication source>.
    Next, split the text into 1-5 logical chunks of reasonable size, number of chunks depends on length and complexity of overall article. Enclose each chunk with <chunk> and </chunk> tags.
    Lastly, to prepare this news article for analysis, give me 1 set of key words used for online searches to find related current affairs or context. This set of keywords should be extremely pertinent to the subject matter of the article, and the search results should fill in gaps in your latest knowledge. Examples key words: US tariff on Canada and Mexico.
    Preserve the original text and do not alter main article. Return ONLY the required material with no additional explanation.
    
    Output format:
    <title> [Title of the article] </title> <br>
    <author> [Author of the article] </author> <br>
    <publication source> [Source of the article] </publication source> <br>
    <search keywords> [Keywords for online search] </search keywords> <br>
    <chunk> </chunk> <br>
    ...

    Text to split:
    {news_text}
    """
    print("Calling chunking send_message")
    chunk_text = chat_session.send_message(prompt, tool_config=tool_config_from_model("none"))
    print("Returned chunking output")

    # Parse the tagged string
    print(f"DIRECT PREPROCESSING FROM GEMINI: \n {chunk_text.text}")
    parsed_data = parse_tagged_string(chunk_text.text)
    state.news_title = parsed_data['title']
    state.news_author = parsed_data['author']
    state.news_source = parsed_data['publication source']
    state.search_input_text = parsed_data['search keywords']
    state.news_text = parsed_data['chunk']

    return chunk_text.text

# Display the news input
def display_news_article():
    state = me.state(State)
    if state.news_text:
        with me.box(style=me.Style(padding=me.Padding.all(10), border=me.Border.all(me.BorderSide(width=3, color="black", style="double")), border_radius=10, overflow="auto", max_height=400)):
            me.text("News Article:")
            me.text(state.news_text, type="body-1")  # Display the entire text


# ------------------------------------------ Search Functionality -------------------------------------------
# Input window for user questions
def search_input():
    state = me.state(State)
    with me.box(style=me.Style(padding=me.Padding(top=32))):
        me.text("Search Online for Related Information:", style=me.Style(font_weight="bold"))
    with me.box(style=me.Style(padding=me.Padding.all(8), background="white", display="flex", width="100%", border=me.Border.all(me.BorderSide(width=0, style="solid", color="black")), border_radius=12, box_shadow="0 10px 20px #0000000a, 0 2px 6px #0000000a, 0 0 1px #0000000a", margin=me.Margin(top=16))):
        with me.box(style=me.Style(flex_grow=1)):
            me.native_textarea(
                value=state.search_input_text,
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
        with me.content_button(type="icon", on_click=click_search_send):
            me.icon("send")

# Retains the user's input in the chat window
def search_textarea_on_blur(e: me.InputBlurEvent):
    state = me.state(State)
    state.search_input_text = e.value

# Initiate the search process 
def click_search_send(e: me.ClickEvent):
    yield from initiate_search()

def initiate_search():
    global _related_search_results
    state = me.state(State)
    if not state.search_input_text.strip():
        return

    state.in_progress = True  # Start spinner for progress indication
    user_prompt = state.search_input_text
    state.search_display = "Searching for results...\n"
    yield

    # Conduct the web search
    search_results = web_search(user_prompt)

    # Format the search results for display
    if search_results:
        # Rough parse of search results
        state.search_output = {'organic_results': [], 'top_stories': [], 'knowledge_graph': {}}
        if 'organic' in search_results.keys():
            state.search_output['organic_results'] = search_results['organic']
        if 'topStories' in search_results.keys():
            state.search_output['top_stories'] = search_results['topStories']
        if 'knowledgeGraph' in search_results.keys():
            state.search_output['knowledge_graph'] = search_results['knowledgeGraph']

        _related_search_results = state.search_output
        state.search_display = f"# Search Results for '{user_prompt}':\n\n" + format_search_results(state.search_output)
    else:
        state.search_display = "No results found for your query."

    state.in_progress = False
    yield

def format_search_results(results):
    search_display_text = ""
    search_display_text += "### Organic Results\n"
    for o_r in results['organic_results']:
        for title, item in o_r.items():
            search_display_text += f"    {title.replace('_', ' ').capitalize()}: {item}\n"
        search_display_text += "\n"

    search_display_text += "### Top Stories\n"
    for t_s in results['top_stories']:
        for title, item in t_s.items():
            search_display_text += f"    {title.replace('_', ' ').capitalize()}: {item}\n"
        search_display_text += "\n"

    search_display_text += "### Knowledge Graph\n"
    for title, item in results['knowledge_graph'].items():
        if title not in ['serpapi_knowledge_graph_search_link', 'kgmid', 'imageUrl']:
            search_display_text += f"    {title.replace('_', ' ').capitalize()}: {item}\n"
    search_display_text += "\n"
    return search_display_text

def web_search(user_prompt):
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
    "q": user_prompt,
    })
    headers = {
    'X-API-KEY': '6315ba893c6d94e58d5f0a386592a0cab6e8a78c',
    'Content-Type': 'application/json'
    }
    # Conduct a Google Custom Search query
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    search = res.read()
    print("Returned search results: " + search.decode("utf-8")[:200])
    
    return json.loads(search.decode('utf-8'))

def search_display():
    state = me.state(State)
    if state.search_display:
        with me.box(
            style=me.Style(
                background="#F0F4F9",
                padding=me.Padding.all(16),
                border_radius=16,
                margin=me.Margin(top=36),
                max_height=400,
                overflow="auto",
            )
        ):
            me.markdown(state.search_display)

# -------------------------------------------------------- Prompting ------------------------------------------------------------
# User input for GenAI prompting
def chat_input():
    state = me.state(State)
    with me.box(style=me.Style(padding=me.Padding(top=32, bottom=16))):
        me.text(
            "One Final Step! Double Click send to generate an analysis.",
            style=me.Style(font_weight="bold")
        )
        me.text(
            "(if buggy response, refresh the page and try again)",
        )
    with me.box(style=me.Style(padding=me.Padding.all(8), background="white", display="flex", width="100%", border=me.Border.all(me.BorderSide(width=0, style="solid", color="black")), border_radius=12, box_shadow="0 10px 20px #0000000a, 0 2px 6px #0000000a, 0 0 1px #0000000a",)):
        with me.box(style=me.Style(flex_grow=1,)):
            me.native_textarea(
                value=state.input,
                autosize=True,
                max_rows=6,
                placeholder="Enter any additional information or context if needed.",
                style=me.Style(padding=me.Padding(top=16, left=16), background="white", outline="none", width="100%", overflow_y="auto", border=me.Border.all(me.BorderSide(style="none"),),),
                on_blur=textarea_on_blur,
            )
        with me.content_button(type="icon", on_click=initiate_analysis):
            me.icon("send")

def textarea_on_blur(e: me.InputBlurEvent):
    state = me.state(State)
    state.input = e.value

def initiate_analysis(e: me.ClickEvent):
    global _prediction_engine
    global _related_search_results
    state = me.state(State)

    # Kickstart veracity analysis workflow
    if not state.input.strip():  # Check if input is empty or contains only whitespace
        state.input = "User didn't provide more context."  # Default FCT prompt if no input is provided
        return
    
    state.in_progress = True
    yield

    # Predictive model
    if _prediction_engine is None:
        train_predictive()
    predict_score = _prediction_engine.predict_new_example(convert_statement_to_series(state.news_text))['overall']
    print(f"Predictive model score: {predict_score}")

    # Function Calling
    if state.use_function_calling:
        function_calling_outputs = call_function(state.news_text)
    else:
        function_calling_outputs = None

    # RAG
    if state.use_rag:
        top_100_statements = get_top_100_statements(state.news_text)
    else:
        top_100_statements = None

    # Search results
    yield from initiate_search()
    if state.use_search_online:
        search_domain = state.search_display
    else:
        search_domain = None

    # Generate FCoT prompt
    context = " "
    combined_input = generate_fct_prompt(
        state.news_text, 
        predict_score, 
        search_result=search_domain,
        function_calling_outputs=function_calling_outputs, 
        rag = top_100_statements,
        additional_info = state.input,
        prompt_type=state.prompt_type,
    )

    for chunk in call_api(context, combined_input):
        state.output += chunk
        yield

    state.output += '\n\n The result of factuality scores function calling based on news article:\n\n' + str(function_calling_outputs) + '\n\n'
    state.output += '\n\n The probability of the statement truthness:\n\n' + str(top_100_statements) + '\n\n'

    state.in_progress = False
    yield

def convert_statement_to_series(statement):
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
def generate_fct_prompt(
    input_text, 
    predict_score, 
    search_result=None,
    function_calling_outputs=None, 
    rag=None,
    additional_info=None, 
    iterations=3, 
    prompt_type="FCoT",
):
    # Normal prompting
    if prompt_type == "normal":
        print("------ NORMAL PROMPTING ------")
        prompt = f'''
        Analyze the provided news article content using ALL of the following **Factuality Factors** to detect disinformation or misinformation effectively. Your analysis should be as factual, logical, and reasonable as possible, while taking into account additional information.

        ------------

        ### Factuality Factors:
        1.** Frequency Heuristics **:
        - (micro factor 1) Repetition Analysis: How frequently and widely is the claim repeated across different platforms and sources?
        - (micro factor 2) Origin Tracing: Where did the frequently repeated information originate? Is it from a credible source, or is it from a questionable origin (e.g., bot network, conspiracy website)?
        - (micro factor 3) Evidence Verification: Does the claim have supporting evidence, or does its apparent truth rely solely on repetition?

        2.** Misleading Intentions **:
        - (micro factor 1) Omission Checks: Does the content deliberately leave out crucial details that significantly alter the interpretation or perception of the information presented?
        - (micro factor 2) Exaggeration Analysis: Does the content contain unsupported claims, exaggerations, or hyperbole that distort the truth or mislead the audience?
        - (micro factor 3) Target Audience Assessment: Does the content target vulnerable or impressionable groups (e.g., children, elderly, people with specific beliefs) with the apparent intention of misleading them?

        ------------

        ### Elements to include in your analysis
        - Assign a score out of [0, 1, 2, 3, 4] for each of the micro factors, the larger the number, the more prominent (untruthful) the factor is.
        - Ensure that all areas with suspicion have been eliminated through fact checking with your knowledge.
        - Identify and check potential areas of concern, ensure all gaps or omissions are addressed.
        - Include a summary highlighting key adjustments and final observations.
        - Calculate a final **Truthfulness Score**, and provide a verdict using one of these six ordinal labels, they should closely match the truthfulness score you've derived above: <labels> “True”: 0.90-1, “Mostly-True”: 0.6-0.9, “Half-True”: 0.4-0.6, “Barely-True”: 0.25-0.4, “False”: 0.1-0.25, “Pants on Fire”: 0-0.1 </labels>

        ** Important Note **
        Consistency check: All of the scoring rubrics are in the same scale, meaning the larger the score, the more untruthful the article is for this factor. Ensure that all of your scores make sense on their individual scales based on your reasoning, as well as their combined truthfulness score. For example, if one of your factor scores have increased, meaning you found the article to be less truthful, then the overall truthfulness score should decrease during that iteration. Vice versa. The degree of influence on the overall score depends on the significance of the individual factors, and the weights should be determined by you. Use this equation for calculating truthfulness score: 
        <equation> Truthfulness score = 1 - ((w_0 * Score(Repetition Analysis) + w_1 * Score(Origin Tracing) + w_2 * Score(Evidence Verification) + w_3 * Score(Omission Check) + w_4 * Score(Exaggeration Analysis) + w_5 * Score(Target Audience Assessment))/24) </equation>
        
        ------------

        ### Factual Additional Information:
        These additional information should help you to guide your analysis:
        - **Predictive Classification Model**: The overall truthfulness label (true, mostly-true, half-true, barely-true, false, pants-fire) predicted using a classifier model. <prediction label> {predict_score} </ prediction label>
        - **Related Search Results**: Information retrieved from online sources related to the news article. <online search> {search_result} </ online search>
        - **Function Calling Outputs**: Based on the news article, different factuality scores are examined utilizing separate function calling and these are the results for different factors. <function calling> {function_calling_outputs} </ function calling>
        - **Retrieval-augmented Generation**: Here is a list of the top 100 related news statements from the LiarPLUS dataset, a collection of ground truth statements. <rag> {rag} </ rag>
        - **Additional Information**: Any additional context or information provided by the user. <additional info> {additional_info} </ additional info>

        ------------

        ### Output format:
        1. **Frequency Heuristics**:
        - **Repetition Analysis**: [Your score]
        - Explanation: [Explanation of the score]
        - **Origin Tracing**: [Your score]
        - Explanation: [Explanation of the score]
        - **Evidence Verification**: [Your score]
        - Explanation: [Explanation of the score]

        2. **Misleading Intentions**:
        - **Omission Checks**: [Your score]
        - Explanation: [Explanation of the score]
        - **Exaggeration Analysis**: [Your score]
        - Explanation: [Explanation of the score]
        - **Target Audience Assessment**: [Your score]
        - Explanation: [Explanation of the score]

        3. **Final Truthfulness Score**:
        - Based on refined scores, calculate a final truthfulness score (0 to 1).
        - Provide a summary explaining the final score and key observations.

        ------------

        ### News Article:
        {input_text}
        '''

    # Only use regular CoT prompting if identified
    if prompt_type == "CoT":
        print("------ CHAIN OF THOUGHT PROMPTING ------")
        prompt = f'''
        You are an expert at identifying misinformation and disinformation within news articles, such as bias, manipulative tactics, or false information. You will perform all analysis based on supporting evidence either from your existing knowledge or additional context. All fact-checking must be thorough and accurate. 

        ### Objective:
        Think step by step, analyze the provided news article content using ALL of the following **Factuality Factors** to detect disinformation or misinformation effectively. Your analysis should be as factual, logical, and reasonable as possible, while taking into account additional information. 

        ------------

        ### Factuality Factors:
        1.** Frequency Heuristics **: 
        - (micro factor 1) Repetition Analysis: How frequently and widely is the claim repeated across different platforms and sources? 

        - (micro factor 2) Origin Tracing: Where did the frequently repeated information originate? Is it from a credible source, or is it from a questionable origin (e.g., bot network, conspiracy website)? 

        - (micro factor 3) Evidence Verification: Does the claim have supporting evidence, or does its apparent truth rely solely on repetition? This is a critical check to avoid the "illusory truth effect." 

        2.** Misleading Intentions **:
        - (micro factor 1) Omission Checks: Does the content deliberately leave out crucial details that significantly alter the interpretation or perception of the information presented? 

        - (micro factor 2) Exaggeration Analysis: Does the content contain unsupported claims, exaggerations, or hyperbole that distort the truth or mislead the audience?

        - (micro factor 3) Target Audience Assessment: Does the content target vulnerable or impressionable groups (e.g., children, elderly, people with specific beliefs) with the apparent intention of misleading them? 
        ------------

        ### Elements to include in your analysis
        - Assign a score out of [0, 1, 2, 3, 4] for each of the micro factors, the larger the number, the more prominent (untruthful) the factor is.
        - Ensure that all areas with suspicion have been eliminated through fact checking with your knowledge.
        - Identify and check potential areas of concern, ensure all gaps or omissions are addressed.
        - Include a summary highlighting key adjustments and final observations.
        - Calculate a final **Truthfulness Score**, and provide a verdict using one of these six ordinal labels, they should closely match the truthfulness score you've derived above: <labels> “True”: 0.90-1, “Mostly-True”: 0.6-0.9, “Half-True”: 0.4-0.6, “Barely-True”: 0.25-0.4, “False”: 0.1-0.25, “Pants on Fire”: 0-0.1 </labels>

        ** Important Note **
        Consistency check: All of the scoring rubrics are in the same scale, meaning the larger the score, the more untruthful the article is for this factor. Ensure that all of your scores make sense on their individual scales based on your reasoning, as well as their combined truthfulness score. For example, if one of your factor scores have increased, meaning you found the article to be less truthful, then the overall truthfulness score should decrease during that iteration. Vice versa. The degree of influence on the overall score depends on the significance of the individual factors, and the weights should be determined by you. Use this equation for calculating truthfulness score: 
        <equation> Truthfulness score = 1 - ((w_0 * Score(Repetition Analysis) + w_1 * Score(Origin Tracing) + w_2 * Score(Evidence Verification) + w_3 * Score(Omission Check) + w_4 * Score(Exaggeration Analysis) + w_5 * Score(Target Audience Assessment))/24) </equation>
        
        ------------

        ### Factual Additional Information:
        These additional information should help you to guide your analysis:
        - **Predictive Classification Model**: The overall truthfulness label (true, mostly-true, half-true, barely-true, false, pants-fire) predicted using a classifier model. <prediction label> {predict_score} </ prediction label>
        - **Related Search Results**: Information retrieved from online sources related to the news article. <online search> {search_result} </ online search>
        - **Function Calling Outputs**: Based on the news article, different factuality scores are examined utilizing separate function calling and these are the results for different factors. <function calling> {function_calling_outputs} </ function calling>
        - **Retrieval-augmented Generation**: Here is a list of the top 100 related news statements from the LiarPLUS dataset, a collection of ground truths from PolitiFact. <100 statements> {rag} </ 100 statements>
        - **User-provided Context**: Any additional context or information provided by the user. <additional context> {additional_info} </ additional context>
        
        ------------

        ### Output format:
        1. **Frequency Heuristics**:
        - **Repetition Analysis**: [Your score]
        - Explanation: [Explanation of the score]
        - **Origin Tracing**: [Your score]
        - Explanation: [Explanation of the score]
        - **Evidence Verification**: [Your score]
        - Explanation: [Explanation of the score]

        2. **Misleading Intentions**:
        - **Omission Checks**: [Your score]
        - Explanation: [Explanation of the score]
        - **Exaggeration Analysis**: [Your score]
        - Explanation: [Explanation of the score]
        - **Target Audience Assessment**: [Your score]
        - Explanation: [Explanation of the score]

        3. **Final Truthfulness Score**:
        - Based on refined scores, calculate a final truthfulness score (0 to 1).
        - Provide a summary explaining the final score and key observations.

        ------------

        ### News Article:
        {input_text}
        '''
        return prompt
    
    # Fractal Chain of Thought Prompting with Objective Functions
    if prompt_type == "FCoT":
        print("------ FRACTAL CHAIN OF THOUGHT PROMPTING ------")
        prompt = f'''
        You are an expert at identifying misinformation and disinformation within news articles, such as bias, manipulative tactics, or false information. You will perform all analysis based on supporting evidence either from your existing knowledge or additional context. All fact-checking must be thorough and accurate. 

        ### Objective:
        Analyze the provided news article content using the following **Factuality Factors** to detect disinformation or misinformation effectively. Your analysis should be as factual, logical, and reasonable as possible, while taking into account additional information.

        ------------

        ### Factuality Factors:
        1.** Frequency Heuristics **: 
        - (micro factor 1) Repetition Analysis: How frequently and widely is the claim repeated across different platforms and sources? Score based on the following criteria:
        0 (Rare): Claim appears very infrequently and in limited locations.
        1 (Limited): Claim appears in a few sources, with limited reach, and shows no clear pattern of spread.
        2 (Moderate): Claim appears in a moderate number of sources, shows some reach, and may have some periods of increased activity.
        3 (Frequent): Claim appears frequently, has significant reach, and shows a clear pattern of spread across multiple platforms.
        4 (Widespread): Claim appears extremely frequently, has very high reach, is actively spreading on multiple platforms, and may be trending.

        - (micro factor 2) Origin Tracing: Where did the frequently repeated information originate? Is it from a credible source, or is it from a questionable origin (e.g., bot network, conspiracy website)? Score based on the following criteria:
        0 (Highly Credible): Originates from a highly credible and reliable source (e.g., a reputable news agency with a history of accurate reporting, a peer-reviewed scientific journal).
        1 (Credible): Originates from a generally credible source (e.g., established news organization, government agency), but there might be some caveats (e.g., known biases).
        2 (Neutral): Originates from a source with a neutral reputation or a source where credibility is difficult to assess (e.g., a blog with no clear editorial standards, a social media post from an individual with no established expertise).
        3 (Questionable): Originates from a source with a questionable reputation (e.g., a website known for spreading rumors or conspiracy theories, a social media account with a history of spreading misinformation).
        4 (Highly Questionable): Originates from a source known to be unreliable or deceptive (e.g., a known purveyor of fake news, a bot network, a source linked to disinformation campaigns).

        - (micro factor 3) Evidence Verification: Does the claim have supporting evidence, or does its apparent truth rely solely on repetition? This is a critical check to avoid the "illusory truth effect." Score based on the following criteria:
        0 (Strongly Supported): The claim is supported by robust evidence from multiple credible sources. There is a clear consensus among experts that the claim is accurate.
        1 (Supported): The claim is supported by some evidence from credible sources, but there may be some caveats or limitations.
        2 (Mixed Evidence): There is mixed evidence supporting and refuting the claim. The evidence may be inconclusive, or there may be conflicting studies.
        3 (Unsupported): There is little or no credible evidence to support the claim. The available evidence suggests that the claim is likely false.
        4 (Strongly Refuted): The claim is strongly refuted by credible evidence. There is a clear consensus among experts that the claim is false. Fact-checking websites have debunked the claim.

        2.** Misleading Intentions **:
        - (micro factor 1) Omission Checks: Does the content deliberately leave out crucial details that significantly alter the interpretation or perception of the information presented? Score based on the following criteria: 
        0 (No Significant Omissions): No relevant details are omitted, or the omissions do not significantly affect the interpretation of the information.
        1 (Minor Omissions): Minor details are omitted, but they have a limited impact on the overall understanding of the information.
        2 (Moderate Omissions): Relevant details are omitted, leading to a slightly skewed or incomplete understanding of the information.
        3 (Significant Omissions): Crucial details are omitted, significantly altering the interpretation of the information and potentially leading to inaccurate conclusions.
        4 (Egregious Omissions): The content deliberately omits vital information to create a false or misleading narrative, with a high likelihood of deceiving the audience.

        - (micro factor 2) Exaggeration Analysis: Does the content contain unsupported claims, exaggerations, or hyperbole that distort the truth or mislead the audience? Score based on the following criteria:
        0 (No Exaggerations): The content contains no unsupported claims, exaggerations, or hyperbole.
        1 (Minor Exaggerations): Minor exaggerations or hyperbole are present, but they do not significantly distort the truth or mislead the audience.
        2 (Moderate Exaggerations): Some claims are exaggerated or presented without sufficient evidence, leading to a slightly distorted understanding of the information.
        3 (Significant Exaggerations): Significant claims are exaggerated or presented without evidence, significantly distorting the truth and potentially misleading the audience.
        4 (Gross Exaggerations): The content is filled with gross exaggerations, unsupported claims, and hyperbole, intended to deceive and manipulate the audience.

        - (micro factor 3) Target Audience Assessment: Does the content target vulnerable or impressionable groups (e.g., children, elderly, people with specific beliefs) with the apparent intention of misleading them? Score based on the following criteria:
        0 (No Targeted Content): The content does not appear to be specifically targeted at any vulnerable or impressionable group.
        1 (Minimal Targeting): The content may have some elements that appeal to a specific audience, but there is no clear intention to mislead them.
        2 (Moderate Targeting): The content targets a specific audience and may contain some elements that could potentially mislead them.
        3 (Significant Targeting): The content is clearly targeted at a vulnerable or impressionable group, with a moderate likelihood of misleading them.
        4 (Exploitative Targeting): The content is explicitly designed to exploit the vulnerabilities of a specific group for malicious purposes, with a high likelihood of causing harm.

        ------------

        ### Iterative Analysis Instructions:
        Perform analysis over **{iterations} iterations**, refining the results in each pass:
        1. **Iteration 1**:
        - Conduct a preliminary analysis using the Factuality Factors, with your knowledge base.
        - Identify potential areas of concern that warrant further investigation.
        - Assign preliminary scores for each factor and provide explanations for the scores.
        - Conclude with a preliminary **Truthfulness Score** between range [0, 1] (the higher the more truthful, this scale is different from the factor scores). 

        2. **Iteration 2**:
        - Reflect on areas where the initial analysis missed nuances or misjudged factors.
        - Refine the analysis with deeper insights from context and search results.
        - Adjust scores for each factor and document improvements, and if score has changed, explain the reason for the change.
        - Provide an updated **Truthfulness Score**.

        3. **Iteration 3**:
        - Conduct a final review focusing on comprehensiveness:
        - Ensure that all areas with suspicion have been eliminated through fact checking with your database.
        - Confirm that all gaps or omissions identified in earlier iterations are addressed.
        - Include a summary highlighting key adjustments and final observations.
        - Calculate a final **Truthfulness Score**, and provide a verdict using one of these six ordinal labels, they should closely match the truthfulness score you've derived above: <labels> “True”: 0.90-1, “Mostly-True”: 0.6-0.9, “Half-True”: 0.4-0.6, “Barely-True”: 0.25-0.4, “False”: 0.1-0.25, “Pants on Fire”: 0-0.1 </labels>

        ** Important Note **
        Consistency check: All of the scoring rubrics are provided in the same scale, meaning the larger the score, the more untruthful the article is for this factor. Ensure that all of your scores make sense on their individual scales based on your reasoning, as well as their combined truthfulness score at the end of each iteration. For example, if one of your factor scores have increased, meaning you found the article to be less truthful, then the overall truthfulness score should decrease during that iteration. Vice versa. The degree of influence on the overall score depends on the significance of the individual factors, and the weights should be determined by you. Use this equation for calculating truthfulness score: 
        <equation> Truthfulness score = 1 - ((w_0 * Score(Repetition Analysis) + w_1 * Score(Origin Tracing) + w_2 * Score(Evidence Verification) + w_3 * Score(Omission Check) + w_4 * Score(Exaggeration Analysis) + w_5 * Score(Target Audience Assessment))/24) </equation>
        
        ------------

        ### Output format for each iteration:
        1. **Frequency Heuristics**:
        - **Repetition Analysis**: [Your score]
        - Explanation: [Explanation of the score]
        - **Origin Tracing**: [Your score]
        - Explanation: [Explanation of the score]
        - **Evidence Verification**: [Your score]
        - Explanation: [Explanation of the score]

        2. **Misleading Intentions**:
        - **Omission Checks**: [Your score]
        - Explanation: [Explanation of the score]
        - **Exaggeration Analysis**: [Your score]
        - Explanation: [Explanation of the score]
        - **Target Audience Assessment**: [Your score]
        - Explanation: [Explanation of the score]

        3. **Final Truthfulness Score**:
        - Based on refined scores, calculate a final truthfulness score (0 to 1).
        - Provide a summary explaining the final score and key observations.

        ------------

        ### Factual Additional Information:
        All of these additional information should be used to guide your analysis:
        - **Predictive Classification Model**: The overall truthfulness label (true, mostly-true, half-true, barely-true, false, pants-fire) predicted using a classifier model. <prediction label> {predict_score} </ prediction label>
        - **Related Search Results**: Information retrieved from online sources related to the news article. <online search> {search_result} </ online search>
        - **Function Calling Outputs**: Based on the news article, different factuality scores are examined utilizing separate function calling and these are the results for different factors. <function calling> {function_calling_outputs} </ function calling>
        - **Retrieval-augmented Generation**: Here is a list of the top 100 related news statements from the LiarPLUS dataset, a collection of ground truths from PolitiFact. <100 statements> {rag} </ 100 statements>
        - **User-provided Context**: Any additional context or information provided by the user. <additional context> {additional_info} </ additional context>
        
        ------------

        ### News Article:
        {input_text}

        ### Tabular Output
        At last, besides from the content above, give me a tabular form of output with this format: [Factor Name | Score | Consideration | Reference from article]
        '''
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
   
# Sends API call to GenAI model with user input
def call_api(context, input_text):
    # Add context to the prompt
    full_prompt = f"Context: {context}\n\nUser: {input_text}"
    response = chat_session.send_message(full_prompt, tool_config=tool_config_from_model("none"))
    # time.sleep(0.5)
    # yield response.candidates[0].content.parts[0].text
    print(f"DIRECT RESPONSE FROM GEMINI: \n {response.parts[0].text[:100]}")
    yield response.parts[0].text

def call_function(input_text):
    context = " "
    # Add context to the prompt
    full_prompt = f"Context: {context}\n\nUser: {input_text}"
    response = chat_session.send_message(full_prompt, tool_config=tool_config_from_model("auto"))
    
    response_text = ""
    # _function_calling_outputs = ""
    if response.candidates:
        for candidate in response.candidates:
            if candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call'): #If the model calls a function
                        func_call = part.function_call
                        #Get the function from utils.py
                        func_name = func_call.name
                        func_args = func_call.args

                        try:
                            func_to_call = globals().get(func_name) #get the function from the global scope
                            
                            if func_to_call:
                                if func_args and "text" in func_args: #if "text" is in func_args, we pass all of the args as they are
                                    func_output = func_to_call(**func_args)
                                else: #otherwise we pass it with the text parameter we have
                                    func_output = func_to_call(text = input_text)
                                response_text += f"Function call '{func_name}' output: {func_output}\n"
                                # _function_calling_outputs += f"Function call '{func_name}' output: {func_output}\n"
                            else:
                                response_text += f"Function '{func_name}' not found\n"
                                # _function_calling_outputs += f"Function '{func_name}' not found\n"
                            
                        except Exception as e:
                                response_text += f"Error calling function: {str(e)}\n"
                                # _function_calling_outputs += f"Error calling function: {str(e)}\n"

                    elif hasattr(part, 'text'):
                        response_text += part.text
                        # _function_calling_outputs += part.text
                        
    else:
        response_text += "No candidates returned in response"
        # _function_calling_outputs += "No candidates returned in response"
    
    return response_text

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
        # Fix this line when need to use
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
        for fh in frequency_heuristic:
            prompt += f"{fh['description']}: {fh['details']}\n"
        prompt += "\nFactuality Factor 2: Misleading Intentions:\n"
        # Fix this line when need to use
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
