# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Union
import os
from dotenv import load_dotenv
import requests
import time
import json
import logging
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from urllib.parse import quote
import regex
import threading  # Added for thread safety
import hashlib    # Added for hashing
from collections import OrderedDict  # For implementing LRU cache (optional)

# ----------------------------
# Configuration and Setup
# ----------------------------

load_dotenv()

app = FastAPI(
    title="AI-Integrated Search API",
    description="An API that processes input text, generates search queries using Azure OpenAI, fetches search results from DirtSearch and Bing, aggregates them, and provides a final summarized response.",
    version="1.2.0"
)

#start the app
#uvicorn main:app --host 0.0.0.0 --port 8000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------------
# Cache Implementation
# ----------------------------

# Define the maximum size of the cache
MAX_CACHE_SIZE = 100  # You can adjust this value as needed

# Create a thread-safe cache using an OrderedDict
cache = OrderedDict()
cache_lock = threading.Lock()

def get_cache_key(text: str) -> str:
    """
    Generates a cache key by hashing the cleaned input text.

    :param text: The input text to generate a key for.
    :return: A hexadecimal hash string of the input text.
    """
    # Clean the text to ensure consistency
    cleaned_text = clean_text(text)
    # Use SHA256 hash of the cleaned text as the cache key
    hash_object = hashlib.sha256(cleaned_text.encode('utf-8'))
    cache_key = hash_object.hexdigest()
    return cache_key

def add_to_cache(key: str, value):
    """
    Adds a new item to the cache and ensures the cache size limit.

    :param key: The cache key.
    :param value: The value to store in the cache.
    """
    with cache_lock:
        # If the key already exists, move it to the end (most recently used)
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        # If the cache exceeds the maximum size, remove the oldest item
        if len(cache) > MAX_CACHE_SIZE:
            oldest_key = next(iter(cache))
            cache.pop(oldest_key)

def get_from_cache(key: str):
    """
    Retrieves an item from the cache.

    :param key: The cache key.
    :return: The cached value or None if not found.
    """
    with cache_lock:
        value = cache.get(key)
        if value:
            # Move the key to the end to mark it as recently used
            cache.move_to_end(key)
        return value


# ----------------------------
# Pydantic Models
# ----------------------------

class InputText(BaseModel):
    text: str

class DirtSearchResult(BaseModel):
    title: str
    url: HttpUrl

class BingSearchResultItem(BaseModel):
    title: str
    url: HttpUrl

class SearchResult(BaseModel):
    source: str
    query: Optional[str] = None
    results: Union[List[DirtSearchResult], List[BingSearchResultItem]]

class ProcessedTextResponse(BaseModel):
    source_used: str
    search_results: List[SearchResult]
    result: str
    status: str

# ----------------------------
# Environment Variables
# ----------------------------

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

BING_API_KEY = os.getenv("BING_API_KEY")
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

DIRTSEARCH_BASE_URL = "https://dirtsearch.azurewebsites.net/searchdirt"

# ----------------------------
# Validate Environment Variables
# ----------------------------

required_env_vars = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "BING_API_KEY"
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    missing = ", ".join(missing_vars)
    logger.error(f"Missing required environment variables: {missing}")
    raise EnvironmentError(f"Missing required environment variables: {missing}")

# ----------------------------
# Helper Functions
# ----------------------------

def clean_llm_response(response_text: str) -> str:
    """
    Removes Markdown code block indicators from the LLM response.

    :param response_text: The raw response text from the LLM.
    :return: Cleaned response text without code block markers.
    """
    cleaned_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'```$', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = cleaned_text.replace('```', '')
    return cleaned_text.strip()

def convert_dirtsearch_response(dirtsearch_response: list) -> SearchResult:
    """
    Converts DirtSearch response into the unified SearchResult format.

    :param dirtsearch_response: List of DirtSearch result dictionaries.
    :return: SearchResult instance with source 'DirtSearch'.
    """
    # Transform each result to match the DirtSearchResult model
    transformed_results = []
    for item in dirtsearch_response:
        transformed_item = {
            "title": item.get("claim", "No Claim"),
            "url": item.get("url", "http://example.com")
        }
        transformed_results.append(DirtSearchResult(**transformed_item))
    
    return SearchResult(
        source="DirtSearch",
        results=transformed_results
    )

def convert_bing_search_response(bing_search_response: dict) -> List[SearchResult]:
    """
    Converts Bing Search API response into the unified SearchResult format.

    :param bing_search_response: Dictionary mapping queries to lists of Bing search result items.
    :return: List of SearchResult instances with source 'Bing Search'.
    """
    search_results = []
    
    for query, items in bing_search_response.items():
        parsed_items = []
        for item in items:
            transformed_item = {
                "title": item.get("name", "No Name"),
                "url": item.get("url", "http://example.com"),
            }
            parsed_items.append(BingSearchResultItem(**transformed_item))
        
        search_result = SearchResult(
            source="Bing Search",
            query=query,
            results=parsed_items
        )
        
        search_results.append(search_result)
    
    return search_results

@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException) | retry_if_exception_type(HTTPException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def generate_search_queries(text: str) -> list:
    """
    Generates relevant search queries based on input text using Azure OpenAI.

    :param text: The input text to generate search queries for.
    :return: A list of search query strings.
    """
    logger.info("Generating search queries using Azure OpenAI.")
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }
    prompt = f"""
    You are an assistant that generates relevant search queries based on the following input text.

    Instructions:

    Analyze the input text to determine if it contains any factual claims, statements, or assertions that require validation or further exploration.
    If the text does contain such claims, generate a JSON array of up to 5 search queries that can help validate or explore these claims.
    If the text does not contain any claims needing validation, return an empty JSON array.
    Input: "{text}"

    Output format: JSON array of strings only. Do not include any code blocks or additional text.
    """

    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.7,
        "n": 1
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        logger.debug(f"Azure OpenAI Response Status Code: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Azure OpenAI API error: {response.text}")
            raise HTTPException(status_code=500, detail=f"Azure OpenAI API error: {response.text}")
        
        response_json = response.json()
        queries_text = response_json['choices'][0]['message']['content'].strip()
        logger.debug(f"Raw Queries Text: {queries_text}")
        
        # Clean the response to remove any code block indicators
        queries_text = clean_llm_response(queries_text)
        
        # Parse JSON array
        queries = json.loads(queries_text)
        if not isinstance(queries, list):
            logger.error("The LLM did not return a list of queries.")
            raise HTTPException(status_code=500, detail="The LLM did not return a list of queries.")
        
        logger.info(f"Generated Queries: {queries}")
        return queries

    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException while contacting Azure OpenAI: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError while parsing search queries: {e}")
        raise HTTPException(status_code=500, detail=f"JSONDecodeError while parsing search queries: {e}")
    except Exception as e:
        logger.error(f"Generic error while parsing search queries: {e}")
        raise

def fetch_dirtsearch_results(query: str) -> list:
    """
    Fetches results from the DirtSearch API for a given query.

    :param query: The search query string.
    :return: A list of DirtSearch result dictionaries.
    """
    dirtsearch_url = f"{DIRTSEARCH_BASE_URL}/{quote(query)}"
    logger.info(f"Fetching DirtSearch results for query: {query}")
    try:
        response = requests.get(dirtsearch_url, timeout=10)
        logger.debug(f"DirtSearch Response Status Code for '{query}': {response.status_code}")
        if response.status_code != 200:
            logger.error(f"DirtSearch API error for query '{query}': {response.text}")
            raise HTTPException(status_code=500, detail=f"DirtSearch API error for query '{query}': {response.text}")
        
        results = response.json()
        if not isinstance(results, list):
            logger.error(f"DirtSearch API did not return a list for query '{query}'.")
            raise HTTPException(status_code=500, detail=f"DirtSearch API did not return a list for query '{query}'.")
        
        logger.debug(f"DirtSearch Results for '{query}': {results}")
        return results

    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException while contacting DirtSearch API for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=f"RequestException while contacting DirtSearch API for query '{query}': {e}")
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError while parsing DirtSearch results for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=f"JSONDecodeError while parsing DirtSearch results for query '{query}': {e}")
    except Exception as e:
        logger.error(f"Generic error while fetching DirtSearch results for query '{query}': {e}")
        raise HTTPException(status_code=500, detail=f"Generic error while fetching DirtSearch results for query '{query}': {e}")

@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException) | retry_if_exception_type(HTTPException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def final_prompting(original_text: str, aggregated_data: str) -> str:
    """
    Sends the aggregated search results back to Azure OpenAI for summarization or analysis.

    :param original_text: The original input text containing the claim.
    :param aggregated_data: The aggregated search results as a formatted string.
    :return: The final summarized or analyzed response from Azure OpenAI.
    """
    logger.info("Performing final prompting with aggregated search results using Azure OpenAI.")
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version=2024-02-15-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY
    }
    prompt = f"""
    You are an assistant that analyzes aggregated search results to provide a comprehensive overview.

    Original Claim:
    "{original_text}"

    Here are the aggregated search results:

    {aggregated_data}

    Based on the above information, please do the following:

    1. **Provide a summary** that validates or refutes the original claim.
    2. **Rate** the accuracy of the original claim as "High," "Medium," or "Low."

    **Output format:**

    [Your summary here.]

    Rating: [High/Medium/Low]

    Do not include any code blocks or additional formatting.
    """


    logger.debug(f"Final Prompt: {prompt}")

    data = {
        "messages": [
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.1,
        "n": 1
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        logger.debug(f"Azure OpenAI Final Prompt Response Status Code: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Azure OpenAI API error during final prompting: {response.text}")
            raise HTTPException(status_code=500, detail=f"Azure OpenAI API error during final prompting: {response.text}")
        
        response_json = response.json()
        final_result = response_json['choices'][0]['message']['content'].strip()
        logger.info("Final response successfully obtained from Azure OpenAI.")
        logger.debug(f"Final Result: {final_result}")
        
        # Clean the response to remove any code block indicators, just in case
        final_result = clean_llm_response(final_result)
        
        return final_result

    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException while contacting Azure OpenAI for final prompting: {e}")
        raise
    except KeyError as e:
        logger.error(f"KeyError while parsing final prompt response: {e}")
        raise HTTPException(status_code=500, detail=f"KeyError while parsing final prompt response: {e}")
    except Exception as e:
        logger.error(f"Generic error during final prompting: {e}")
        raise

def execute_dirtsearch_queries(queries: list) -> list:
    """
    Executes all search queries using DirtSearch and collects high-confidence results.

    :param queries: A list of search query strings.
    :return: A list of DirtSearch result dictionaries with high confidence.
    """
    logger.info("Executing all search queries using DirtSearch.")
    all_results = []

    for query in queries:
        logger.info(f"Processing query: {query}")
        try:
            dirtsearch_results = fetch_dirtsearch_results(query)
            high_confidence_results = [
                result for result in dirtsearch_results 
                if result.get("matchConfidence", "").lower() == "high"
            ]
            if high_confidence_results:
                logger.info(f"High confidence results found in DirtSearch for query '{query}'.")
                all_results.extend(high_confidence_results)
            else:
                logger.info(f"No high confidence results in DirtSearch for query '{query}'.")
        except HTTPException as http_exc:
            logger.error(f"HTTPException for query '{query}': {http_exc.detail}")
            raise http_exc  # Re-raise to be handled by FastAPI
        except Exception as e:
            logger.error(f"Unexpected error for query '{query}': {e}")
            raise HTTPException(status_code=500, detail=f"Unexpected error for query '{query}': {e}")

    logger.info(f"Total high confidence DirtSearch results: {len(all_results)}")
    return all_results

def execute_bing_search_queries(queries: list) -> dict:
    """
    Executes all search queries using Bing Search API.

    :param queries: A list of search query strings.
    :return: A dictionary mapping each query to its Bing search results.
    """
    logger.info("Executing all search queries using Bing Search API.")
    search_results = {}
    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}

    for query in queries:
        logger.info(f"Searching for query: {query}")
        params = {
            "q": query,
            "textDecorations": True,
            "textFormat": "HTML",
            "freshness": "Month",
            "sortBy": "Date"
        }
        try:
            response = requests.get(BING_ENDPOINT, headers=headers, params=params, timeout=10)
            logger.debug(f"Bing Search Response Status Code for '{query}': {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Bing Search API error for query '{query}': {response.text}")
                raise HTTPException(status_code=500, detail=f"Bing Search API error for query '{query}': {response.text}")
            
            data = response.json()
            web_pages = data.get("webPages", {}).get("value", [])
            search_results[query] = web_pages
            logger.debug(f"Bing Search Results for '{query}': {web_pages}")
            
            time.sleep(1)  # To respect API rate limits

        except requests.exceptions.RequestException as e:
            logger.error(f"RequestException while contacting Bing Search API for query '{query}': {e}")
            raise HTTPException(status_code=500, detail=f"RequestException while contacting Bing Search API for query '{query}': {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError while parsing Bing Search results for query '{query}': {e}")
            raise HTTPException(status_code=500, detail=f"JSONDecodeError while parsing Bing Search results for query '{query}': {e}")
        except Exception as e:
            logger.error(f"Generic error while executing Bing Search for query '{query}': {e}")
            raise HTTPException(status_code=500, detail=f"Generic error while executing Bing Search for query '{query}': {e}")

    return search_results

def aggregate_dirtsearch_results(dirtsearch_results: list) -> str:
    """
    Aggregates DirtSearch results into a structured string for analysis.

    :param dirtsearch_results: A list of DirtSearch result dictionaries.
    :return: A formatted string containing aggregated DirtSearch information.
    """
    logger.info("Aggregating DirtSearch results.")
    aggregated_info = ""

    for item in dirtsearch_results:
        claim = item.get("claim", "No Claim")
        explanation = item.get("explanation", "No Explanation")
        url = item.get("url", "No URL")
        searchscore = item.get("searchscore", "No Score")
        match_confidence = item.get("matchConfidence", "No Confidence")
        
        aggregated_info += f"**Claim**: {claim}\n**Explanation**: {explanation}\n**URL**: {url}\n**Match Confidence**: {match_confidence}\n**Search Score**: {searchscore}\n\n"

    logger.debug(f"Aggregated DirtSearch Information: {aggregated_info}")
    return aggregated_info

def aggregate_bing_search_results(bing_search_results: dict) -> str:
    """
    Aggregates Bing Search results into a structured string for analysis.

    :param bing_search_results: A dictionary mapping queries to their Bing search results.
    :return: A formatted string containing aggregated Bing search information.
    """
    logger.info("Aggregating Bing Search results.")
    aggregated_info = ""

    for query, web_pages in bing_search_results.items():
        aggregated_info += f"### Results for query: {query} (Source: Bing)\n"
        if not web_pages:
            aggregated_info += "No results found.\n\n"
            continue
        for page in web_pages[:3]:  # Limit to top 3 results per query
            title = page.get("name", "No Title")
            url = page.get("url", "No URL")
            snippet = page.get("snippet", "No Snippet")
            aggregated_info += f"**Title**: {title}\n**URL**: {url}\n**Snippet**: {snippet}\n\n"
        aggregated_info += "\n"

    logger.debug(f"Aggregated Bing Search Information: {aggregated_info}")
    return aggregated_info

# ----------------------------
# API Endpoint
# ----------------------------
@app.head('/')
@app.get('/')
async def test():
    return {'msg': 'Hello World'}

@app.head('/process-text')
async def test():
    return {'msg': 'Hello World'}

@app.post("/process-text")
def process_text(input: InputText):
    return dirtsearch(input.text)

@app.get("/dirtsearch/{input}")
def dirtsearch(input: str):
    """
    Processes the input text by generating search queries, executing them using DirtSearch and Bing,
    aggregating results, and providing a final summarized response based on the aggregated search data.
    
    :param input: The input text containing the claim to be validated or explored.
    :return: A JSON object containing the final summarized result along with search results.
    """

    logger.info(f"Received input text: {input}")
    input_text = clean_text(input)
    logger.info(f"cleaned input text: {input_text}")

        # Generate cache key from the cleaned input text
    cache_key = get_cache_key(input_text)
    logger.info(f"Generated cache key: {cache_key}")

    # Check if response is in cache
    cached_response = get_from_cache(cache_key)
    if cached_response:
        logger.info("Cache hit for input text.")
        return cached_response

    logger.info("Cache miss. Processing the input text.")

    try:
        # Step 1: Generate search queries using Azure OpenAI with retry
        queries = generate_search_queries(input_text)
        
        if not queries:
            logger.info("No search queries generated. Returning response indicating no claims to validate.")
            return ProcessedTextResponse(
                source_used="None",
                search_results=[],
                result="No factual claims detected that require validation.",
                status="None"
            )
        
        # Step 2: Execute search queries using DirtSearch
        dirtsearch_results = execute_dirtsearch_queries(queries)
        
        if dirtsearch_results:
            logger.info("Using DirtSearch results for aggregation and final prompting.")
            aggregated_data = aggregate_dirtsearch_results(dirtsearch_results)
            source_used = "Microsoft Dirt Search"
            # Convert DirtSearch results to unified format
            search_result = convert_dirtsearch_response(dirtsearch_results)
            search_results_response = [search_result]
        else:
            logger.info("No high confidence results from DirtSearch. Falling back to Bing Search API.")
            # Step 3: Execute search queries using Bing Search API
            bing_search_results = execute_bing_search_queries(queries)
            aggregated_data = aggregate_bing_search_results(bing_search_results)
            source_used = "High Auth Bing Results"
            # Convert Bing Search results to unified format
            search_results_response = convert_bing_search_response(bing_search_results)
        
        # Step 4: Perform final prompting with aggregated data using Azure OpenAI with retry
        final_response = final_prompting(input_text, aggregated_data)
        final_status = extract_rating(final_response)
        
        logger.info("Successfully processed the input text.")
        response = ProcessedTextResponse(
            source_used=source_used,
            search_results=search_results_response,
            result=final_response,
            status=final_status
        )
        # Store the response in cache
        add_to_cache(cache_key, response)
        return response
    
    except HTTPException as http_exc:
        logger.error(f"HTTPException: {http_exc.detail}")
        raise http_exc  # Re-raise the HTTPException to be handled by FastAPI
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


def clean_text(text):
    """
    Cleans the input text by removing all special characters.
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")
    
    text = text.strip()
    
    if not text:
        return ""
    
    cleaned_text = regex.sub(r'[^\p{L}\p{N} ]+', '', text)
    
    return cleaned_text

def extract_rating(llm_response):
    """
    Extracts the Rating value from the LLM response text using the regex module.

    Parameters:
        llm_response (str): The text response from the LLM.

    Returns:
        str or None: The extracted Rating value if found, otherwise None.
    """
    # Define a regex pattern to match 'Rating: <value>'
    # This pattern is case-insensitive and accounts for possible whitespace variations
    # It also captures multi-word ratings like "Very High" or "3 Stars"
    pattern = regex.compile(
        r'Rating\s*:\s*([A-Za-z0-9\s]+)', 
        regex.IGNORECASE
    )

    # Search for the pattern in the response text
    match = pattern.search(llm_response)
    
    if match:
        # Strip any leading/trailing whitespace from the captured group
        rating = match.group(1).strip()
        return rating
    else:
        # If no match is found, return None or handle accordingly
        return None