import google.generativeai as genai
from groq import AsyncGroq  # Changed to AsyncGroq for proper async support
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from autocorrect import Speller
import cohere
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from urllib.parse import urlparse
from dotenv import load_dotenv
import os
import logging
from retry import retry
from nltk.translate.bleu_score import sentence_bleu
from typing import List, Dict, Tuple
import asyncio
import aiohttp
from functools import lru_cache
from datetime import datetime
import re
import hashlib
from ai_modules import PromptValidator, SourceVerifier, LogicalReasoner
import wikipedia  # Import Wikipedia
from duckduckgo_search import DDGS  # Import DDGS

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")  # Removed hardcoded fallback
cohere_api_key = os.getenv("COHERE_API_KEY")

# Validate environment variables
if not google_api_key:
    logger.error("Missing Google API key in environment variables")
    raise ValueError("Google API key must be set in .env file")
if not groq_api_key:
    logger.error("Missing Groq API key in environment variables")
    raise ValueError("Groq API key must be set in .env file")
if not cohere_api_key:
    logger.error("Missing Cohere API key in environment variables")
    raise ValueError("Cohere API key must be set in .env file")

# Initialize APIs
genai.configure(api_key=google_api_key)
groq_client = AsyncGroq(api_key=groq_api_key)  # Using AsyncGroq for async support
cohere_client = cohere.Client(cohere_api_key)

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {str(e)}")
    raise

# Preload Sentence Transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Loaded SentenceTransformer: all-MiniLM-L6-v2")
except Exception as e:
    logger.warning(f"Error loading all-MiniLM-L6-v2: {e}")
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        logger.info("Loaded fallback: paraphrase-MiniLM-L6-v2")
    except Exception as e2:
        logger.error(f"Error loading fallback model: {e2}")
        model = None

# Initialize KeyBERT and spell checker
kw_model = KeyBERT() if model else None
spell = Speller(lang='en')

# Cache for tokenized sentences
sentence_cache = {}

async def process_prompt(prompt: str) -> Dict:
    """Enhanced prompt processing with validation and source verification."""

    validator = PromptValidator()
    verifier = SourceVerifier()
    reasoner = LogicalReasoner()

    logger.debug(f"Processing prompt: {prompt}")
    validation_results = await validator.validate_prompt(prompt)
    # Only block for future events, not for missing entities
    if validation_results.get("error") == "future_event":
        logger.warning("Future event prompt detected")
        return {
            "error": "Future event not supported",
            "suggestions": [validation_results.get("suggestion", "Please ask about past or present events only.")],
            "confidence": validation_results.get("confidence", 1.0)
        }

    # Do NOT block for ambiguous or missing entities
    # if validation_results["ambiguous"]:
    #     logger.warning("Ambiguous prompt detected")
    #     return {
    #         "error": "Ambiguous prompt",
    #         "suggestions": validation_results["suggestions"],
    #         "entities": validation_results["entities"]
    #     }

    results = await get_processed_results(prompt, verifier)

    # Advanced Data Analysis Section
    advanced_analysis = []
    # Example: Response similarity (cosine similarity between model answers)
    try:
        google_resp = results.get('google_response', '')
        groq_resp = results.get('groq_response', '')
        cohere_resp = results.get('cohere_response', '')
        if google_resp and groq_resp and cohere_resp:
            emb_google = model.encode(google_resp, convert_to_tensor=True)
            emb_groq = model.encode(groq_resp, convert_to_tensor=True)
            emb_cohere = model.encode(cohere_resp, convert_to_tensor=True)
            sim_gg = float(util.pytorch_cos_sim(emb_google, emb_groq))
            sim_gc = float(util.pytorch_cos_sim(emb_google, emb_cohere))
            sim_gcq = float(util.pytorch_cos_sim(emb_groq, emb_cohere))
            advanced_analysis.append(f"Google vs Groq similarity: {sim_gg:.2f}")
            advanced_analysis.append(f"Google vs Cohere similarity: {sim_gc:.2f}")
            advanced_analysis.append(f"Groq vs Cohere similarity: {sim_gcq:.2f}")
    except Exception as e:
        logger.warning(f"Advanced analysis (similarity) failed: {e}")
    # Example: Keyword overlap
    try:
        if kw_model and google_resp and groq_resp and cohere_resp:
            kw_google = set(kw_model.extract_keywords(google_resp, top_n=5))
            kw_groq = set(kw_model.extract_keywords(groq_resp, top_n=5))
            kw_cohere = set(kw_model.extract_keywords(cohere_resp, top_n=5))
            overlap_gg = len(kw_google & kw_groq)
            overlap_gc = len(kw_google & kw_cohere)
            overlap_gcq = len(kw_groq & kw_cohere)
            advanced_analysis.append(f"Keyword overlap (Google/Groq): {overlap_gg}")
            advanced_analysis.append(f"Keyword overlap (Google/Cohere): {overlap_gc}")
            advanced_analysis.append(f"Keyword overlap (Groq/Cohere): {overlap_gcq}")
    except Exception as e:
        logger.warning(f"Advanced analysis (keywords) failed: {e}")
    # Example: Sentiment analysis (if available)
    # You can add more advanced analysis here
    results["advanced_analysis"] = advanced_analysis

    for api in ["Google", "Groq", "Cohere"]:
        consistency_score = results.get(f"{api.lower()}_consistency", {}).get("score", 0.0)
        if consistency_score < 0.3:
            reasoner.set_symbol(f"H_{api}", True)

    results["logical_inferences"] = reasoner.forward_chaining()

    # Prepare additional insights based on logical inferences
    additional_insights = []
    if "Google response is likely hallucinated" in results["logical_inferences"]:
        additional_insights.append("Google response is likely hallucinated.")
    if "Groq response is likely hallucinated" in results["logical_inferences"]:
        additional_insights.append("Groq response is likely hallucinated.")
    if "Cohere response is likely hallucinated" in results["logical_inferences"]:
        additional_insights.append("Cohere response is likely hallucinated.")

    results["additional_insights"] = additional_insights

    # Ensure all required keys are present
    results.update({
        "wiki_url": results.get("wiki_url", ""),
        "wiki_reliability": results.get("wiki_reliability", 0.0),
        "ddg_url": results.get("ddg_url", ""),
        "ddg_reliability": results.get("ddg_reliability", 0.0),
        "google_url": results.get("google_url", ""),
        "google_reliability": results.get("google_reliability", 0.0),
        "confidence_google": results.get("confidence_google", 0.0),
        "confidence_groq": results.get("confidence_groq", 0.0),
        "confidence_cohere": results.get("confidence_cohere", 0.0),
        "hallucination_google": results.get("hallucination_google", 0.0),
        "hallucination_groq": results.get("hallucination_groq", 0.0),
        "hallucination_cohere": results.get("hallucination_cohere", 0.0),
        "prompt_complexity": {"warning": results.get("prompt_complexity", {}).get("warning", "")},
        "ensemble_warnings": results.get("ensemble_warnings", []),
        "best_api": results.get("best_api", ""),
        "best_answer": results.get("best_answer", ""),
        "best_hallucination": results.get("best_hallucination", 0.0),
        "hallucinated_api": results.get("hallucinated_api", ""),
        "hallucinated_answer": results.get("hallucinated_answer", ""),
        "correct_answer": results.get("correct_answer", ""),
        "explanation": results.get("explanation", ""),
        "google_marked_response": results.get("google_marked_response", ""),
        "groq_marked_response": results.get("groq_marked_response", ""),
        "cohere_marked_response": results.get("cohere_marked_response", "")
    })

    logger.debug(f"Final results: {results}")
    return results

@lru_cache(maxsize=100)
async def get_processed_results(prompt: str, verifier: SourceVerifier) -> Dict:
    apis = ["google", "groq", "cohere"]
    responses = {}

    logger.debug("Fetching API responses")
    tasks = [
        get_google_response(prompt),
        get_groq_response(prompt),
        get_cohere_response(prompt)
    ]

    try:
        api_responses = await asyncio.gather(*tasks, return_exceptions=True)
        for api, response in zip(apis, api_responses):
            if isinstance(response, Exception):
                logger.error(f"{api.capitalize()} API failed: {str(response)}")
                responses[f"{api}_response"] = ""
            else:
                responses[f"{api}_response"] = response or ""

        logger.debug("Fetching external references")
        external_data = await get_external_reference(prompt)
        for source in external_data["sources"]:
            source["trust_score"] = verifier.calculate_trust_score(
                source["url"], source["content"])

        logger.debug("Processing responses")
        results = process_responses(prompt, responses, external_data)

        for api in apis:
            response = responses.get(f"{api}_response", "")
            flagged_sentences = results.get(f"{api}_consistency", {}).get("flagged_sentences", [])
            marked_response = mark_hallucinated_sentences(response, flagged_sentences)
            results[f"{api}_marked_response"] = marked_response

        return results
    except Exception as e:
        logger.error(f"Error processing API responses: {str(e)}", exc_info=True)
        return {"error": f"API processing failed: {str(e)}"}

def mark_hallucinated_sentences(text: str, flagged_sentences: List[Tuple[str, float]]) -> str:
    """Mark hallucinated sentences with HTML spans for UI highlighting."""
    marked_text = text or ""
    for sentence, confidence in flagged_sentences:
        if confidence < 0.3:
            color = "red"
            tooltip = "High likelihood of hallucination"
        elif confidence < 0.6:
            color = "yellow"
            tooltip = "Moderate likelihood of hallucination"
        else:
            continue

        marked_sentence = f'<span class="hallucinated-text" style="background-color: {color}; padding: 2px;" title="{tooltip}">{sentence}</span>'
        marked_text = marked_text.replace(sentence, marked_sentence)

    return marked_text

@retry(tries=2, delay=0.5)
async def get_google_response(prompt: str) -> str:
    """Get response from Google's Gemini API."""
    try:
        model = genai.GenerativeModel(model_name='gemini-2.0-flash-thinking-exp-1219')
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        # Fix: Using async-compatible approach for Google API
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(
                prompt,
                safety_settings=safety_settings
            )
        )
        return response.text if response and hasattr(response, 'text') else ""
    except Exception as e:
        logger.error(f"Google API error: {str(e)}")
        return ""

@retry(tries=2, delay=1.0)  # Increased delay for retry
async def get_groq_response(prompt: str) -> str:
    """Get response from Groq API using AsyncGroq client."""
    try:
        response = await groq_client.chat.completions.create(
            model="llama3-70b-8192",  # Use correct model name for Groq
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,  # Set for creative thinking
            max_tokens=512
        )
        logger.debug(f"Groq API raw response: {response}")
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content.strip()
        else:
            logger.error("Groq API returned no choices or unexpected format.")
            return "No response from Groq API."
    except Exception as e:
        logger.error(f"Groq API error: {str(e)}")
        return f"Groq API error: {str(e)}"

@retry(tries=2, delay=0.5)
async def get_cohere_response(prompt: str) -> str:
    """Get response from Cohere API."""
    try:
        loop = asyncio.get_event_loop()
        # Using synchronous client in an executor as a workaround
        response = await loop.run_in_executor(
            None,
            lambda: cohere_client.generate(
                prompt=prompt,
                max_tokens=512,
                temperature=0.75,  # Set for creative thinking
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE'
            )
        )
        return response.generations[0].text.strip() if response.generations else ""
    except Exception as e:
        logger.error(f"Cohere API error: {str(e)}")
        return ""

async def fetch_url(url: str) -> str:
    """Fetch content from URL with timeout and error handling."""
    try:
        async with aiohttp.ClientSession() as session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            try:
                async with session.get(url, timeout=10, headers=headers) as response:
                    if response.status == 200:
                        try:
                            text = await response.text()
                            soup = BeautifulSoup(text, 'html.parser')
                            for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                                element.decompose()
                            content = ' '.join(soup.stripped_strings)
                            return content[:2000]
                        except Exception as content_err:
                            logger.error(f"Error processing content from {url}: {str(content_err)}")
                            return ""
                    else:
                        logger.warning(f"Received status code {response.status} from {url}")
                        return ""
            except aiohttp.ClientError as client_err:
                logger.error(f"Client error fetching {url}: {str(client_err)}")
                return ""
            except asyncio.TimeoutError:
                logger.error(f"Timeout fetching {url}")
                return ""
    except Exception as e:
        logger.error(f"General error fetching URL {url}: {str(e)}")
        return ""

async def get_external_reference(prompt: str) -> Dict:
    """Get external references for fact checking, prioritizing diverse URLs."""
    sources = []
    wiki_url = ""
    wiki_reliability = 0.0
    ddg_url = ""
    ddg_reliability = 0.0
    google_url = ""
    google_reliability = 0.0

    # Wikipedia search
    try:
        loop = asyncio.get_event_loop()

        def wiki_search_and_get():
            try:
                wiki_results = wikipedia.search(prompt, results=1)
                if wiki_results:
                    try:
                        page = wikipedia.page(wiki_results[0], auto_suggest=False)
                        return {
                            "url": page.url,
                            "content": page.content[:5000],  # Limit content size
                            "type": "wikipedia"
                        }
                    except (wikipedia.DisambiguationError, wikipedia.PageError) as e:
                        logger.warning(f"Wikipedia error for {wiki_results[0]}: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Wikipedia search error: {str(e)}")
                return None

        wiki_data = await loop.run_in_executor(None, wiki_search_and_get)
        if wiki_data:
            wiki_url = wiki_data["url"]
            wiki_reliability = 0.6
            sources.append(wiki_data)
    except Exception as e:
        logger.error(f"Wikipedia error: {str(e)}")

    # DuckDuckGo search
    try:
        def ddg_search():
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(prompt, max_results=3))
                    return results
            except Exception as e:
                logger.error(f"DuckDuckGo search error: {str(e)}")
                return []

        ddg_results = await asyncio.get_event_loop().run_in_executor(None, ddg_search)

        for result in ddg_results:
            if result and 'href' in result and result['href'] != wiki_url:  # Check for uniqueness
                url = result['href']
                content = await fetch_url(url)
                if content:
                    ddg_url = url
                    ddg_reliability = 0.5
                    sources.append({
                        "url": url,
                        "content": content,
                        "type": "web"
                    })
                    break
    except Exception as e:
        logger.error(f"DuckDuckGo processing error: {str(e)}")

    # Google search
    try:
        def google_search():
            try:
                from googlesearch import search
                results = list(search(prompt, num_results=3))
                return results
            except Exception as e:
                logger.error(f"Google search error: {str(e)}")
                return []

        google_results = await asyncio.get_event_loop().run_in_executor(None, google_search)

        for url in google_results:
            if url and url != wiki_url and url != ddg_url:  # Check for uniqueness
                content = await fetch_url(url)
                if content:
                    google_url = url
                    google_reliability = 0.5
                    sources.append({
                        "url": url,
                        "content": content,
                        "type": "web"
                    })
                    break
    except Exception as e:
        logger.error(f"Google processing error: {str(e)}")

    # If no unique URLs were found, add the first available URL from each source
    if not ddg_url and ddg_results and ddg_results[0] and 'href' in ddg_results[0]:
        ddg_url = ddg_results[0]['href']
        content = await fetch_url(ddg_url)
        if content:
            ddg_reliability = 0.5
            sources.append({
                "url": ddg_url,
                "content": content,
                "type": "web"
            })
    if not google_url and google_results:
        url = google_results[0]
        content = await fetch_url(url)
        if content:
            google_url = url
            google_reliability = 0.5
            sources.append({
                "url": url,
                "content": content,
                "type": "web"
            })

    # Ensure at least one source
    if not sources:
        sources.append({
            "url": "",
            "content": "",
            "type": "none",
            "trust_score": 0.0
        })

    return {
        "sources": sources,
        "wiki_url": wiki_url,
        "wiki_reliability": wiki_reliability,
        "ddg_url": ddg_url,
        "ddg_reliability": ddg_reliability,
        "google_url": google_url,
        "google_reliability": google_reliability,
        "timestamp": datetime.now().isoformat()
    }

def process_responses(prompt: str, responses: Dict, external_data: Dict) -> Dict:
    """Process and analyze responses for hallucinations."""
    results = {
        "best_api": "",
        "best_answer": "",
        "best_hallucination": 0.0,
        "hallucinated_api": "",
        "hallucinated_answer": "",
        "correct_answer": "",
        "explanation": "",
        "error": None,
        "suggestions": [],
        "wiki_url": external_data.get("wiki_url", ""),
        "wiki_reliability": external_data.get("wiki_reliability", 0.0),
        "ddg_url": external_data.get("ddg_url", ""),
        "ddg_reliability": external_data.get("ddg_reliability", 0.0),
        "google_url": external_data.get("google_url", ""),
        "google_reliability": external_data.get("google_reliability", 0.0),
        "confidence_google": 0.0,
        "confidence_groq": 0.0,
        "confidence_cohere": 0.0,
        "hallucination_google": 0.0,
        "hallucination_groq": 0.0,
        "hallucination_cohere": 0.0
    }

    apis = ["google", "groq", "cohere"]

    # Pre-tokenize external source content
    source_sentences = {}
    for source in external_data["sources"]:
        content = source.get("content", "")
        if content:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash not in sentence_cache:
                sentence_cache[content_hash] = sent_tokenize(content)
            source_sentences[source["url"]] = sentence_cache[content_hash]

    # Populate external reference URLs
    for source in external_data["sources"]:
        if source["type"] == "wikipedia":
            results["wiki_url"] = source["url"]
            results["wiki_reliability"] = source.get("trust_score", 0.0)
        elif source["type"] == "web":
            if not results["ddg_url"]:
                results["ddg_url"] = source["url"]
                results["ddg_reliability"] = source.get("trust_score", 0.0)
            else:
                results["google_url"] = source["url"]
                results["google_reliability"] = source.get("trust_score", 0.0)

    for api in apis:
        response = responses.get(f"{api}_response", "")
        results[f"{api}_consistency"] = {
            "score": 0.0,
            "flagged_sentences": []
        }

        if not response:
            logger.warning(f"{api.capitalize()} API returned an empty response.")
            continue

        response_hash = hashlib.md5(response.encode()).hexdigest()
        if response_hash not in sentence_cache:
            sentence_cache[response_hash] = sent_tokenize(response)
        sentences = sentence_cache[response_hash]

        if not sentences:
            logger.warning(f"No sentences found in {api.capitalize()} response.")
            continue

        flagged_sentences = []
        if model:
            try:
                sentence_embeddings = model.encode(sentences, batch_size=32)
                for idx, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
                    max_similarity = 0.0
                    for source in external_data["sources"]:
                        if source.get("trust_score", 0.0) < 0.5:
                            continue
                        src_sentences = source_sentences.get(source["url"], [])
                        if not src_sentences:
                            continue
                        src_embeddings = model.encode(src_sentences, batch_size=32)
                        if len(src_embeddings) > 0:
                            similarities = util.cos_sim(embedding, src_embeddings)
                            max_similarity = max(max_similarity, float(similarities.max()))
                    flagged_sentences.append((sentence, max_similarity))
            except Exception as e:
                logger.error(f"Error in sentence comparison: {str(e)}")
                flagged_sentences = [(s, 0.0) for s in sentences]
        else:
            for sentence in sentences:
                max_similarity = 0.0
                for source in external_data["sources"]:
                    if source.get("trust_score", 0.0) < 0.5:
                        continue
                    src_sentences = source_sentences.get(source["url"], [])
                    if not src_sentences:
                        continue
                    try:
                        bleu_score = sentence_bleu([s.split() for s in src_sentences],
                                                 sentence.split())
                        max_similarity = max(max_similarity, bleu_score)
                    except Exception as e:
                        logger.error(f"Error in BLEU score calculation: {str(e)}")
                        continue
                flagged_sentences.append((sentence, max_similarity))

        consistency_score = sum(score for _, score in flagged_sentences) / len(flagged_sentences) if flagged_sentences else 0.0

        results[f"{api}_consistency"] = {
            "score": consistency_score,
            "flagged_sentences": flagged_sentences
        }

    api_scores = [(api, results[f"{api}_consistency"]["score"])
                 for api in apis
                 if f"{api}_consistency" in results]

    if api_scores:
        best_api = max(api_scores, key=lambda x: x[1])[0]
        worst_api = min(api_scores, key=lambda x: x[1])[0]

        results.update({
            "best_api": best_api,
            "best_answer": responses.get(f"{best_api}_response", ""),
            "best_hallucination": results[f"{best_api}_consistency"]["score"],
            "hallucinated_api": worst_api,
            "hallucinated_answer": responses.get(f"{worst_api}_response", ""),
            "correct_answer": responses.get(f"{best_api}_response", ""),
            "explanation": "Based on comparison with verified sources",
            "confidence_google": results["google_consistency"]["score"],
            "confidence_groq": results["groq_consistency"]["score"],
            "confidence_cohere": results["cohere_consistency"]["score"],
            "hallucination_google": 1.0 - results["google_consistency"]["score"],
            "hallucination_groq": 1.0 - results["groq_consistency"]["score"],
            "hallucination_cohere": 1.0 - results["cohere_consistency"]["score"]
        })
    else:
        results.update({
            "error": "No valid responses from APIs",
            "suggestions": ["Please try again with a different prompt"]
        })

    logger.debug(f"Processed results: {results}")
    return results