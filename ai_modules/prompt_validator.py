from datetime import datetime
from dateutil import parser
import re
import wikipedia
from duckduckgo_search import DDGS
from googleapiclient.discovery import build
from typing import List, Dict, Tuple, Optional
import os
from dotenv import load_dotenv
import logging
from functools import lru_cache
import asyncio
import warnings
from bs4 import BeautifulSoup

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
google_fact_check_api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

# Initialize Google Fact Check API
try:
    fact_check_service = build('factchecktools', 'v1alpha1',
        developerKey=google_fact_check_api_key,
        cache_discovery=False)
except Exception as e:
    logger.error(f"Failed to initialize Google Fact Check API: {str(e)}")
    fact_check_service = None

class PromptValidator:
    """Validates prompts for real-world entities and facts."""

    def __init__(self):
        self.current_date = datetime.now()
        self.event_keywords = ['win', 'won', 'happen', 'occurred', 'take place', 'result']
        self.cache = lru_cache(maxsize=100)(self.validate_entity)
        self.interrogatives = ['Who', 'What', 'Where', 'When', 'Why', 'How']

    def extract_date(self, text: str) -> Optional[datetime]:
        """Extract and validate date from text."""
        date_patterns = [
            r'\b(19|20)\d{2}\b',  # Years
            r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s+)?(19|20)\d{2}\b'
        ]

        for pattern in date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    date = parser.parse(match.group(), fuzzy=True)
                    return date
                except ValueError:
                    continue
        return None

    def is_future_event(self, prompt: str) -> Tuple[bool, str]:
        """Check if prompt refers to a future event."""
        date = self.extract_date(prompt)
        if date and date > self.current_date:
            if any(keyword in prompt.lower() for keyword in self.event_keywords):
                return True, f"This event will occur in the future ({date.strftime('%Y')})"
        return False, ""

    async def validate_entity(self, entity: str) -> Dict:
        """Check if an entity exists using multiple sources."""
        results = {
            "exists": False,
            "confidence": 0.0,
            "source": None,
            "disambiguation": []
        }

        logger.debug(f"Validating entity: {entity}")
        # Check Wikipedia
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                wiki_page = wikipedia.page(entity, auto_suggest=True)
                for warning in caught_warnings:
                    if "No parser was explicitly specified" in str(warning.message):
                        logger.warning("BeautifulSoup parser warning caught. Consider specifying 'features=lxml'.")
                results["exists"] = True
                results["confidence"] = 0.8
                results["source"] = "Wikipedia"
                return results
        except wikipedia.DisambiguationError as e:
            results["disambiguation"] = e.options[:3]
            results["exists"] = True
            results["confidence"] = 0.5
            results["source"] = "Wikipedia (disambiguation)"
            return results
        except wikipedia.PageError:
            pass

        # Check Google Fact Check API
        if fact_check_service:
            try:
                query = fact_check_service.claims().search(query=entity).execute()
                if query.get('claims'):
                    results["exists"] = True
                    results["confidence"] = 0.7
                    results["source"] = "Google Fact Check"
                    return results
            except Exception as e:
                logger.debug(f"Google Fact Check failed for {entity}: {str(e)}")

        # DuckDuckGo fallback
        try:
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(entity, max_results=1))
                if ddg_results:
                    results["exists"] = True
                    results["confidence"] = 0.6
                    results["source"] = "DuckDuckGo"
                    return results
        except Exception as e:
            logger.debug(f"DuckDuckGo failed for {entity}: {str(e)}")

        return results

    def extract_entities(self, prompt: str) -> List[str]:
        """Extract potential entities from prompt."""
        patterns = [
            r'"([^"]+)"',  # Quoted strings
            r'by ([A-Z][a-z]+ [A-Z][a-z]+)',  # Author names
            r'(?:the |a |an )?([A-Z][a-z]+ (?:[A-Z][a-z]+ )?(?:Book|Novel|Movie|Film|Song|Album))',  # Titles
            r'(?:in |at |from )?([A-Z][a-z]+ (?:University|College|Institute|School))',  # Institutions
            r'(?:in |at |from )?([A-Z][a-z]+ (?:[A-Z][a-z]+ )?(?:Country|City|State|Province))',  # Locations
            r'([A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)?)',  # Proper nouns (2-3 words)
            r'([A-Z][a-z]+)'  # Single proper nouns
        ]

        entities = []
        for pattern in patterns:
            matches = re.finditer(pattern, prompt)
            entities.extend(match.group(1) for match in matches if match.group(1) not in self.interrogatives)

        # Enhanced keyword-based entity recognition for questions
        if any(prompt.startswith(q) for q in self.interrogatives):
            keywords = ['telephone', 'invented', 'invent']  # Refined keywords
            entities.extend(keyword for keyword in keywords if keyword in prompt.lower() and keyword not in self.interrogatives)

        return list(set(entities))

    async def validate_prompt(self, prompt: str) -> Dict:
        """Validate prompt and return validation results."""
        # Check for future events first
        is_future, future_message = self.is_future_event(prompt)
        if is_future:
            return {
                "is_valid": False,
                "confidence": 1.0,
                "error": "future_event",
                "message": future_message,
                "suggestion": "Please ask about past or present events only."
            }

        # Extract and validate entities
        entities = self.extract_entities(prompt)
        results = {
            "is_valid": True,  # Always allow prompt
            "confidence": 1.0,
            "ambiguous": False,
            "entities": [],
            "suggestions": []
        }

        logger.debug(f"Extracted entities: {entities}")
        entity_validations = []
        if entities:
            tasks = [self.cache(entity) for entity in entities]
            validations = await asyncio.gather(*tasks)

            for entity, validation in zip(entities, validations):
                entity_validations.append({
                    "entity": entity,
                    "validation": validation
                })

                if validation["disambiguation"]:
                    results["ambiguous"] = True
                    disambiguation_options = validation['disambiguation'][:3]
                    if disambiguation_options:
                        results["suggestions"].append(f"Did you mean one of these: {', '.join(disambiguation_options)}?")
                    else:
                        results["suggestions"].append(f"The entity '{entity}' is ambiguous, but no specific suggestions are available.")
                elif not validation["exists"]:
                    results["suggestions"].append(f"The entity '{entity}' could not be found. Please verify the spelling or try a different term.")

        results["entities"] = entity_validations
        # Never set is_valid to False for missing entities
        return results