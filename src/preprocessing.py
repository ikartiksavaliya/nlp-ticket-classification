"""
Preprocessing module for Customer Support Ticket Classification.
Contains functions for text cleaning, normalization, tokenization, stemming,
lemmatization, emoji/URL/email handling, spelling correction, and language detection.
"""

import re
import unicodedata
from typing import List, Tuple, Union
import pandas as pd

# Global variable to cache the SpaCy model
_nlp = None

def detect_language(text: str) -> str:
    """
    Detect the primary language of the text.
    Uses langdetect to identify language code (e.g., 'en', 'fr').
    
    Args:
        text: Input raw text string.
        
    Returns:
        Two-letter ISO language code (e.g., 'en').
    """
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"

def normalize_unicode(text: str) -> str:
    """
    Apply Unicode normalization (NFKD) to decompose accented characters.
    
    Args:
        text: Input string.
        
    Returns:
        Unicode normalized string.
    """
    if not isinstance(text, str):
        return ""
    # Decompose accented characters and filter out combining marks (accents)
    # This preserves non-ASCII characters like emojis
    return "".join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

def handle_contractions(text: str) -> str:
    """
    Expand contractions in text (e.g., "don't" -> "do not", "I'm" -> "I am").
    
    Args:
        text: Input string.
        
    Returns:
        String with expanded contractions.
    """
    if not isinstance(text, str):
        return ""
    import contractions
    return contractions.fix(text)

def handle_emojis(text: str) -> Tuple[str, List[str]]:
    """
    Extract, convert, or remove emojis and emoticons.
    Optionally replaces them with their text equivalents (e.g., :smile: -> " smile ").
    
    Args:
        text: Input string.
        
    Returns:
        A tuple of (cleaned_text, list_of_extracted_emojis).
    """
    if not isinstance(text, str):
        return "", []
    import emoji
    
    # Extract list of emojis
    extracted = [item['emoji'] for item in emoji.emoji_list(text)]
    
    # Convert emojis to text descriptors
    demojized = emoji.demojize(text)
    
    # Clean the output (e.g. :smile_face: -> " smile_face ")
    cleaned_text = re.sub(r':([a-zA-Z0-9_-]+):', r' \1 ', demojized)
    
    return cleaned_text, extracted

def mask_entities(text: str) -> str:
    """
    Locate and mask URLs, emails, and mentions with standardized tokens.
    URLs -> <URL>
    Emails -> <EMAIL>
    Mentions -> <MENTION>
    
    Args:
        text: Input string.
        
    Returns:
        String with masked entities.
    """
    if not isinstance(text, str):
        return ""
        
    # Mask URLs
    url_pattern = r'https?://\S+|www\.\S+'
    text = re.sub(url_pattern, '<URL>', text)
    
    # Mask Emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    text = re.sub(email_pattern, '<EMAIL>', text)
    
    # Mask Mentions
    mention_pattern = r'@\w+'
    text = re.sub(mention_pattern, '<MENTION>', text)
    
    return text

def remove_punctuation_and_numbers(text: str) -> str:
    """
    Remove punctuation and numeric characters from the text.
    Preserves <URL>, <EMAIL>, and <MENTION> mask tokens.
    
    Args:
        text: Input string.
        
    Returns:
        Cleaned text with only alphabetic characters and whitespace.
    """
    if not isinstance(text, str):
        return ""
        
    # Temporarily hide the mask tokens in a unique alphabetic format
    text = text.replace("<URL>", " URLMASK ").replace("<EMAIL>", " EMAILMASK ").replace("<MENTION>", " MENTIONMASK ")
    
    # Remove all punctuation and numbers, keeping underscores for emoji text
    text = re.sub(r'[^a-zA-Z_\s]', ' ', text)
    
    # Restore the standard mask tokens
    text = text.replace("URLMASK", "<URL>").replace("EMAILMASK", "<EMAIL>").replace("MENTIONMASK", "<MENTION>")
    
    return text

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace by removing leading/trailing spaces and collapsing
    multiple spaces into a single space.
    
    Args:
        text: Input string.
        
    Returns:
        Whitespace-normalized string.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize raw text into a list of sentences.
    
    Args:
        text: Input text string.
        
    Returns:
        List of sentence strings.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    import nltk
    return nltk.sent_tokenize(text)

def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text/sentence into a list of word tokens.
    
    Args:
        text: Input text string or sentence.
        
    Returns:
        List of word tokens.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    import nltk
    return nltk.word_tokenize(text)

def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Filter out common stopwords from a list of tokens.
    
    Args:
        tokens: List of word tokens.
        
    Returns:
        Filtered list of tokens.
    """
    if not tokens:
        return []
    import nltk
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # Make sure we don't accidentally drop the entity mask tokens or placeholders
    masks = {"<URL>", "<EMAIL>", "<MENTION>", "URLMASK", "EMAILMASK", "MENTIONMASK"}
    return [t for t in tokens if t.lower() not in stop_words or t.upper() in masks]

def apply_stemming(tokens: List[str]) -> List[str]:
    """
    Apply Porter stemming to a list of tokens.
    
    Args:
        tokens: List of word tokens.
        
    Returns:
        List of stemmed tokens.
    """
    if not tokens:
        return []
    import nltk
    stemmer = nltk.stem.PorterStemmer()
    masks = {"<URL>", "<EMAIL>", "<MENTION>", "URLMASK", "EMAILMASK", "MENTIONMASK"}
    return [stemmer.stem(t) if t.upper() not in masks else t for t in tokens]

def apply_lemmatization(tokens: List[str]) -> List[str]:
    """
    Apply POS-aware lemmatization using SpaCy.
    
    Args:
        tokens: List of word tokens.
        
    Returns:
        List of lemmatized tokens.
    """
    if not tokens:
        return []
    global _nlp
    import spacy
    from src.config import SPACY_MODEL
    
    # Lazy load SpaCy pipeline
    if _nlp is None:
        try:
            _nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
        except Exception:
            import spacy.cli
            spacy.cli.download(SPACY_MODEL)
            _nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
            
    # Map any <URL>, <EMAIL>, <MENTION> format to alphabetic placeholders for SpaCy
    processed_tokens = []
    for t in tokens:
        if t == "<URL>":
            processed_tokens.append("URLMASK")
        elif t == "<EMAIL>":
            processed_tokens.append("EMAILMASK")
        elif t == "<MENTION>":
            processed_tokens.append("MENTIONMASK")
        else:
            processed_tokens.append(t)
            
    # Join tokens into a sentence
    temp_sentence = " ".join(processed_tokens)
    doc = _nlp(temp_sentence)
    
    lemmas = []
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma == "urlmask":
            lemmas.append("<URL>")
        elif lemma == "emailmask":
            lemmas.append("<EMAIL>")
        elif lemma == "mentionmask":
            lemmas.append("<MENTION>")
        else:
            lemmas.append(lemma)
            
    return lemmas

def handle_spelling_variations(tokens: List[str]) -> List[str]:
    """
    Correct common spelling variations by collapsing character repetitions
    of 3 or more times down to 2 times (e.g., "coooool" -> "cool").
    
    Args:
        tokens: List of word tokens.
        
    Returns:
        List of spelling-corrected tokens.
    """
    if not tokens:
        return []
    return [re.sub(r'(.)\1{2,}', r'\1\1', t) for t in tokens]

def preprocess_text(text: str, method: str = "lemmatize") -> str:
    """
    Main orchestrator function that runs a raw text string through the full cleaning pipeline.
    Pipeline flow:
        Raw Text -> Unicode Normalization -> Lowercasing
        -> Mask URLs/Emails/Mentions -> Expand Contractions -> Handle Emojis
        -> Remove Punctuation/Numbers -> Whitespace Normalization -> Word Tokenization
        -> Remove Stopwords -> Spelling Correction -> Lemmatization/Stemming -> Cleaned String.
        
    Args:
        text: Raw issue description.
        method: Vector reduction technique ('lemmatize', 'stem', or 'none').
        
    Returns:
        A clean, space-separated string of tokens ready for representation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
        
    # 1. Unicode Normalization
    text = normalize_unicode(text)
    
    # 2. Lowercasing
    text = text.lower()
    
    # 3. Expand Contractions
    text = handle_contractions(text)
    
    # 4. Mask Entities
    text = mask_entities(text)
    
    # 5. Handle Emojis
    text, _ = handle_emojis(text)
    
    # 6. Remove Punctuation and Numbers (while preserving masks)
    text = remove_punctuation_and_numbers(text)
    
    # 7. Normalize Whitespace
    text = normalize_whitespace(text)
    
    # 8. Replace masks with safe alphabetic placeholders before tokenization
    text = text.replace("<URL>", "URLMASK").replace("<EMAIL>", "EMAILMASK").replace("<MENTION>", "MENTIONMASK")
    
    # 9. Tokenize Words
    tokens = tokenize_words(text)
    
    # 10. Remove Stopwords
    tokens = remove_stopwords(tokens)
    
    # 11. Handle Spelling Variations
    tokens = handle_spelling_variations(tokens)
    
    # 12. Reduce words to roots / lemmas
    if method == "lemmatize":
        tokens = apply_lemmatization(tokens)
    elif method == "stem":
        tokens = apply_stemming(tokens)
        
    # 13. Reassemble tokens and restore mask formatting
    cleaned_tokens = []
    for t in tokens:
        t_clean = t.strip()
        if not t_clean:
            continue
        if t_clean.upper() == "URLMASK":
            cleaned_tokens.append("<URL>")
        elif t_clean.upper() == "EMAILMASK":
            cleaned_tokens.append("<EMAIL>")
        elif t_clean.upper() == "MENTIONMASK":
            cleaned_tokens.append("<MENTION>")
        else:
            cleaned_tokens.append(t_clean)
            
    return " ".join(cleaned_tokens)

def preprocess_dataframe(df: pd.DataFrame, text_col: str, target_col: str) -> pd.DataFrame:
    """
    Preprocess an entire dataframe containing text and labels.
    Performs data cleaning, drops rows with nulls, and runs preprocess_text.
    
    Args:
        df: Input raw pandas DataFrame.
        text_col: Column name containing raw text.
        target_col: Column name containing categories.
        
    Returns:
        Preprocessed DataFrame containing cleaned text and encoded target.
    """
    # Drop rows where target or text is missing
    df_clean = df.dropna(subset=[text_col, target_col]).copy()
    
    from tqdm import tqdm
    tqdm.pandas(desc="Preprocessing text column")
    
    df_clean[f"cleaned_{text_col}"] = df_clean[text_col].progress_apply(preprocess_text)
    
    return df_clean
