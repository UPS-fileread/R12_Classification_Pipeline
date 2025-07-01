#!/usr/bin/env python3
"""
Classify and summarize legal documents using OpenAI LLM and Pydantic for schema validation.
"""
import argparse
import sys
import warnings
import os
from enum import Enum
from dotenv import load_dotenv
import json
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# Load OpenAI API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load classification definitions from JSON config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "definitions.json")
with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

CONTEXT_TYPES = cfg["context_types"]
SUBCATEGORIES  = cfg["subcategories"]
SUBCATEGORY_DEFINITIONS  = cfg["subcategory_definitions"]

# Model and categories
LANGUAGE_MODEL = 'gpt-4.1-2025-04-14'

class Category(Enum):
    """Legal context categories."""
    Contract   = 'Contract'
    Litigation = 'Litigation'
    Regulatory = 'Regulatory'
    Financial  = 'Financial'
    Statutory  = 'Statutory'
    Email      = 'Email'
    other      = 'other'

class ClassificationResult(BaseModel):
    """
    Pydantic model for classification results.
    """
    category: Category = Field(
        ...,
        description="One of the predefined legal-context categories",
        example="Regulatory"
    )
    subcategory: str = Field(
        ...,
        description="One of the predefined subcategories for the chosen category. Use the provided subcategory definitions to ensure the best fit.",
        example="Inspection Report"
    )
    summary: str = Field(
        ...,
        description="A brief summary of the document, written in 1-2 sentences. This should capture the main topic, parties, and purpose if possible."
    )
    key_themes: list[str] = Field(
        ...,
        description="A list of 3 concise bullet points highlighting the most important facts, events, obligations, or issues in the document. These should serve as an extended, detailed highlight of the document—covering key events, important parties, main legal or factual points, and anything a litigator would want to quickly grasp."
    )
    class Config:
        use_enum_values = True
        validate_assignment = True

def classify_context(text: str) -> ClassificationResult:
    """
    Classify and summarize the given document text using an LLM.
    Returns a ClassificationResult object.
    """
    # Build the prompt listing categories and their subcategories (with definitions)
    categories_fmt = '\n'.join(f" * {k}: {v}" for k, v in CONTEXT_TYPES.items())
    subcats_fmt = ''
    for cat in CONTEXT_TYPES:
        subcats_fmt += f"   - {cat}:\n"
        for sub in SUBCATEGORIES.get(cat, []):
            definition = SUBCATEGORY_DEFINITIONS.get(cat, {}).get(sub, "")
            if definition:
                subcats_fmt += f"       • {sub}: {definition}\n"
            else:
                subcats_fmt += f"       • {sub}\n"
    system_prompt = f"""
You are an AI assistant trained to **classify and summarize legal documents for litigation teams**.
You will receive the full text of a document (or an excerpt).  Your tasks, in strict order, are:

1. **Select exactly one Category** whose description best matches the document.  
2. **Select exactly one Subcategory** that best fits within the chosen Category.  
3. **Write a concise, factual Summary** of the document in **4–6 sentences**.  
   - Audience : litigators.  
   - Tone : neutral, third‑person, strictly factual.  
   - Content : state what the document is, its overall purpose, parties (if named), key dates, amounts, or procedural posture.  
   - **Do NOT** provide legal analysis, legal advice, or mention your own reasoning.  
4. **Provide 3 Key Themes** – the recurring ideas a litigator should remember.  
   Key themes in a legal document serve to:  
   - **Organize information** – clarify the structure of the case or argument.  
   - **Enhance persuasion** – frame facts to shape the reader’s understanding.  
   - **Leave a lasting impression** – lodge memorable concepts or phrases.  
   - **Convey a larger story** – knit individual facts into a cohesive narrative.  

   Your task: return **exactly three** short bullet strings (≈ 5–15 words each) that capture the document’s most important facts, events, obligations, deadlines, or issues.  
   * Stay strictly factual and neutral.  
   * Avoid legal analysis, advice, or speculation.  
   * Do not repeat the summary; elevate the *ideas* that unify the document.  
   * Each bullet should read like a takeaway a litigator might highlight in the margins.

**Available Categories** (must match exactly):  
{categories_fmt}

**Available Subcategories** (must match exactly):  
{subcats_fmt}

**Output format – return *only* this JSON object**  

{{
  "category": "<Category>",
  "subcategory": "<Subcategory>",
  "summary": "<4–6 sentence summary of the document>",
  "key_themes": ["<Theme 1>", "<Theme 2>", "<Theme 3>"]
}}

Do not output anything else—no markdown, explanations, or extra keys.  
Ensure *category* and *subcategory* values exactly match one of the provided options.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": text},
    ]

    # Make the API call to classify the text
    resp = client.responses.parse(
        model=LANGUAGE_MODEL,
        input=messages,  # pass messages directly as before
        text_format=ClassificationResult
    )

    # Will either return a valid ClassificationResult or raise ValidationError
    if resp.output_parsed is None:
        raise ValidationError("No valid response from LLM.")
    return resp.output_parsed

def main():
    """
    Command-line interface for classifying a TXT file into a legal context and subcategory.
    """
    parser = argparse.ArgumentParser(
        description="Classify a TXT file into a legal context and subcategory."
    )
    parser.add_argument('txt_file', help="Path to the TXT file to classify")
    args = parser.parse_args()

    # Read the input text file
    try:
        text = open(args.txt_file, 'r', encoding='utf-8').read()
    except Exception as e:
        print(f"Error reading '{args.txt_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Classify and print results
    try:
        result = classify_context(text)
        print(f"Category   : {result.category}")
        print(f"Subcategory: {result.subcategory}")
        print(f"Summary    : {result.summary}")
        print("Key Themes :")
        for idx, theme in enumerate(result.key_themes, 1):
            print(f"  {idx}. {theme}")
    except ValidationError as e:
        warnings.warn(f"Failed to validate LLM response: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()