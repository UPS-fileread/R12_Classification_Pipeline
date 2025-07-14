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
import re
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator, TypeAdapter
from typing import Union, Literal

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

# Dynamically construct a subcategory Enum for each Category
from enum import Enum as _Enum

SubcategoryEnumMap: dict[str, _Enum] = {}
for cat_name, subs in SUBCATEGORIES.items():
    enum_name = f"{cat_name}Subcategory"
    SubcategoryEnumMap[cat_name] = _Enum(enum_name, {s.replace(" ", ""): s for s in subs})

# Expose each subcategory enum for discriminated-union models
ContractSubcategory   = SubcategoryEnumMap["Contract"]
LitigationSubcategory = SubcategoryEnumMap["Litigation"]
RegulatorySubcategory = SubcategoryEnumMap["Regulatory"]
FinancialSubcategory  = SubcategoryEnumMap["Financial"]
StatutorySubcategory  = SubcategoryEnumMap["Statutory"]
EmailSubcategory      = SubcategoryEnumMap["Email"]
OtherSubcategory      = SubcategoryEnumMap["other"]

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


# Discriminated-union models for classification
class ContractClassification(BaseModel):
        category: Literal["Contract"] = Field(
            ...,
            title="Document category",
            description="Must be 'Contract' for contract-related documents"
        )
        subcategory: ContractSubcategory = Field(
            ...,
            title="Contract subcategory",
            description="Specific contract subcategory as defined in definitions.json"
        )
        summary: str = Field(
            ...,
            title="Summary",
            description="4–6 sentence summary for litigators, neutral and factual"
        )
        key_themes: list[str] = Field(
            ...,
            title="Key Themes",
            description="Exactly three key themes (5–15 words each)",
            min_items=3,
            max_items=3
        )

class LitigationClassification(BaseModel):
        category: Literal["Litigation"] = Field(
            ...,
            title="Document category",
            description="Must be 'Litigation' for litigation-related documents"
        )
        subcategory: LitigationSubcategory = Field(
            ...,
            title="Litigation subcategory",
            description="Specific litigation subcategory as defined in definitions.json"
        )
        summary: str = Field(
            ...,
            title="Summary",
            description="4–6 sentence summary for litigators, neutral and factual"
        )
        key_themes: list[str] = Field(
            ...,
            title="Key Themes",
            description="Exactly three key themes (5–15 words each)",
            min_items=3,
            max_items=3
        )

class RegulatoryClassification(BaseModel):
        category: Literal["Regulatory"] = Field(
            ...,
            title="Document category",
            description="Must be 'Regulatory' for regulatory-related documents"
        )
        subcategory: RegulatorySubcategory = Field(
            ...,
            title="Regulatory subcategory",
            description="Specific regulatory subcategory as defined in definitions.json"
        )
        summary: str = Field(
            ...,
            title="Summary",
            description="4–6 sentence summary for litigators, neutral and factual"
        )
        key_themes: list[str] = Field(
            ...,
            title="Key Themes",
            description="Exactly three key themes (5–15 words each)",
            min_items=3,
            max_items=3
        )

class FinancialClassification(BaseModel):
        category: Literal["Financial"] = Field(
            ...,
            title="Document category",
            description="Must be 'Financial' for financial-related documents"
        )
        subcategory: FinancialSubcategory = Field(
            ...,
            title="Financial subcategory",
            description="Specific financial subcategory as defined in definitions.json"
        )
        summary: str = Field(
            ...,
            title="Summary",
            description="4–6 sentence summary for litigators, neutral and factual"
        )
        key_themes: list[str] = Field(
            ...,
            title="Key Themes",
            description="Exactly three key themes (5–15 words each)",
            min_items=3,
            max_items=3
        )

class StatutoryClassification(BaseModel):
        category: Literal["Statutory"] = Field(
            ...,
            title="Document category",
            description="Must be 'Statutory' for statutory or legislative documents"
        )
        subcategory: StatutorySubcategory = Field(
            ...,
            title="Statutory subcategory",
            description="Specific statutory subcategory as defined in definitions.json"
        )
        summary: str = Field(
            ...,
            title="Summary",
            description="4–6 sentence summary for litigators, neutral and factual"
        )
        key_themes: list[str] = Field(
            ...,
            title="Key Themes",
            description="Exactly three key themes (5–15 words each)",
            min_items=3,
            max_items=3
        )

class EmailClassification(BaseModel):
        category: Literal["Email"] = Field(
            ...,
            title="Document category",
            description="Must be 'Email' for email communications"
        )
        subcategory: EmailSubcategory = Field(
            ...,
            title="Email subcategory",
            description="Specific email subcategory as defined in definitions.json"
        )
        summary: str = Field(
            ...,
            title="Summary",
            description="4–6 sentence summary for litigators, neutral and factual"
        )
        key_themes: list[str] = Field(
            ...,
            title="Key Themes",
            description="Exactly three key themes (5–15 words each)",
            min_items=3,
            max_items=3
        )

class OtherClassification(BaseModel):
        category: Literal["other"] = Field(
            ...,
            title="Document category",
            description="Must be 'other' for documents not covered by other categories"
        )
        subcategory: OtherSubcategory = Field(
            ...,
            title="Other subcategory",
            description="Specific other subcategory as defined in definitions.json"
        )
        summary: str = Field(
            ...,
            title="Summary",
            description="4–6 sentence summary for litigators, neutral and factual"
        )
        key_themes: list[str] = Field(
            ...,
            title="Key Themes",
            description="Exactly three key themes (5–15 words each)",
            min_items=3,
            max_items=3
        )

ClassificationResult = Union[
    ContractClassification,
    LitigationClassification,
    RegulatoryClassification,
    FinancialClassification,
    StatutoryClassification,
    EmailClassification,
    OtherClassification,
]

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
* Do NOT invent any new Category or Subcategory values.
* If the document fits none of the provided subcategories, default to:
    • category: "other"
    • subcategory: "other"

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
    # Call ChatCompletion and parse JSON response manually
    chat_resp = client.chat.completions.create(
        model=LANGUAGE_MODEL,
        messages=messages,
        response_format={"type": "json_object"},  # ensure well‑formed JSON
        temperature=0,
        max_tokens=1024
    )
    content = chat_resp.choices[0].message.content
    # NOTE: If "Trust Agreement" or other values are not recognized as valid subcategories,
    # you must add them to definitions.json under "subcategories" → "Contract".
    try:
        parsed = TypeAdapter(ClassificationResult).validate_json(content)
    except Exception:
        # Fallback for unrecognized subcategories: try to parse JSON, fixing minor formatting issues
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Attempt to insert missing commas between fields when keys run together
            fixed = re.sub(r'"\s*\n\s*"', r'",\n    "', content)
            data = json.loads(fixed)

        # Fallback for any mismatch: unrecognized category or subcategory
        cat_val = data.get("category")
        sub_val = data.get("subcategory")
        # If category is unknown, or subcategory is not valid under that category, route to other/other
        valid_subs = {member.value for member in SubcategoryEnumMap.get(cat_val, [])}
        if cat_val not in Category._value2member_map_ or sub_val not in valid_subs:
            return OtherClassification(
                category="other",
                subcategory=OtherSubcategory.other,
                summary=data.get("summary", ""),
                key_themes=data.get("key_themes", [])
            )

        # Propagate other unexpected cases
        raise RuntimeError(f"Failed to validate LLM response and no fallback available.\nResponse content: {content}")
    return parsed

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
        print(f"Subcategory: {result.subcategory.value}")
        print(f"Summary    : {result.summary}")
        print("Key Themes :")
        for idx, theme in enumerate(result.key_themes, 1):
            print(f"  {idx}. {theme}")
    except ValidationError as e:
        warnings.warn(f"Failed to validate LLM response: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()