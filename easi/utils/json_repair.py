"""JSON repair for LLM output. Ported from EmbodiedBench planner_utils.py."""
from __future__ import annotations

import re


def _replace_single_quotes(json_str: str) -> str:
    """Replace single quotes used as JSON delimiters with double quotes.

    Only replaces single quotes that appear to be JSON structural delimiters
    (around keys and values), not apostrophes inside English text.
    """
    # Replace single-quoted JSON keys/values: 'key' or 'value'
    # Matches: single quote, content without single quotes, single quote
    # followed by a JSON structural character (: , ] } or whitespace before those)
    result = []
    i = 0
    n = len(json_str)
    in_double_quote = False
    while i < n:
        ch = json_str[i]
        if ch == '"' and (i == 0 or json_str[i - 1] != '\\'):
            in_double_quote = not in_double_quote
            result.append(ch)
        elif ch == "'" and not in_double_quote:
            # Apostrophe if letter on BOTH sides (e.g., don't, it's, I'm)
            has_alpha_before = i > 0 and json_str[i - 1].isalpha()
            has_alpha_after = i + 1 < n and json_str[i + 1].isalpha()
            if has_alpha_before and has_alpha_after:
                result.append(ch)
            else:
                result.append('"')
        else:
            result.append(ch)
        i += 1
    return "".join(result)


def fix_json(json_str: str) -> str:
    """Fix common JSON errors in LLM output.

    Handles:
    - Single quotes used as JSON delimiters -> double quotes
    - Markdown code fences
    - Unescaped quotes inside reasoning_and_reflection value
    """
    # Replace single quotes used as JSON delimiters (not apostrophes)
    json_str = _replace_single_quotes(json_str)

    json_str = json_str.replace('```json', '').replace('```', '')

    # Fix unescaped double quotes inside reasoning_and_reflection value.
    # Pattern: match from the key's opening quote to just before "language_plan".
    pattern = r'("reasoning_and_reflection"\s*:\s*")(?P<value>.*?)(?=",\s*"language_plan")'

    def replacer(match):
        prefix = match.group(1)
        value = match.group("value")
        # Escape any double quote that is not already escaped.
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)
        return prefix + fixed_value

    json_str = re.sub(pattern, replacer, json_str, flags=re.DOTALL)
    return json_str
