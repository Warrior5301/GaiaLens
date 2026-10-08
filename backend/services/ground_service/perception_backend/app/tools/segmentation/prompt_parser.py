"""Multi-target natural language prompt decomposition for remote sensing visual grounding.

Extracts multiple distinct target entity phrases from complex sentences like:
"Locate the lake, tennis court and swimming pool."
-> ["lake", "tennis court", "swimming pool"]
"""

from __future__ import annotations

import re


def parse_multi_target_prompt(prompt: str) -> list[str]:
    """Decompose complex query into distinct target referent expressions.

    Examples:
        >>> parse_multi_target_prompt("Locate the lake, tennis court and swimming pool.")
        ['lake', 'tennis court', 'swimming pool']
        >>> parse_multi_target_prompt("Find ship, harbor and boats.")
        ['ship', 'harbor', 'boats']
        >>> parse_multi_target_prompt("Locate the tennis court.")
        ['tennis court']
    """
    clean = prompt.strip()
    if not clean:
        return []

    # 1. Remove introductory command prefixes
    prefixes = [
        r"^(?:please\s+)?locate\s+(?:the\s+|all\s+|any\s+)?",
        r"^(?:please\s+)?find\s+(?:the\s+|all\s+|any\s+)?",
        r"^(?:please\s+)?segment\s+(?:the\s+|all\s+|any\s+)?",
        r"^(?:please\s+)?detect\s+(?:the\s+|all\s+|any\s+)?",
        r"^(?:please\s+)?show\s+(?:me\s+)?(?:the\s+|all\s+)?",
        r"^(?:please\s+)?identify\s+(?:the\s+|all\s+)?",
        r"^(?:please\s+)?where\s+(?:is|are)\s+(?:the\s+|all\s+)?",
    ]
    for p in prefixes:
        clean = re.sub(p, "", clean, flags=re.IGNORECASE).strip()

    # 2. Strip trailing punctuation
    clean = clean.rstrip(".!?")

    # 3. Replace conjunctions with delimiters
    # e.g., ", and ", " and ", " as well as ", " together with "
    clean = re.sub(r"\s+as\s+well\s+as\s+", ", ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+together\s+with\s+", ", ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s*,\s*and\s+", ", ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+and\s+", ", ", clean, flags=re.IGNORECASE)

    # 4. Split by comma or semicolon
    raw_tokens = re.split(r"[,;]+", clean)

    targets: list[str] = []
    for token in raw_tokens:
        t = token.strip()
        # Remove leading determiners: "the", "a", "an", "all"
        t = re.sub(r"^(?:the|a|an|all)\s+", "", t, flags=re.IGNORECASE).strip()
        if t and t not in targets:
            targets.append(t)

    # Fallback: if nothing parsed, return original stripped prompt
    return targets if targets else [prompt.strip().rstrip(".!?")]
