"""
Keyword evaluator using a minimal Porter stemmer and stop word filtering.
"""

import re

# Common English stop words to filter out
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those", "i",
    "me", "my", "we", "our", "you", "your", "he", "she", "his", "her",
    "they", "them", "their", "what", "which", "who", "whom", "when",
    "where", "why", "how", "all", "each", "both", "few", "more", "most",
    "some", "such", "no", "not", "only", "same", "so", "than", "too",
    "very", "just", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "out", "off", "over", "under", "again",
    "then", "once", "any", "also", "about", "up", "if", "there",
}


def _is_consonant(word: str, index: int) -> bool:
    """Return True if character at index is a consonant."""
    ch = word[index]
    if ch in "aeiou":
        return False
    if ch == "y":
        return index == 0 or not _is_consonant(word, index - 1)
    return True


def _measure(word: str) -> int:
    """
    Compute the measure m of a word (number of VC sequences).
    """
    n = len(word)
    count = 0
    i = 0
    # skip leading consonants
    while i < n and _is_consonant(word, i):
        i += 1
    while i < n:
        # skip vowels
        while i < n and not _is_consonant(word, i):
            i += 1
        # skip consonants
        j = i
        while j < n and _is_consonant(word, j):
            j += 1
        if j > i:
            count += 1
        i = j
    return count


def _has_vowel(word: str) -> bool:
    """Return True if word contains at least one vowel."""
    return any(not _is_consonant(word, i) for i in range(len(word)))


def _ends_double_consonant(word: str) -> bool:
    """Return True if word ends with a double consonant."""
    if len(word) < 2:
        return False
    return word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """
    Return True if word ends with consonant-vowel-consonant
    and the last consonant is not w, x, or y.
    """
    if len(word) < 3:
        return False
    i = len(word) - 1
    return (
        _is_consonant(word, i)
        and not _is_consonant(word, i - 1)
        and _is_consonant(word, i - 2)
        and word[i] not in "wxy"
    )


def porter_stem(word: str) -> str:
    """
    Apply a minimal Porter stemmer to reduce a word to its stem.
    Handles: plurals, past tense, gerunds, adverbs, derivational suffixes.
    """
    word = word.lower()

    # Short words don't need stemming
    if len(word) <= 2:
        return word

    # --- Step 1a: plural / third person singular ---
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-3] + "i"
    elif word.endswith("ss"):
        pass  # leave as-is
    elif word.endswith("s") and not word.endswith("ss"):
        # only strip if root is non-trivial
        if len(word) > 2 and word[-2] not in "aeiou":
            word = word[:-1]
        elif len(word) > 3:
            word = word[:-1]

    # --- Step 1b: past tense / gerunds ---
    changed = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]  # eed -> ee
    elif word.endswith("ed"):
        stem = word[:-2]
        if _has_vowel(stem):
            word = stem
            changed = True
    elif word.endswith("ing"):
        stem = word[:-3]
        if _has_vowel(stem):
            word = stem
            changed = True

    if changed:
        if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
            word = word + "e"
        elif _ends_double_consonant(word) and word[-1] not in "lsz":
            word = word[:-1]
        elif _measure(word) == 1 and _ends_cvc(word):
            word = word + "e"

    # --- Step 1c: y -> i ---
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    # --- Step 2: derivational suffixes (m > 0) ---
    step2_rules = [
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
    ]
    for suffix, replacement in step2_rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 0:
                word = stem + replacement
            break

    # --- Step 3: further suffix removal (m > 0) ---
    step3_rules = [
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ]
    for suffix, replacement in step3_rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 0:
                word = stem + replacement
            break

    # --- Step 4: remove remaining derivational suffixes (m > 1) ---
    step4_rules = [
        "ement", "ment", "ance", "ence", "able", "ible",
        "ant", "ent", "ion", "ism", "ate", "iti", "ous",
        "ive", "ize", "al", "er", "ic",
    ]
    for suffix in step4_rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            m = _measure(stem)
            if suffix == "ion":
                if m > 1 and stem and stem[-1] in "st":
                    word = stem
            elif m > 1:
                word = stem
            break

    # --- Step 5a: remove trailing e (m > 1, or m == 1 and not *o) ---
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1:
            word = stem
        elif m == 1 and not _ends_cvc(stem):
            word = stem

    # --- Step 5b: double ll -> l (m > 1) ---
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]

    return word


def _tokenize_and_stem(text: str) -> dict:
    """
    Tokenize text, remove stop words, and stem each remaining word.
    Returns a dict mapping stem -> list of original words.
    """
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    stem_map: dict[str, list[str]] = {}
    for token in tokens:
        if token in STOP_WORDS:
            continue
        stem = porter_stem(token)
        if stem not in stem_map:
            stem_map[stem] = []
        stem_map[stem].append(token)
    return stem_map


def evaluate_keywords(
    user_answer: str,
    reference_answer: str,
    max_score: float,
) -> dict:
    """
    Evaluate keyword coverage of user_answer against reference_answer.

    Args:
        user_answer: The student's answer text.
        reference_answer: The reference/model answer text.
        max_score: The maximum possible score for this metric.

    Returns:
        dict with keys:
            score, max_score, normalized_score, confidence,
            coverage, matched_keywords, missing_keywords,
            matched_count, total_reference_keywords
    """
    if not reference_answer or not reference_answer.strip():
        return {
            "score": 0.0,
            "max_score": max_score,
            "normalized_score": 0.0,
            "confidence": 0.95,
            "coverage": 0.0,
            "matched_keywords": [],
            "missing_keywords": [],
            "matched_count": 0,
            "total_reference_keywords": 0,
        }

    ref_stems = _tokenize_and_stem(reference_answer)
    answer_stems = _tokenize_and_stem(user_answer)

    total_ref = len(ref_stems)

    if total_ref == 0:
        return {
            "score": 0.0,
            "max_score": max_score,
            "normalized_score": 0.0,
            "confidence": 0.95,
            "coverage": 0.0,
            "matched_keywords": [],
            "missing_keywords": list(reference_answer.split()),
            "matched_count": 0,
            "total_reference_keywords": 0,
        }

    matched_keywords = []
    missing_keywords = []

    for stem, ref_words in ref_stems.items():
        # Use the first representative word from the reference
        representative = ref_words[0]
        if stem in answer_stems:
            matched_keywords.append(representative)
        else:
            missing_keywords.append(representative)

    matched_count = len(matched_keywords)
    coverage = matched_count / total_ref if total_ref > 0 else 0.0
    score = coverage * max_score
    normalized_score = coverage  # already in [0, 1]

    return {
        "score": score,
        "max_score": max_score,
        "normalized_score": normalized_score,
        "confidence": 0.95,
        "coverage": coverage,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "matched_count": matched_count,
        "total_reference_keywords": total_ref,
    }
