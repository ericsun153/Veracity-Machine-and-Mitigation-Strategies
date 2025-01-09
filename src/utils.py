import re

def repetition_analysis(text: str) -> dict:
    # Detect overused words, phrases, and repetitive structures
    words = re.findall(r'\b\w+\b', text.lower())
    total_word_count = len(words)
    
    # Track word frequencies
    word_frequency = {word: words.count(word) for word in set(words)}
    high_frequency_words = [word for word, freq in word_frequency.items() if freq > 0.05 * total_word_count]
    
    # Identify repetitive phrases with significant emotional weight
    emotional_repetitions = [word for word in high_frequency_words if word in ["freedom", "rights", "danger", "attack", "urgent"]]
    
    # Score calculation based on high-frequency words and emotionally charged repetitions
    repetition_score = (1 - len(set(words)) / total_word_count) * 100
    score = min(100, int(repetition_score + (20 if len(emotional_repetitions) > 3 else 0)))
    
    details = (
        f"High repetition detected with key emotional words ({', '.join(emotional_repetitions)})."
        if score > 70 else f"Moderate repetition; key terms like {', '.join(high_frequency_words[:3])} appear often."
        if score > 40 else "Low repetition; balanced word choice."
    )
    
    return {"score": score, "details": details}


def origin_tracing(text: str) -> dict:
    # Track credible sources, vague sources, and contextual fallacies
    credible_sources = ["New York Times", "BBC", "Journal", "Harvard Study", "Pew Research", "NBC", "CNN"]
    vague_sources = ["many say", "some claim", "experts believe", "widely reported", "it is said"]
    fallacious_phrases = ["everybody knows", "most people agree", "common knowledge", "widely accepted"]

    source_mentions = [source for source in credible_sources if source.lower() in text.lower()]
    vague_source_mentions = [phrase for phrase in vague_sources if phrase.lower() in text.lower()]
    fallacy_mentions = [phrase for phrase in fallacious_phrases if phrase.lower() in text.lower()]
    
    # Score adjustments for credible sources, vague sources and logical fallacies
    score = 85 if len(source_mentions) >= 3 else 65 if len(source_mentions) == 2 else 45 if len(source_mentions) == 1 else 25
    if vague_source_mentions:
        score -= 15 
    if fallacy_mentions:
        score -= 10
    
    details = (
        f"Strong source support with credible references like {', '.join(source_mentions[:3])}."
        if score > 70 else f"Moderate sourcing but reliance on vague claims ({', '.join(vague_source_mentions)}) or fallacies."
        if score > 50 else "Limited credible sources and potential use of logical fallacies, reducing reliability."
    )
    
    return {"score": score, "details": details}


def evidence_verification(text: str) -> dict:
    # Focus on authority and the presence of logical fallacies
    expert_keywords = ["expert", "analysis", "data", "study", "evidence", "research"]
    authority_phrases = ["according to a study", "research shows", "data suggests", "experts agree", "studies confirm"]
    unsupported_authority = ["believe me", "trust me", "I'm an expert"]

    expert_mentions = sum(text.lower().count(kw) for kw in expert_keywords)
    authority_mentions = sum(text.lower().count(phrase) for phrase in authority_phrases)
    unsupported_mentions = [phrase for phrase in unsupported_authority if phrase.lower() in text.lower()]
    
    # Score based on valid authority phrases, penalize unsupported assertions
    score = 30 + (expert_mentions * 10) + (authority_mentions * 15)
    score -= 10 * len(unsupported_mentions) 
    score = min(score, 100)
    
    details = (
        f"Strong evidence support with multiple authoritative mentions such as {', '.join(authority_phrases[:3])}."
        if score > 70 else f"Moderate evidence support with some expert references; limited or no unsupported assertions."
        if score > 50 else "Weak support; lacks references to credible evidence or includes unsupported claims."
    )
    
    return {"score": score, "details": details}


def omission_checks(text: str) -> dict:
    # Detect language implying critical omissions and checks for selective framing
    omission_indicators = ["left out", "excluded", "did not mention", "missing", "fails to include", "overlooks"]
    selective_phrases = ["only tells part of the story", "fails to provide full context", "hides important details", "one-sided"]

    omissions_detected = any(indicator in text.lower() for indicator in omission_indicators)
    selective_detected = any(phrase in text.lower() for phrase in selective_phrases)
    score = 70 if omissions_detected else 30
    score += 10 if selective_detected else 0
    
    details = (
        "Critical omissions detected, potentially distorting the narrative by leaving out key information."
        if score > 60 else "No significant omissions; content appears balanced."
    )
    
    return {"score": score, "details": details}


def exaggeration_analysis(text: str) -> dict:
    # Identify hyperbolic words and intensifiers
    hyperbole_words = ["scandal", "unbelievable", "never seen before", "shocking", "incredible", "outrage", "catastrophe"]
    intensifiers = ["very", "extremely", "incredibly", "deeply", "highly", "absolutely"]
    fear_inducing_language = ["threat", "emergency", "crisis", "alarming", "grave"]

    hyperbolic_phrases = sum(text.lower().count(word) for word in hyperbole_words)
    intensifier_mentions = sum(text.lower().count(word) for word in intensifiers)
    fear_mentions = sum(text.lower().count(word) for word in fear_inducing_language)

    # Calculate score based on hyperbole, intensifiers, and fear language
    score = 40 + (hyperbolic_phrases * 12) + (intensifier_mentions * 5) + (fear_mentions * 8)
    score = min(score, 100)
    
    details = (
        f"High exaggeration with terms like {', '.join(hyperbole_words[:3])} and fear language ({', '.join(fear_inducing_language[:3])})."
        if score > 70 else "Moderate exaggeration; some hyperbolic or fear-inducing phrases noted."
        if score > 40 else "Minimal exaggeration; tone appears mostly neutral."
    )
    
    return {"score": score, "details": details}


def target_audience_assessment(text: str) -> dict:
    # Look for emotionally manipulative, urgency and tribalism languages
    targeting_keywords = ["you must", "they don’t want you to know", "us vs. them", "wake up", "be aware", "fight back"]
    urgency_phrases = ["protect your family", "act now", "don't be fooled", "stand up", "they are hiding"]
    tribal_phrases = ["our side", "their side", "them vs us", "we are right", "they are wrong"]

    manipulative_phrases = sum(text.lower().count(kw) for kw in targeting_keywords)
    urgency_mentions = sum(text.lower().count(phrase) for phrase in urgency_phrases)
    tribal_mentions = sum(text.lower().count(phrase) for phrase in tribal_phrases)

    # Score based on presence of urgency and tribalism indicators
    score = 50 + (manipulative_phrases * 10) + (urgency_mentions * 8) + (tribal_mentions * 12)
    score = min(score, 100)
    
    details = (
        f"Strong emotional manipulation with urgency ({', '.join(urgency_phrases[:3])}) and tribal language ({', '.join(tribal_phrases[:3])})."
        if score > 70 else "Moderate emotional language; some urgency or tribalistic phrases."
        if score > 40 else "Minimal manipulation; tone appears mostly neutral."
    )
    
    return {"score": score, "details": details}