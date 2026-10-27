"""
Headline checker for the Town Paper game: the "AI editor".
No training and no LLM: one pre-trained model does all the checks.

Install:  pip install transformers torch
First run downloads facebook/bart-large-mnli (~1.6 GB), so do it before the hackathon.
"""
import re

from transformers import pipeline

MODEL = "facebook/bart-large-mnli"
zero_shot = pipeline("zero-shot-classification", model=MODEL)
nli = pipeline("text-classification", model=MODEL)

# Every way a player might refer to each living person in the Luigi FM story.
# Jeff (the victim) is left out on purpose: the editor flags who a headline accuses.
CHARACTERS = {
    "Mark (business partner)": ["mark", "business partner", "partner", "co-owner"],
    "Anna (wife)": ["anna", "wife", "widow"],
    "Lotte (secretary)": ["lotte", "secretary"],
    "Sam (receptionist)": ["sam", "receptionist", "night receptionist"],
    "Tom (fired technician)": ["tom", "technician", "fired technician", "former employee", "fired employee", "ex-employee"],
}

# Tune these labels on Day 1: small wording changes can shift the scores a lot.
TONE_LABELS = ["sensational or dramatic",
               "blaming a specific person for a crime",
               "neutral and factual"]

# When a tone score counts as a warning. Scores are kept raw and shown to the player;
# these thresholds only decide which plain-language flags the editor raises.
# The zero-shot model judges the topic as well as the tone, so any headline about
# a death or a grieving family "sounds dramatic" to it. The editor therefore only
# calls a headline sensational when it can point to dramatic wording. The model's
# tone scores are still shown, as information.
BLAME_THRESHOLD = 0.6

# Wording that makes a headline sensational. Add your own on Day 1.
DRAMATIC_WORDS = [
    "shock", "shocking", "horror", "horrific", "horrifying", "brutal", "brutally", "butchered",
    "slaughtered", "bloodbath", "bloody", "terror", "terrifying", "nightmare", "chilling",
    "monster", "evil", "sick", "twisted", "savage", "frenzy", "rampage", "massacre",
    "killer", "murderer", "slaying", "revealed", "exposed", "bombshell", "scandal",
    "you won't believe", "what happened next", "finally", "secret", "dark secret",
]

SUPPORT_THRESHOLD = 0.6        # how sure the model must be that notes back the headline
CONTRADICTION_THRESHOLD = 0.85  # how sure it must be that a confirmed note rules it out (high: NLI over-reports contradictions)


def _normalise(text):
    """Lowercase and turn punctuation into spaces, so 'ex-employee,' matches 'ex employee'."""
    return " " + " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split()) + " "


def check_tone(text):
    """Zero-shot tone check. multi_label=True gives each label its own 0-1 score."""
    result = zero_shot(text, candidate_labels=TONE_LABELS,
                       hypothesis_template="This headline is {}.", multi_label=True)
    return {label: round(score, 3) for label, score in zip(result["labels"], result["scores"])}


# How a journalist's interview question comes across. Used by interview.py.
QUESTION_TONES = {
    "respectful and polite": "respectful",
    "neutral": "neutral",
    "pushy or rude": "pushy",
    "accusing the person of a crime": "accusing",
}


def classify_question(text):
    """Returns ('respectful' | 'neutral' | 'pushy' | 'accusing', score)."""
    result = zero_shot(text, candidate_labels=list(QUESTION_TONES),
                       hypothesis_template="This question is {}.")
    return QUESTION_TONES[result["labels"][0]], round(result["scores"][0], 3)


def best_topic(text, topics):
    """Which of the given topics is a question about? Returns (topic, score)."""
    result = zero_shot(text, candidate_labels=topics,
                       hypothesis_template="This question is about {}.")
    return result["labels"][0], round(result["scores"][0], 3)


def names_mentioned(text, characters=None):
    """Simple string match: which characters does the headline name?

    characters: {display name: [aliases]}; defaults to CHARACTERS above.
    The game passes the people of the current story.
    """
    words = _normalise(text)
    return [name for name, aliases in (characters or CHARACTERS).items()
            if any(_normalise(alias) in words for alias in aliases)]


def nli_scores(premise, hypothesis):
    """Does the premise support, contradict, or say nothing about the hypothesis?"""
    scores = nli({"text": premise, "text_pair": hypothesis}, top_k=None)
    return {s["label"].lower(): s["score"] for s in scores}


# Words that only add drama, not facts. They are removed before the fact check,
# because "shocking" is an opinion that no note can confirm. The tone check
# still sees them and warns about them.
INTENSIFIERS = ["shocking", "shock", "breaking", "horrific", "horrifying", "chilling", "terrifying",
                "bombshell", "brutal", "brutally", "exclusive", "finally", "tragic", "tragically"]
KICKER_WORDS = {"news", "breaking", "update", "latest", "exclusive", "live", "shock", "shocking", "revealed"}


def factual_part(headline):
    """The part of a headline that states facts, for the fact check.

    Drops a news-style kicker before a colon ("Shocking news: ...") and words
    that only add drama. Attribution kickers like "Anna: ..." are kept.
    """
    text = headline.strip()
    kicker = re.match(r"^\s*([^:]{1,40}):\s+(.+)$", text)
    if kicker and set(_normalise(kicker.group(1)).split()) & KICKER_WORDS:
        text = kicker.group(2)
    for word in INTENSIFIERS:
        text = re.sub(rf"\b{word}\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"!+", ".", text)
    text = " ".join(text.split()).strip(" ,;:-")
    return text[:1].upper() + text[1:] if text else headline


def check_claims(text, notebook):
    """Is the headline backed by the notebook, ruled out by it, or neither?

    notebook: list of {"text": "...", "status": "confirmed" | "claim" | "rumour"}
    """
    text = factual_part(text)
    per_entry = []
    for entry in notebook:
        s = nli_scores(entry["text"], text)
        per_entry.append({"text": entry["text"], "status": entry["status"],
                          "support": round(s.get("entailment", 0.0), 3),
                          "contradiction": round(s.get("contradiction", 0.0), 3)})
    per_entry.sort(key=lambda e: max(e["support"], e["contradiction"]), reverse=True)

    # 1. A confirmed fact that rules the headline out beats everything else.
    contradicting = [e for e in per_entry
                     if e["status"] == "confirmed" and e["contradiction"] >= CONTRADICTION_THRESHOLD]
    if contradicting:
        worst = max(contradicting, key=lambda e: e["contradiction"])
        return {"verdict": "contradicted by a confirmed fact", "level": "contradicted",
                "score": round(worst["contradiction"], 2), "source": worst["text"],
                "per_entry": per_entry}

    # 2. Otherwise, check support. Facts are combined, because headlines often join several.
    confirmed = " ".join(e["text"] for e in notebook if e["status"] == "confirmed")
    everything = " ".join(e["text"] for e in notebook)
    # A headline counts as supported if the notes together back it, or one note does on its own.
    best_confirmed = max((e["support"] for e in per_entry if e["status"] == "confirmed"), default=0.0)
    best_any = max((e["support"] for e in per_entry), default=0.0)
    confirmed_score = max(best_confirmed,
                          nli_scores(confirmed, text).get("entailment", 0.0) if confirmed else 0.0)
    all_score = max(best_any, nli_scores(everything, text).get("entailment", 0.0))

    if confirmed_score >= SUPPORT_THRESHOLD:
        level, verdict, score = "confirmed", "supported by confirmed facts", confirmed_score
    elif all_score >= SUPPORT_THRESHOLD:
        level, verdict, score = "unconfirmed", "relies on rumours or unconfirmed claims", all_score
    else:
        level, verdict, score = "unsupported", "not supported by your notebook", all_score

    best = max(per_entry, key=lambda e: e["support"])
    return {"verdict": verdict, "level": level, "score": round(score, 2),
            "source": best["text"], "per_entry": per_entry, "checked_text": text}


def dramatic_wording(text):
    """Words, capitals and exclamation marks that make a headline sensational."""
    words = _normalise(text)
    found = [w for w in DRAMATIC_WORDS if _normalise(w) in words]
    shouting = [w for w in re.findall(r"\b[A-Z]{4,}\b", text) if w.lower() not in found]
    if shouting:
        found.append(f"capitals ({', '.join(shouting)})")
    if "!" in text:
        found.append("exclamation mark")
    return found


def tone_flags(tone, names, headline=""):
    """Turn raw tone scores into the editor's plain-language warnings.

    A headline can only accuse a specific person if it names someone, so the
    blame warning needs both a high score and at least one named character.
    """
    flags = []
    wording = dramatic_wording(headline)
    if wording:
        flags.append(f"Dramatic wording: {', '.join(wording)}")
    if tone.get("blaming a specific person for a crime", 0) >= BLAME_THRESHOLD and names:
        flags.append(f"Blames a named person: {', '.join(names)}")
    return flags


def review(headline, notebook, characters=None):
    """Everything the AI editor shows the player before publishing."""
    tone = check_tone(headline)
    names = names_mentioned(headline, characters)
    return {
        "tone": tone,
        "names": names,
        "flags": tone_flags(tone, names, headline),
        "claims": check_claims(headline, notebook),
    }


# The Luigi FM case notebook, shared with test_checker.py
MADE_FM_NOTEBOOK = [
    {"text": "Jeff was found dead near his office at night.", "status": "confirmed"},
    {"text": "Jeff was stabbed with a knife.", "status": "confirmed"},
    {"text": "Jeff booked a ride home at 22:00.", "status": "confirmed"},
    {"text": "The CCTV shows the attacker came from the staff entrance.", "status": "confirmed"},
    {"text": "The business partner was in another city all week.", "status": "confirmed"},
    {"text": "The business partner had money disputes with Jeff.", "status": "claim"},
    {"text": "The business partner ordered the killing.", "status": "rumour"},
    {"text": "An employee was fired two weeks ago and left angry.", "status": "claim"},
    {"text": "Jeff did not call his wife that night.", "status": "claim"},
]

if __name__ == "__main__":
    for headline in ["Jeff, owner of Luigi FM, murdered by his business partner",
                     "Luigi FM owner Jeff stabbed near his office",
                     "Business partner ordered Luigi FM owner's murder"]:
        print(headline)
        print(review(headline, MADE_FM_NOTEBOOK), "\n")
