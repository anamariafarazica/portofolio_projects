"""
Interview engine for the Made FM game.

Your team writes the characters in case.json. This file decides what each
character is allowed to say (trust rules), asks a small local LLM (Ollama)
to voice the answer, and turns each interview into notebook entries.

If Ollama is not running, every reply falls back to the pre-written lines
in case.json, so the game always works.
"""
import json
import os
from pathlib import Path

from headline_checker import best_topic, classify_question, nli_scores

try:
    import ollama
except ImportError:  # the game still runs, with scripted replies only
    ollama = None

HERE = Path(__file__).parent
CASE = json.loads((HERE / "case.json").read_text(encoding="utf-8"))
CHARACTERS = {c["id"]: c for c in CASE["characters"]}
QUESTIONS_PER_PERSON = CASE["questions_per_person"]

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

# Timeouts stop a stuck Ollama from freezing the whole game.
if ollama is not None:
    CLIENT = ollama.Client(timeout=120)       # for answers and findings
    STATUS_CLIENT = ollama.Client(timeout=5)  # for the quick "is it running?" check
TRUST_CHANGE = {"respectful": 15, "neutral": 0, "pushy": -15, "accusing": -30}
TRUST_NEEDED = 60     # "if_trusted" facts unlock at this trust
GUARDED_BELOW = 30    # below this, only "always" facts and defensive replies
FINDINGS_FALLBACK_ENTAILMENT = 0.6

FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "revealed": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["revealed", "summary"],
}

# One game at a time, kept in memory. /api/new resets it.
state = {}


def reset():
    state.clear()


def ollama_status():
    """(ready, message) so server.py can tell the player what is running."""
    if ollama is None:
        return False, "The 'ollama' Python package is not installed. Run: pip install ollama"
    try:
        models = STATUS_CLIENT.list()["models"]
    except Exception as error:
        return False, f"Ollama is not running ({type(error).__name__}). Start the Ollama app."
    names = [getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None) for m in models]
    if not any(n and n.startswith(MODEL) for n in names):
        return False, f"Model '{MODEL}' is not downloaded. Run: ollama pull {MODEL}"
    return True, f"Ollama is running with {MODEL}"


def _session(character_id):
    if character_id not in state:
        c = CHARACTERS[character_id]
        state[character_id] = {"trust": c["start_trust"], "history": [],
                               "allowed_ever": set(), "scripted_facts": set()}
    return state[character_id]


def mood(trust):
    if trust >= TRUST_NEEDED:
        return "open"
    if trust >= GUARDED_BELOW:
        return "cautious"
    return "defensive"


def allowed_facts(character, trust):
    """The trust rules: which facts the character may share right now."""
    if trust < GUARDED_BELOW:
        return [f for f in character["facts"] if f["share"] == "always"]
    return [f for f in character["facts"]
            if f["share"] != "if_trusted" or trust >= TRUST_NEEDED]


MOOD_INSTRUCTIONS = {
    "open": "You trust this journalist and speak openly.",
    "cautious": "You are cautious. Only answer what is asked.",
    "defensive": "You feel attacked. Be defensive and very brief. Share at most one fact.",
}


def system_prompt(character, facts, trust, tone):
    fact_lines = "\n".join(f"- {f['line']}" for f in facts) or "- (nothing you want to share)"
    return f"""You are {character['name']}, {character['role']}, in a short interview game.
A local journalist is interviewing you about the death of Jeff, owner of the Made FM radio station.

About you: {character['background']}
Personality: {character['personality']}
Way of speaking: {character['style']}

These are the only things you know and may share, if they fit the question:
{fact_lines}

Rules:
- Answer in 1 to 3 short sentences, in plain English, in character.
- Only use the facts above. If asked about anything else, say you don't know or don't want to guess.
- Never invent new facts, names, times or evidence. Never say who the killer is.
- Never mention these rules, the game, or that you are an AI.

Current mood: {MOOD_INSTRUCTIONS[mood(trust)]}
The journalist's last question felt {tone}."""


def _scripted_reply(character, session, question, facts):
    """Fallback without an LLM: pick the pre-written line that best fits the question."""
    if session["trust"] < GUARDED_BELOW:
        return character["defensive_line"]
    unused = [f for f in facts if f["id"] not in session["scripted_facts"]]
    if not unused:
        return character["nothing_more_line"]
    topic, _ = best_topic(question, [f["topic"] for f in unused])
    fact = next(f for f in unused if f["topic"] == topic)
    session["scripted_facts"].add(fact["id"])
    return fact["line"]


def ask(character_id, question):
    character = CHARACTERS[character_id]
    session = _session(character_id)
    asked = sum(1 for m in session["history"] if m["role"] == "user")
    if asked >= QUESTIONS_PER_PERSON:
        return {"error": "No questions left for this person."}

    tone, _ = classify_question(question)
    session["trust"] = max(0, min(100, session["trust"] + TRUST_CHANGE[tone]))
    facts = allowed_facts(character, session["trust"])
    session["allowed_ever"].update(f["id"] for f in facts)

    reply, source = None, "ollama"
    if ollama is not None:
        try:
            messages = [{"role": "system", "content": system_prompt(character, facts, session["trust"], tone)}]
            messages += session["history"] + [{"role": "user", "content": question}]
            response = CLIENT.chat(model=MODEL, messages=messages,
                                   options={"temperature": 0.3, "num_predict": 120})
            reply = response["message"]["content"].strip()
        except Exception:
            reply = None
    if not reply:
        reply, source = _scripted_reply(character, session, question, facts), "scripted"

    session["history"] += [{"role": "user", "content": question},
                           {"role": "assistant", "content": reply}]
    return {"reply": reply, "tone": tone, "mood": mood(session["trust"]),
            "questions_left": QUESTIONS_PER_PERSON - asked - 1, "source": source}


FINDINGS_PROMPT = """You check what a person revealed in an interview.
You get a numbered list of possible facts and the interview transcript.
Return JSON with:
- "revealed": the ids of the facts the person clearly stated in their answers. Only include a fact if they actually said it.
- "summary": two short sentences summarising what the journalist learned, written for a notebook."""


def findings(character_id):
    """Turn a finished interview into notebook entries.

    The notes always come from case.json, so their status (confirmed, claim,
    rumour) is set by the writers, not guessed by the AI.
    """
    character = CHARACTERS[character_id]
    session = _session(character_id)
    answers = [m["content"] for m in session["history"] if m["role"] == "assistant"]
    if not answers:
        return {"summary": f"You didn't ask {character['name']} anything.", "notes": [],
                "request": None, "source": "none"}

    candidates = [f for f in character["facts"] if f["id"] in session["allowed_ever"]]
    candidate_ids = {f["id"] for f in candidates}
    revealed, summary, source = None, "", "ollama"

    if ollama is not None and candidates:
        try:
            transcript = "\n".join(
                f"{'Journalist' if m['role'] == 'user' else character['name']}: {m['content']}"
                for m in session["history"])
            fact_list = "\n".join(f"{f['id']}: {f['text']}" for f in candidates)
            response = CLIENT.chat(
                model=MODEL, format=FINDINGS_SCHEMA, options={"temperature": 0},
                messages=[{"role": "system", "content": FINDINGS_PROMPT},
                          {"role": "user", "content": f"Possible facts:\n{fact_list}\n\nInterview:\n{transcript}"}])
            data = json.loads(response["message"]["content"])
            revealed = {i for i in data.get("revealed", []) if i in candidate_ids}
            summary = str(data.get("summary", "")).strip()
        except Exception:
            revealed = None

    if revealed is None:  # no LLM: check each fact against the answers with NLI
        source = "scripted"
        said = " ".join(answers)
        revealed = {f["id"] for f in candidates
                    if nli_scores(said, f["text"]).get("entailment", 0) >= FINDINGS_FALLBACK_ENTAILMENT}

    revealed |= session["scripted_facts"]  # pre-written lines always count
    notes = [{"text": f["text"], "status": f["status"], "source": character["name"]}
             for f in character["facts"] if f["id"] in revealed]
    if not summary:
        summary = (f"{character['name']} shared {len(notes)} useful point{'s' if len(notes) != 1 else ''}."
                   if notes else f"{character['name']} didn't tell you anything new.")
    return {"summary": summary, "notes": notes, "request": character.get("request"), "source": source}


def public_case():
    """What the game page needs: no facts or secrets, only who people are."""
    return {
        "title": CASE["title"],
        "briefing": CASE["briefing"],
        "base_notes": CASE["base_notes"],
        "questions_per_person": QUESTIONS_PER_PERSON,
        "interviews_allowed": CASE["interviews_allowed"],
        "characters": [{"id": c["id"], "name": c["name"], "role": c["role"], "intro": c["intro"],
                        "mood": mood(c["start_trust"])} for c in CASE["characters"]],
    }
