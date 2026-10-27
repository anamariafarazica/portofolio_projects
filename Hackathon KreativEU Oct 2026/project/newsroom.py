"""
Newsroom engine for The Midnight Chronicle.

The game is a reporter's career: one story after another, each written as a
file in the cases/ folder. Inside a story, information arrives in news cycles.
A story ends when its final news arrives (the arrest, the council's decision),
or earlier, when readers lose interest and the editor moves you on.

The AI does two jobs:
  1. the AI editor (headline_checker.py) gives advice on every headline,
  2. a small local LLM (Ollama) writes readers' reactions to that exact headline.
All scores and consequences come from the rules below and the case files,
never from the AI. If Ollama is not running, reactions come from the case file.
"""
import json
import os
import re
from pathlib import Path

from headline_checker import CONTRADICTION_THRESHOLD, review

try:
    import ollama
except ImportError:
    ollama = None

HERE = Path(__file__).parent
API = 6   # bump when the page and the engine change together; game.html checks it
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
if ollama is not None:
    CLIENT = ollama.Client(timeout=120)
    STATUS_CLIENT = ollama.Client(timeout=5)


CACHE_FILE = HERE / "ai_cache.json"
try:
    _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
except (OSError, ValueError):
    _cache = {}


def _cache_get(kind, key):
    return _cache.get(kind, {}).get(key)


def _cache_put(kind, key, value):
    _cache.setdefault(kind, {})[key] = value
    try:
        CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # caching is a nice-to-have; the game works without it


def _load_cases():
    cases = []
    for path in sorted((HERE / "cases").glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        case["ITEMS"] = {item["id"]: {**item, "cycle": n}
                         for n, cycle in enumerate(case["cycles"], start=1) for item in cycle["items"]}
        case["PEOPLE"] = {p["id"]: p for p in case["people"]}
        case["NAMES"] = {p["name"]: p["aliases"] for p in case["people"]}
        cases.append(case)
    if not cases:
        raise SystemExit("No stories found. Put the case files in a folder called 'cases' next to newsroom.py.")
    return sorted(cases, key=lambda c: c.get("order", 99))


CASES = _load_cases()

# ---- The rules (tune these) ----
READERS_SENSATIONAL = 1.4   # dramatic or blaming headlines get more readers
READERS_NAMED = 1.2         # naming someone gets more readers
READERS_UNCONFIRMED = 0.7   # an "unconfirmed" label costs readers
HARM_SENSATIONAL = 1.5      # dramatic or blaming headlines hurt people more
HARM_UNCONFIRMED = 0.5      # an "unconfirmed" label halves the harm

# Public trust measures accuracy, not tone, and is based only on what each story
# really rests on (from the case file), never on the AI's judgement. It carries
# over from story to story.
TRUST_BY_STATUS = {"confirmed": 2, "claim": 0, "rumour": -4}           # published as fact
TRUST_BY_STATUS_LABELLED = {"confirmed": 0, "claim": 0, "rumour": -1}  # labelled unconfirmed
TRUST_DEBUNK = 3
FALSE_STORY_PENALTY = -15
FALSE_STORY_PENALTY_LABELLED = -6

# Story interest: how much readers still care about this story. Articles raise
# it, every news cycle it fades, and scoops by the competitor lower it. If it
# falls below the threshold, the editor drops the story before its final news.
INTEREST_START = 50
INTEREST_PER_1000_READERS = 4
INTEREST_FADE = 15
INTEREST_SCOOPED = 5
INTEREST_THRESHOLD = 25
DROP_AFTER_CYCLE = 2        # a story always gets at least this many news cycles

state = {}


# ---------- Career and stories ----------

def case():
    return CASES[state["case_index"]]


def _changed():
    state["version"] = state.get("version", 0) + 1


def story():
    """The current case's story. Every case keeps its own progress, so players can switch."""
    if case()["id"] not in state["stories"]:
        _new_story()
    return state["stories"][case()["id"]]


def new_game():
    version = state.get("version", 0)
    state.clear()
    state.update(case_index=0, stories={}, readers=0, credibility=CASES[0]["start_credibility"],
                 history=[], career_over=False, version=version,
                 career_timeline=[{"case": "", "label": "Start", "credibility": CASES[0]["start_credibility"], "readers": 0}])
    _changed()


def open_case(case_id):
    """Open a case from the case board: start it, or continue where you left off."""
    index = next((i for i, c in enumerate(CASES) if c["id"] == case_id), None)
    if index is None:
        raise ValueError("That case doesn't exist.")
    state["case_index"] = index
    story()
    _changed()
    return view()


def _started(s):
    return bool(s["decisions"] or s["judgements"] or s["cycle"] > 1)


def cases_overview():
    out = []
    for n, c in enumerate(CASES, start=1):
        s = state["stories"].get(c["id"])
        status = ("new" if not s or not _started(s) else s["ended"] if s["finished"] else "in_progress")
        out.append({"id": c["id"], "number": n, "title": c["title"], "teaser": c.get("teaser", c["intro"][0]),
                    "max_cycles": len(c["cycles"]), "final_label": c.get("final_label", "the final news"),
                    "status": status, "cycle": s["cycle"] if s else 1})
    return out


def finish_career():
    state["career_over"] = True
    _changed()
    return {"career": career_summary(), "game": view()}


def _new_story():
    state["stories"][case()["id"]] = dict(
        cycle=1, finished=False, ended=None, interest=INTEREST_START, readers=0,
        harm={pid: {"you": 0.0, "others": 0.0} for pid in case()["PEOPLE"]},
        decisions={}, published=[], verifying={}, competitor_done=set(), penalised=set(),
        trust_log=[], epilogue=None, judgements={}, judged_right=0, judged_total=0, timeline=[])
    _snapshot()
    _record("Start")


def _record(label):
    """Remember the numbers at this moment, for the charts at the end of a case."""
    s = state["stories"][case()["id"]]
    s["timeline"].append({
        "label": label, "readers": round(s["readers"]), "credibility": round(state["credibility"]),
        "interest": round(s["interest"]),
        "pressure": {case()["PEOPLE"][pid]["name"]: round(min(100, h["you"] + h["others"]))
                     for pid, h in s["harm"].items()},
    })
    if label != "Start":  # the career chart follows every news cycle played, across cases
        state["career_timeline"].append({"case": case()["title"], "label": label,
                                         "credibility": round(state["credibility"]),
                                         "readers": round(state["readers"])})


def next_story():
    """Kept for older pages: back to the case board, or the career summary when all cases are closed."""
    if all(c["status"] in ("resolved", "dropped") for c in cases_overview()):
        return finish_career()
    return {"game": view()}


def _snapshot():
    s = story()
    s["trust_log"] = []
    s["cycle_start"] = {"readers": state["readers"], "credibility": state["credibility"],
                        "interest": s["interest"],
                        "harm": {pid: dict(h) for pid, h in s["harm"].items()}}


def _short(headline, limit=70):
    return headline if len(headline) <= limit else headline[:limit].rsplit(" ", 1)[0] + "\u2026"


def _trust(change, reason, key=None):
    if change:
        state["credibility"] = max(0, min(100, state["credibility"] + change))
        story()["trust_log"].append({"change": change, "reason": reason, "key": key})


def _pressure_snapshot():
    return {pid: dict(h) for pid, h in story()["harm"].items()}


def _pressure_changes(before):
    """[{name, change, source}] for everyone whose pressure changed since `before`."""
    out = []
    for pid, h in story()["harm"].items():
        for source in ("you", "others"):
            change = round(h[source] - before[pid][source])
            if change:
                out.append({"name": case()["PEOPLE"][pid]["name"], "change": change, "source": source})
    return out


def _interest(change):
    s = story()
    s["interest"] = max(0, min(100, s["interest"] + change))


def _inbox_entries(cycle):
    """(item_id, is_verification_result) for everything on the desk in a cycle."""
    s, entries = story(), []
    for item in case()["cycles"][cycle - 1]["items"]:
        needs = item.get("requires_published_any")
        if needs and not any(p["id"] in needs and p["cycle"] < cycle and not p.get("debunk") for p in s["published"]):
            continue
        entries.append((item["id"], False))
    entries += [(iid, True) for iid, back in s["verifying"].items() if back == cycle]
    return entries


def _key(iid, cycle):
    return f"{iid}@{cycle}"


def notebook():
    """Everything the player knows about this story, as the AI editor sees it.

    New items carry the player's own judgement of how reliable they are.
    Results of a check carry what the check found.
    """
    notes, seen, items, s = [], set(), case()["ITEMS"], story()
    for cycle in range(1, s["cycle"] + 1):
        for iid, verified in _inbox_entries(cycle):
            item = items[iid]
            if verified:
                entry = item["verify"]["note"]
            elif item.get("skip_judge"):
                entry = {"text": item["note"], "status": item["status"]}
            else:
                judged = s["judgements"].get(_key(iid, cycle))
                if not judged:
                    continue  # not in the notebook until the player has judged it
                entry = {"text": item["note"], "status": judged, "judged": True}
            if entry["text"] not in seen:
                seen.add(entry["text"])
                notes.append({**entry, "source": item["from"], "cycle_label": case()["cycles"][cycle - 1]["label"]})
    return notes


def agenda():
    """Every case the player has opened: its notes and front pages, for the agenda."""
    current, pages = state["case_index"], []
    for index, c in enumerate(CASES):
        if c["id"] not in state["stories"]:
            continue
        state["case_index"] = index
        s = story()
        if not _started(s):
            continue
        pages.append({"case_id": c["id"], "number": index + 1, "title": c["title"],
                      "status": "in_progress" if not s["finished"] else s["ended"],
                      "notes": notebook(),
                      "front_pages": [{"cycle_label": p["cycle_label"], "headline": p["headline"],
                                       "unconfirmed": p["unconfirmed"]} for p in s["published"]]})
    state["case_index"] = current
    return pages


def _current_status(item, verified):
    return item["verify"]["status"] if verified else item["status"]


def _norm(text):
    return " " + " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split()) + " "


def _named_people(headline):
    words = _norm(headline)
    return [pid for pid, p in case()["PEOPLE"].items() if any(_norm(a) in words for a in p["aliases"])]


# ---------- What the page sees ----------

def view():
    c, s = case(), story()
    cycle = s["cycle"]
    inbox = []
    for iid, verified in _inbox_entries(cycle):
        item = c["ITEMS"][iid]
        key = _key(iid, cycle)
        inbox.append({
            "key": key, "kind": item["kind"], "from": item["from"], "text": item["text"],
            # The real status is only shown after a check, or for items that skip judging.
            "status": _current_status(item, verified) if verified or item.get("skip_judge") else None,
            "skip_judge": bool(item.get("skip_judge")), "judge_only": bool(item.get("judge_only")),
            "judged": s["judgements"].get(key),
            # Optional ready-made headline (used in the demo case); the player can still edit it
            "suggested_headline": (item["verify"].get("headline") if verified else item.get("headline")),
            "verification": item["verify"]["text"] if verified else None,
            "request": item.get("request"),
            "decision": s["decisions"].get(key),
            "can_verify": (not verified) and "verify" in item and cycle < len(c["cycles"]),
        })
    return {
        "api": API,   # game.html checks this, so an old server can't silently break the page
        "version": state["version"],
        "outlet": c["outlet"], "competitor": c["competitor"],
        "story_title": c["title"], "story_number": state["case_index"] + 1, "story_count": len(CASES),
        "case_id": c["id"], "story_started": _started(s), "cases": cases_overview(),
        "presentation": bool(c.get("presentation")), "slide": c["cycles"][cycle - 1].get("slide"),
        "team": c.get("team"),
        "all_closed": all(x["status"] in ("resolved", "dropped") for x in cases_overview()),
        "intro": c["intro"], "finished": s["finished"], "ended": s["ended"],
        "career_over": state["career_over"],
        "has_next_story": any(x["status"] not in ("resolved", "dropped") for x in cases_overview()),
        "cycle": cycle, "max_cycles": len(c["cycles"]), "final_label": c.get("final_label", "the final news"),
        "cycle_label": c["cycles"][cycle - 1]["label"], "cycle_intro": c["cycles"][cycle - 1]["intro"],
        "readers": round(state["readers"]), "credibility": round(state["credibility"]),
        "interest": round(s["interest"]), "interest_threshold": INTEREST_THRESHOLD,
        "people": [_person_view(pid) for pid in c["PEOPLE"]],
        "inbox": inbox, "notebook": notebook(),
        "front_pages": [{"cycle_label": p["cycle_label"], "headline": p["headline"],
                         "unconfirmed": p["unconfirmed"]} for p in s["published"]],
        "epilogue": s["epilogue"],
        "agenda": agenda(),
    }


def _person_view(pid):
    h, p = story()["harm"][pid], case()["PEOPLE"][pid]
    return {"id": pid, "name": p["name"], "role": p["role"],
            "you": round(h["you"]), "others": round(h["others"]), "total": round(min(100, h["you"] + h["others"]))}


# ---------- Decisions ----------

def _find(key):
    cycle = story()["cycle"]
    for iid, verified in _inbox_entries(cycle):
        if _key(iid, cycle) == key:
            return case()["ITEMS"][iid], verified
    raise ValueError("That item isn't on your desk right now.")


def _open(key):
    if story()["finished"]:
        raise ValueError("This story is over. Move on to the next one.")
    item, verified = _find(key)
    if key in story()["decisions"]:
        raise ValueError("You already decided about this item.")
    if not verified and not item.get("skip_judge") and key not in story()["judgements"]:
        raise ValueError("First decide how reliable this source is.")
    return item, verified


JUDGEMENTS = ("confirmed", "claim", "rumour")


def judge(key, label):
    """The player decides how reliable a new item is. They can change it until they act on it."""
    if story()["finished"]:
        raise ValueError("This story is over. Move on to the next one.")
    _, verified = _find(key)
    if verified:
        raise ValueError("Your check already showed how reliable this is.")
    if key in story()["decisions"]:
        raise ValueError("You already acted on this item.")
    if label not in JUDGEMENTS:
        raise ValueError("Choose confirmed, one source or unverified.")
    story()["judgements"][key] = label
    item, _ = _find(key)
    if item.get("judge_only"):  # in the demo, some items only ask for a judgement
        story()["decisions"][key] = {"action": "noted"}
    _changed()
    return {"ok": True}


# The editor's verdict comes from the source the article is written from, which
# the game knows for certain. The AI only adds advice: whether the headline
# seems to say more than that source, or clashes with a confirmed note.
SOURCE_VERDICTS = {   # for a new item: the player's own judgement of the source
    "confirmed": ("confirmed", "based on a source you judged confirmed"),
    "claim": ("unconfirmed", "based on a source you judged as one person's account"),
    "rumour": ("unsupported", "based on a source you judged unverified"),
}
PRESET_VERDICTS = {   # for an item that skips judging: its real status
    "confirmed": ("confirmed", "based on a confirmed source"),
    "claim": ("unconfirmed", "based on one person's account"),
    "rumour": ("unsupported", "based on an unverified rumour"),
}
CHECKED_VERDICTS = {  # for a checked item: what the check found
    "confirmed": ("confirmed", "based on your check, which confirmed it"),
    "claim": ("unconfirmed", "based on your check: still one person's account"),
    "rumour": ("unsupported", "based on your check: still unverified"),
    "debunked": ("confirmed", "based on your check, which exposed a fake"),
}
GOES_BEYOND_BELOW = 0.3   # support from the source note below this: "may say more than your source"


def _review(headline):
    notes = notebook()
    key = json.dumps([headline.strip(), [(n["text"], n["status"]) for n in notes], sorted(case()["NAMES"])])
    cached = _cache_get("review", key)
    if cached is None:
        cached = review(headline, notes, case()["NAMES"])
        _cache_put("review", key, cached)
    return json.loads(json.dumps(cached))  # a fresh copy


def editor_review(item, verified, headline):
    result = _review(headline)
    claims = result["claims"]
    if verified:
        level, verdict = CHECKED_VERDICTS[item["verify"]["status"]]
    elif item.get("skip_judge"):
        level, verdict = PRESET_VERDICTS[item["status"]]
    else:
        level, verdict = SOURCE_VERDICTS[story()["judgements"][_find_key(item)]]
    source_note = item["verify"]["note"]["text"] if verified else item["note"]
    advice = []
    support = next((e["support"] for e in claims["per_entry"] if e["text"] == source_note), None)
    if support is not None and support < GOES_BEYOND_BELOW:
        advice.append("Your headline may say more than your source does. Check that every detail is in it.")
    clash = [e for e in claims["per_entry"]
             if e["status"] == "confirmed" and e["text"] != source_note and e["contradiction"] >= CONTRADICTION_THRESHOLD]
    if clash:
        advice.append(f"It may clash with a confirmed note: \u201c{clash[0]['text']}\u201d")
    if item.get("anonymous") and item["anonymous"] in _named_people(headline):
        advice.append("You name a source who asked not to be named.")
    result["claims"] = {**claims, "level": level, "verdict": verdict, "source": source_note,
                        "source_support": support}
    result["advice"] = advice
    return result


def _find_key(item):
    return _key(item["id"], story()["cycle"])


def check(key, headline):
    item, verified = _find(key)
    if not verified and not item.get("skip_judge") and key not in story()["judgements"]:
        raise ValueError("First decide how reliable this source is.")
    return editor_review(item, verified, headline)


def _add_harm(pid, source, amount):
    h = story()["harm"][pid]
    h[source] = max(0.0, h[source] + amount)
    total = h["you"] + h["others"]
    if total > 100:
        h[source] -= total - 100


def _reduce_harm(pid, amount):
    h = story()["harm"][pid]
    from_you = min(h["you"], amount)
    h["you"] -= from_you
    h["others"] = max(0.0, h["others"] - (amount - from_you))


def publish(key, headline, unconfirmed):
    item, verified = _open(key)
    s = story()
    result = editor_review(item, verified, headline)
    dramatic = bool(result["flags"])
    named = _named_people(headline)

    debunk = verified and item["verify"]["status"] == "debunked"
    effects = item["verify"] if debunk else item
    pressure_before = _pressure_snapshot()
    truth = "true" if debunk else item["truth"]

    readers = item["news_value"]
    readers *= READERS_SENSATIONAL if dramatic else 1
    readers *= READERS_NAMED if named else 1
    readers *= READERS_UNCONFIRMED if unconfirmed else 1

    multiplier = (HARM_UNCONFIRMED if unconfirmed else 1) * (HARM_SENSATIONAL if dramatic else 1)
    for pid, amount in effects.get("harm", {}).items():
        _add_harm(pid, "you", amount * multiplier)
    for pid, amount in effects.get("harm_if_named", {}).items():
        if pid in named:
            _add_harm(pid, "you", amount * (HARM_SENSATIONAL if dramatic else 1))
    for pid, amount in effects.get("harm_reduce", {}).items():
        _reduce_harm(pid, amount)
    for pid, share in item.get("harm_reduce_you_share", {}).items():
        s["harm"][pid]["you"] *= (1 - share)

    before = state["credibility"]
    quoted = f"\u201c{_short(headline)}\u201d"
    status = _current_status(item, verified)
    if debunk:
        _trust(TRUST_DEBUNK, f"{quoted} exposes a fake.", key)
    else:
        change = (TRUST_BY_STATUS_LABELLED if unconfirmed else TRUST_BY_STATUS)[status]
        reasons = {"confirmed": "is based on confirmed facts.",
                   "rumour": "spreads an unverified rumour, even with a label." if unconfirmed
                             else "presents an unverified rumour as news."}
        _trust(change, f"{quoted} {reasons.get(status, '')}", key)
    if item.get("credibility_on_publish"):
        _trust(item["credibility_on_publish"], "You published a correction.", key)

    state["readers"] += readers
    s["readers"] += readers
    _interest(readers / 1000 * INTEREST_PER_1000_READERS)

    context_item = {**item, **({"context": item["verify"]["context"]}
                               if debunk and "context" in item["verify"] else {})}
    harsh = dramatic or (status == "rumour" and not unconfirmed)
    comments, source = reader_comments(context_item, headline, unconfirmed, harsh)
    article = {
        "id": item["id"], "key": key, "cycle": s["cycle"], "truth": truth, "debunk": debunk,
        "cycle_label": case()["cycles"][s["cycle"] - 1]["label"], "headline": headline,
        "unconfirmed": unconfirmed, "readers": round(readers), "credibility": state["credibility"] - before,
        "pressure": _pressure_changes(pressure_before),
        "named": [case()["PEOPLE"][p]["name"] for p in named], "comments": comments, "comments_source": source,
    }
    s["published"].append(article)
    s["decisions"][key] = {"action": "publish", "headline": headline, "unconfirmed": unconfirmed}
    _changed()
    return {"article": article, "review": result}


def verify(key):
    item, verified = _open(key)
    if verified or "verify" not in item or story()["cycle"] >= len(case()["cycles"]):
        raise ValueError("This item can't be checked any further.")
    story()["verifying"][item["id"]] = story()["cycle"] + 1
    story()["decisions"][key] = {"action": "verify"}
    _changed()
    return {"ok": True}


def hold(key):
    _open(key)
    story()["decisions"][key] = {"action": "hold"}
    _changed()
    return {"ok": True}


# ---------- End of a news cycle ----------

def end_cycle():
    c, s = case(), story()
    if s["finished"]:
        raise ValueError("This story is over. Move on to the next one.")
    cycle = s["cycle"]
    entries = _inbox_entries(cycle)
    keys = [_key(iid, cycle) for iid, _ in entries]
    if any(k not in s["decisions"] for k in keys):
        raise ValueError("Decide about every item on your desk first.")

    messages, competitor, judgements, recap = [], [], [], []
    names = {"confirmed": "Confirmed", "claim": "One source", "rumour": "Unverified"}
    for (iid, verified), key in zip(entries, keys):
        item, decision = c["ITEMS"][iid], s["decisions"][key]
        action = decision["action"]
        entry = {"kind": item["kind"], "from": item["from"], "action": action, "checked": verified,
                 "judgement": None, "notes": [], "pressure": [], "trust": 0, "readers": 0}

        # 1. How well the player judged the source (new items only)
        if not verified and not item.get("skip_judge"):
            yours, correct = s["judgements"][key], item["status"]
            s["judged_total"] += 1
            s["judged_right"] += yours == correct
            entry["judgement"] = {"yours": names[yours], "correct": names[correct],
                                  "right": yours == correct, "why": item.get("why", "")}
            judgements.append({"kind": item["kind"], "from": item["from"], **entry["judgement"]})

        # 2. What publishing it did (already applied when it was published)
        if action == "publish":
            article = next(a for a in s["published"] if a["key"] == key)
            entry.update(headline=article["headline"], unconfirmed=article["unconfirmed"],
                         readers=article["readers"], pressure=list(article["pressure"]))
            for t in s["trust_log"]:
                if t["key"] == key:
                    entry["trust"] += t["change"]
                    entry["notes"].append(t["reason"])

        # 3. What holding it back or checking it did
        before = _pressure_snapshot()
        if action == "hold" and "on_hold" in item:
            effect = item["on_hold"]
            softer = effect.get("softer_if_published")
            if softer and any(p["id"] == softer["item"] for p in s["published"]):
                effect = softer
            _trust(effect.get("credibility", 0), f"You didn't publish the {item['kind'].lower()} from {item['from']}.", key)
            entry["trust"] += effect.get("credibility", 0)
            for pid, amount in effect.get("harm", {}).items():
                _add_harm(pid, "you", amount)
            messages.append(effect["message"])
            entry["notes"].append(effect["message"])
        if action != "publish" and "competitor" in item and iid not in s["competitor_done"]:
            s["competitor_done"].add(iid)
            competitor.append(item["competitor"]["headline"])
            _interest(-INTEREST_SCOOPED)
            for pid, amount in item["competitor"].get("harm", {}).items():
                _add_harm(pid, "others", amount)
        entry["pressure"] += _pressure_changes(before)
        recap.append(entry)

    earlier = []
    for article in s["published"]:
        item = c["ITEMS"][article["id"]]
        if article["truth"] == "false" and article["key"] not in s["penalised"] \
                and cycle >= item.get("false_revealed_in", len(c["cycles"]) + 1):
            s["penalised"].add(article["key"])
            quoted = f"\u201c{_short(article['headline'])}\u201d"
            if article["unconfirmed"]:
                change = FALSE_STORY_PENALTY_LABELLED
                reason = "It spread a rumour that has now been shown to be false. The unconfirmed label limited the damage."
            else:
                change = FALSE_STORY_PENALTY
                reason = "It has now been shown to be false."
            _trust(change, f"{quoted} {reason}")
            messages.append(f"What you reported in {quoted} turned out to be false.")
            earlier.append({"headline": article["headline"], "cycle_label": article["cycle_label"],
                            "trust": change, "reason": reason})

    if entries or not c.get("presentation"):  # talk-only slides in the demo don't fade the story
        _interest(-INTEREST_FADE)  # every story fades a little each news cycle

    start = s["cycle_start"]
    last = cycle == len(c["cycles"])
    dropped = (not last) and not c.get("presentation") and cycle >= DROP_AFTER_CYCLE and s["interest"] < INTEREST_THRESHOLD
    report = {
        "cycle_label": c["cycles"][cycle - 1]["label"], "slide": c["cycles"][cycle - 1].get("slide"),
        "articles": [a for a in s["published"] if a["cycle"] == cycle],
        "competitor": competitor, "competitor_name": c["competitor"], "messages": messages,
        "readers_now": round(state["readers"]), "credibility_now": round(state["credibility"]),
        "readers_change": round(state["readers"] - start["readers"]),
        "credibility_change": round(state["credibility"] - start["credibility"]),
        "trust_log": list(s["trust_log"]), "judgements": judgements, "recap": recap, "earlier": earlier,
        "interest": round(s["interest"]), "interest_change": round(s["interest"] - start["interest"]),
        "interest_threshold": INTEREST_THRESHOLD, "dropped": dropped,
        "people": [{**_person_view(pid),
                    "change": round(min(100, h["you"] + h["others"]) -
                                    min(100, start["harm"][pid]["you"] + start["harm"][pid]["others"])),
                    "change_you": round(h["you"] - start["harm"][pid]["you"]),
                    "change_others": round(h["others"] - start["harm"][pid]["others"])}
                   for pid, h in s["harm"].items()],
    }

    _record(c["cycles"][cycle - 1]["label"])
    _changed()
    if last or dropped:
        s["finished"] = True
        s["ended"] = "resolved" if last else "dropped"
        s["epilogue"] = epilogue()
        state["history"].append({"title": c["title"], "ended": s["ended"], "cycles": cycle,
                                 "readers": round(s["readers"]),
                                 "harm_you": s["epilogue"]["harm_you"],
                                 "judged_right": s["judged_right"], "judged_total": s["judged_total"]})
        return {"report": report, "epilogue": s["epilogue"], "game": view()}
    s["cycle"] += 1
    _snapshot()
    return {"report": report, "game": view()}


def epilogue():
    c, s = case(), story()
    accuracy = {"true": "Accurate", "false": "False",
                "unproven": "Turned out true, but there was no evidence when you published"}
    pages = [{"cycle_label": a["cycle_label"], "headline": a["headline"], "unconfirmed": a["unconfirmed"],
              "accuracy": accuracy[a["truth"]]} for a in s["published"]]
    people = []
    for pid, p in c["PEOPLE"].items():
        pv = _person_view(pid)
        people.append({**pv, "outcome": next(o["text"] for o in p["outcomes"] if pv["total"] <= o["max"])})
    ended = s["ended"]
    return {
        "ended": ended, "story_title": c["title"],
        "ending_note": (f"The story ran until {c.get('final_label', 'the final news')}." if ended == "resolved" else
                        f"Readers lost interest after {c['cycles'][s['cycle'] - 1]['label']}, so your editor moved you "
                        f"to another story. The Chronicle never reported {c.get('final_label', 'how it ended')}."),
        "truth": c["truth"], "front_pages": pages, "people": people,
        "readers": round(s["readers"]), "credibility": round(state["credibility"]),
        "harm_you": sum(p["you"] for p in people), "harm_others": sum(p["others"] for p in people),
        "judged_right": s["judged_right"], "judged_total": s["judged_total"],
        "timeline": list(s["timeline"]),
    }


def career_summary():
    return {"stories": state["history"], "readers": round(state["readers"]),
            "credibility": round(state["credibility"]),
            "harm_you": sum(h["harm_you"] for h in state["history"]),
            "judged_right": sum(h["judged_right"] for h in state["history"]),
            "judged_total": sum(h["judged_total"] for h in state["history"]),
            "timeline": list(state["career_timeline"])}


# ---------- AI: readers' reactions ----------

COMMENTS_SCHEMA = {
    "type": "object",
    "properties": {"comments": {"type": "array", "items": {
        "type": "object",
        "properties": {"reader": {"type": "string"}, "text": {"type": "string"}},
        "required": ["reader", "text"]}}},
    "required": ["comments"],
}

COMMENTS_PROMPT = """You write realistic online reader comments for a local news website in a small Dutch village, for an educational game about how news shapes public opinion.
Write exactly 3 comments, one from each reader listed, in English.
Each comment is 1 or 2 short sentences and reacts to the headline exactly as it is written.
Readers tend to believe what a headline suggests. If it is labelled UNCONFIRMED, some readers are more careful.
Opinions can be strong, but never write insults, slurs, threats or calls for violence, and never add new facts about the story."""


def reader_comments(item, headline, unconfirmed, harsh):
    """Three reactions to the player's exact headline. Falls back to the case file."""
    pool = case()["readers"]
    count = len(story()["published"])
    readers = [pool[(count + i) % len(pool)] for i in range(3)]
    cache_key = json.dumps([case()["id"], item["id"], headline.strip(), bool(unconfirmed)])
    cached = _cache_get("comments", cache_key)
    if cached:
        return cached, "cached"
    if ollama is not None:
        try:
            listing = "\n".join(f"- {r['name']}, {r['about']}" for r in readers)
            message = (f"Headline: {headline}\nLabel: {'UNCONFIRMED' if unconfirmed else 'none'}\n"
                       f"What the article is based on: {item['context']}\nReaders:\n{listing}")
            response = CLIENT.chat(model=MODEL, format=COMMENTS_SCHEMA,
                                   options={"temperature": 0.8, "num_predict": 300},
                                   messages=[{"role": "system", "content": COMMENTS_PROMPT},
                                             {"role": "user", "content": message}])
            data = json.loads(response["message"]["content"])
            names = {r["name"] for r in readers}
            comments = [{"reader": x["reader"].strip(), "text": x["text"].strip()[:280]}
                        for x in data.get("comments", []) if x.get("text") and x.get("reader", "").strip() in names]
            if comments:
                _cache_put("comments", cache_key, comments[:3])
                return comments[:3], "ollama"
        except Exception:
            pass
    return item["fallback"]["sensational" if harsh else "careful"], "scripted"


def ollama_status():
    if ollama is None:
        return False, "The 'ollama' Python package is not installed. Run: pip install ollama"
    try:
        models = STATUS_CLIENT.list()["models"]
    except Exception as error:
        return False, f"Ollama is not running ({type(error).__name__}). Start the Ollama app"
    names = [getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None) for m in models]
    if not any(n and n.startswith(MODEL) for n in names):
        return False, f"Model '{MODEL}' is not downloaded. Run: ollama pull {MODEL}"
    return True, f"Ollama is running with {MODEL}"


new_game()
