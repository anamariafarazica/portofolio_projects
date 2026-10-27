# The Midnight Chronicle

**A journalism game about truth, speed and people.**
Made at the KreativEU Hackathon by Ana, Andre, Vladimir and Frederico.

You're a journalist at the local newspaper of a small town, and you decide what
gets published next. Information keeps arriving: police statements, emails,
phone calls, posts online. For each one, you judge how reliable it is, then
choose to publish it, check it first, or hold it back. If you publish, you write
the headline yourself. Readers react to exactly what you wrote, and your choices
change how many people read your paper, how much they care about the story, how
much they still trust you, and how much pressure your reporting puts on the real
people in the story.

The Midnight Chronicle is a learning tool for students from middle school to
high school (ages 14 to 18), the age when they start shaping their opinions from
the news. Teachers can use it in media literacy and citizenship lessons, as one
20 to 30 minute session followed by a class discussion.

---

## Features

- **Title screen and case board.** Each case is a dossier. Players pick a case and
  can switch between cases at any time; each case keeps its own progress.
- **Judging sources.** Every new item must be labelled *Confirmed*, *One source*
  or *Unverified* before acting on it. Feedback, with the reason, comes after the
  edition is printed.
- **Publish, check or hold back.** Checking an item takes a news cycle and can
  confirm or debunk it; meanwhile a competitor may publish it anyway.
- **The AI editor.** Gives advice on every headline: dramatic wording, blaming a
  named person, naming a source who asked to stay anonymous, and how the notes
  relate to the headline. It never blocks the player.
- **Readers' reactions.** After printing, readers comment on the exact headline,
  written by a local AI model (or by pre-written lines if the AI isn't running).
- **Readers, public trust and story interest.** Trust is based on what each story
  really rests on, and every change is explained. If story interest falls too
  low, the editor drops the story before its final news.
- **Media pressure on people.** A graded scale from low (green) to severe (red)
  for every person in the story, split into pressure from your reporting and
  from other outlets.
- **Corrections.** Publishing something false can trigger a correction request.
- **The notebook.** An agenda-style notebook with every note and front page, per
  case, grouped by day and time.
- **Endings and charts.** Each case ends with what really happened, where every
  person ended up, and charts of trust, readers, story interest and pressure.
  The career summary covers all cases played.

## The cases

| File | Case | News cycles |
| --- | --- | --- |
| `cases/luigi_fm.json` | The death at Luigi FM, the village radio station in Made | up to 5 |
| `cases/carnival_hall.json` | The council's vote on the 1928 Carnival Hall | up to 3 |
| `cases/kreativeu_hackathon.json` | Rumours at the KreativEU Hackathon | up to 3 |

---

## Project structure

| File | What it does |
| --- | --- |
| `game.html` | The game: title screen, case board, desk, reports, notebook, endings |
| `newsroom.py` | The game rules: judging, publishing, checking, readers, trust, interest, pressure, endings; AI reader comments; the AI cache |
| `headline_checker.py` | The AI editor: tone, dramatic wording, names, and how notes support or contradict a headline |
| `server.py` | Runs everything locally (Python standard library) |
| `cases/*.json` | The stories, one file per case |
| `warm_cache.py` | Prepares a demo by saving AI results in `ai_cache.json` |
| `dashboard.html` | Test desk for trying headlines against any notebook |
| `test_checker.py` | Terminal version of the test desk |
| `requirements.txt` | Python packages |

---

## Requirements

- Python 3.10, 3.11 or 3.12
- About 4 GB of free disk space for the two AI models
- 8 GB of RAM works; 16 GB is comfortable
- [Ollama](https://ollama.com/download) for the readers' reactions (optional:
  without it, the game uses pre-written comments)

## Setup (once)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Mac / Linux

# 2. Install the Python packages
pip install -r requirements.txt
#    No GPU? This smaller PyTorch install works too:
#    pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Download the AI editor's model (about 1.6 GB, once)
python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"

# 4. Install Ollama, start it, and download the small model (about 2 GB)
ollama pull llama3.2:3b
```

## Run

```bash
python server.py
```

Wait for `Game running at http://127.0.0.1:8000`, then open
**http://127.0.0.1:8000** in your browser. The headline test desk is at
http://127.0.0.1:8000/dashboard. Press Ctrl+C in the terminal to stop.

To use a different Ollama model, set `OLLAMA_MODEL` before starting
(`set OLLAMA_MODEL=gemma3:4b` on Windows, `export OLLAMA_MODEL=gemma3:4b` on
Mac / Linux) and pull it first with `ollama pull gemma3:4b`.

## Preparing a demo

The AI editor's advice and the readers' comments are saved in `ai_cache.json`
and reused for the same headline, so a rehearsed demo loads instantly.

1. Stop `server.py`.
2. With Ollama running, run `python warm_cache.py` (the scripted demo in the
   KreativEU case), or `python warm_cache.py luigi_fm` for any case by its id.
3. Start `server.py` again.

Only the exact steps and headlines in `warm_cache.py` are cached; anything else
runs live, which takes a few seconds per step.

---

## How the rules work

All numbers are at the top of `newsroom.py` and in the case files, so they can
be tuned without changing any logic.

- **Readers:** each item's `news_value`, more for dramatic or naming headlines,
  less when labelled unconfirmed.
- **Public trust:** measures accuracy, not tone, and is based only on what each
  story really rests on (set in the case files), never on the AI. It rises for
  stories based on confirmed facts, exposing fakes and corrections; it drops for
  presenting rumours as news, stories later proven false, staying silent on
  important news, or ignoring a correction request. Trust carries over between
  cases.
- **Story interest:** articles raise it, it fades every news cycle, and scoops by
  the competitor lower it. Below the threshold (25), after at least two news
  cycles, the story is dropped.
- **Media pressure** (called `harm` in the code and case files): from each item's
  `harm` and `harm_if_named` values. Dramatic headlines add more, an unconfirmed
  label halves it, and naming someone always counts. Stories you held back can
  still be published by the competitor; that pressure counts as "other outlets".

## How the AI is used

- **AI editor** (`facebook/bart-large-mnli`, local): reads the headline's tone,
  checks it against the player's notes, and gives advice. Dramatic-wording
  warnings name the exact words that triggered them.
- **Readers' reactions** (`llama3.2:3b` through Ollama, local): three short
  comments on the published headline. The prompt forbids insults, threats,
  slurs and invented facts.
- **What the AI never does:** decide what's true, set the scores, or block the
  player. People wrote the stories, the rules and the outcomes.
- Everything runs on the laptop itself; no player data is sent anywhere.

---

## Writing your own case

Copy a file in `cases/`, give it a new `id` and `order`, and rewrite the content.
The game picks it up after a restart.

**Case:** `id`, `order`, `title`, `teaser` (shown on the case board), `intro`,
`final_label`, `outlet`, `competitor`, `start_credibility`, `truth` (what really
happened), `readers` (personas for the AI comments), `people` (with `aliases`
and three `outcomes` by pressure level), and `cycles`.

**News cycle:** `label` (for example `"Wednesday, 07:30"`; the time sets the
light in the office window), `intro`, and `items`.

**Item:**

| Field | Meaning |
| --- | --- |
| `kind`, `from`, `text` | What the player sees |
| `status` | `confirmed`, `claim` or `rumour`; hidden from the player, who judges it |
| `why` | One sentence explaining the right judgement, shown after printing |
| `note` | The fact as it goes into the notebook (one simple sentence) |
| `context` | A short description for the AI reader comments |
| `truth` | `true`, `false` or `unproven`; `false_revealed_in` gives the cycle when a false story is exposed |
| `news_value` | Readers when published |
| `harm`, `harm_if_named`, `harm_reduce` | Pressure on people when published |
| `on_hold` | What happens if the player doesn't publish it |
| `competitor` | What the competitor publishes if the player doesn't |
| `verify` | What checking reveals next cycle (`status` can become `debunked`) |
| `fallback` | Pre-written reader comments, `careful` and `sensational` |

Optional item fields: `request` and `anonymous` (a source who asked not to be
named), `requires_published_any` (only appears if certain items were published,
for example a correction letter), `credibility_on_publish`,
`harm_reduce_you_share`, and `headline` (a pre-filled headline).

---

## Troubleshooting

- **The page keeps loading, or says the server is out of date:** an old server is
  probably still running. Close every terminal running `server.py` (or restart
  the laptop), then start it again from the project folder. The startup message
  shows the folder and engine version it's using.
- **"Port 8000 is already in use":** same cause as above.
- **"Ollama is not running":** start the Ollama app. The game still works with
  pre-written comments.
- **Slow answers:** the first answer after starting loads the model and can take
  10 to 30 seconds. Use `warm_cache.py` before a demo, or a smaller model such
  as `llama3.2:1b`.
- **Fonts look plain:** the fonts (UnifrakturMaguntia, Fredoka, Special Elite,
  Literata, Atkinson Hyperlegible) load from Google Fonts when online. Offline,
  the game falls back to system fonts and works the same.
