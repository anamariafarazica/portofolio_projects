"""
Prepare a demo: plays a case once and saves the AI editor's advice and the
readers' comments in ai_cache.json, so the same headlines load instantly later.

Run it with Ollama running, with server.py stopped:
    python warm_cache.py                 (every case)
    python warm_cache.py kreativeu_hackathon   (one case, by its id)

Each item is published with its ready-made "headline" if the case has one,
otherwise with the first sentence of the item. Only those exact headlines are
cached; any other headline is generated live.
"""
import sys

import newsroom

wanted = sys.argv[1:] or [c["id"] for c in newsroom.CASES]
for case_id in wanted:
    newsroom.new_game()
    newsroom.open_case(case_id)
    print(f"{case_id}:")
    while True:
        game = newsroom.view()
        for item in game["inbox"]:
            if item["decision"]:
                continue
            if not item["skip_judge"] and not item["verification"]:
                newsroom.judge(item["key"], "claim")
                if item["judge_only"]:
                    continue
            headline = item["suggested_headline"] or item["text"].strip('"').split(". ")[0][:90]
            newsroom.check(item["key"], headline)
            result = newsroom.publish(item["key"], headline, False)
            print(f"  {game['cycle_label']}: \u201c{headline}\u201d  (comments: {result['article']['comments_source']})")
        if "epilogue" in newsroom.end_cycle():
            break
print("Done. Comments marked 'scripted' mean Ollama wasn't running.")
