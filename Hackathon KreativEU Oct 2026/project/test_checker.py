"""
Type your own headlines and see what the AI editor says.
Run:  python test_checker.py      (type 'q' to quit)
Keep this file in the same folder as headline_checker.py.
"""
from headline_checker import MADE_FM_NOTEBOOK, review

# Edit headline_checker.MADE_FM_NOTEBOOK, or replace this line, to test other situations
NOTEBOOK = MADE_FM_NOTEBOOK

print("AI editor ready. Type a headline, or 'q' to quit.\n")
while True:
    headline = input("Headline: ").strip()
    if headline.lower() == "q":
        break
    if not headline:
        continue

    result = review(headline, NOTEBOOK)

    print("  Tone:")
    for label, score in result["tone"].items():
        print(f"    {label:<40} {score:.2f}")
    print(f"  Names mentioned: {', '.join(result['names']) or 'none'}")
    claims = result["claims"]
    print(f"  Claim check: {claims['verdict']} (score {claims['score']})")
    if claims["level"] == "contradicted":
        print(f"    Ruled out by: \"{claims['source']}\"")
    print("  Per note (supports / contradicts):")
    for e in claims["per_entry"]:
        print(f"    {e['support']:.2f} / {e['contradiction']:.2f}  [{e['status']}] {e['text']}")
    print()
