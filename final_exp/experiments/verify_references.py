"""Phase-0 reference verification helper.

Queries OpenAlex for each reference title and writes the top matches to
``results/reference_check.json``.  This does not replace opening the publisher page; it
is a triage tool to flag entries that need manual verification.

Usage:
    python experiments/verify_references.py
"""
import json
import os
import time
import urllib.parse
import urllib.request

REFS = [
    "Edge Service Caching With Delayed Hits: Minimizing Latency and Reducing Cost",
    "Dynamic regret of randomized online service caching in edge computing",
    "Amortized efficiency of list update and paging rules",
    "Competitive paging algorithms",
    "On-line file caching",
    "A primal-dual randomized algorithm for weighted paging",
    "Online Computation and Competitive Analysis",
    "The design of competitive online algorithms via a primal-dual approach",
    "Competitive snoopy caching",
    "Competitive randomized algorithms for nonuniform problems",
    "Caching with delayed hits",
    "Online convex programming and generalized infinitesimal gradient ascent",
    "Introduction to online convex optimization",
    "Online learning and online convex optimization",
    "Dominant resource fairness: Fair allocation of multiple resource types",
    "Multi-resource packing for cluster schedulers",
    "Cloud programming simplified: A Berkeley view on serverless computing",
    "SOCK: Rapid task provisioning with serverless-optimized containers",
    "Catalyzer: Sub-millisecond startup for serverless computing with initialization-less booting",
    "Google cluster-usage traces: format + schema",
    "The Grid Workloads Archive",
    "Online knapsack problems",
    "Vector bin packing",
    "Caching with fetch delay",
    "Serverless cold start",
    "Edge caching survey",
    "Online caching with bypassing",
    "Multi-resource allocation online",
    "Deep reinforcement learning for caching",
    "Competitive analysis of caching",
    "Ski rental problem",
    "Online file caching with rejection",
    "Paging with delayed hits",
    "Edge service placement",
    "Latency-critical edge caching",
    "Cost-aware caching",
    "Online algorithms for knapsack",
    "Randomized rounding dependent rounding",
    "Online convex optimization with switching costs",
]


def search(title, rows=3):
    q = urllib.parse.quote(title)
    url = f"https://api.openalex.org/works?search={q}&per-page={rows}"
    req = urllib.request.Request(url, headers={"User-Agent": "phase0-ref-check/1.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)
    out = []
    for w in data.get("results", []):
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
        authors = [a.get("author", {}).get("display_name") for a in w.get("authorships", [])[:3]]
        out.append({
            "title": w.get("display_name"),
            "year": w.get("publication_year"),
            "doi": w.get("doi"),
            "venue": src,
            "authors": authors,
        })
    return out


def main():
    here = os.path.dirname(__file__)
    out_path = os.path.join(here, "..", "results", "reference_check.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = []
    for i, title in enumerate(REFS, 1):
        try:
            matches = search(title)
            results.append({"query": title, "matches": matches})
            top = matches[0] if matches else {}
            print(f"[{i:02d}] {title}")
            print(f"     -> {top.get('title')} ({top.get('year')}) {top.get('doi')}")
        except Exception as e:  # network / API error
            results.append({"query": title, "matches": [], "error": str(e)})
            print(f"[{i:02d}] {title} -> ERROR: {e}")
        time.sleep(0.4)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()