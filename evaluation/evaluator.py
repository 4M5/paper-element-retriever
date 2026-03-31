"""
evaluation/evaluator.py

6-metric RAG evaluation suite using Ollama as judge.
All metrics scored 0.0 → 1.0.

Metrics:
  Faithfulness        — answer only uses retrieved context (anti-hallucination)
  Answer Relevancy    — answer addresses the question
  Context Precision   — best chunks ranked first
  Context Recall      — context covers the ground truth answer
  Context Relevancy   — retrieved chunks are relevant to query
  Answer Correctness  — answer matches ground truth
"""

import json, time, os
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv

from ingestion.vector_store import retrieve, get_all_chunks
from query.rag_engine import answer
from utils.ollama_client import chat

load_dotenv()

# ── Test set generation ────────────────────────────────────────────────────

QA_PROMPT = """Given this research paper excerpt, generate one question-answer pair
that tests factual understanding of the content.

Excerpt:
{chunk}

Respond ONLY in this JSON format (no markdown, no extra text):
{{"question": "...", "answer": "..."}}
"""


def generate_test_set(session_id: str, n: int = 10) -> List[Dict]:
    chunks = get_all_chunks(session_id)
    if not chunks:
        raise ValueError("No papers ingested yet.")

    step     = max(1, len(chunks) // n)
    selected = chunks[::step][:n]
    cases    = []

    print(f"[Eval] Generating {len(selected)} Q&A pairs...")
    for i, chunk in enumerate(selected):
        try:
            raw = chat(
                "Generate Q&A pairs from research text. Output only valid JSON.",
                QA_PROMPT.format(chunk=chunk["text"][:700]),
                temperature=0.3,
            )
            raw   = raw.strip().strip("```json").strip("```").strip()
            pair  = json.loads(raw)
            cases.append({
                "question":     pair["question"],
                "ground_truth": pair["answer"],
                "source_chunk": chunk["text"],
                "section":      chunk.get("section", "Unknown"),
                "filename":     chunk.get("filename", "Unknown"),
            })
        except Exception as e:
            print(f"  [Eval] Skipped chunk {i}: {e}")

    print(f"[Eval] {len(cases)} test cases ready")
    return cases


# ── Scoring prompts ────────────────────────────────────────────────────────

def _score(prompt: str) -> float:
    raw = chat("You are an evaluation judge. Output only a float between 0.0 and 1.0.", prompt)
    try:
        return max(0.0, min(1.0, float(raw.strip().split()[0].replace(",", "."))))
    except Exception:
        return 0.0


def score_faithfulness(question, answer_text, contexts):
    return _score(f"""Question: {question}
Answer: {answer_text}
Context: {chr(10).join(contexts)}

Score how many claims in the answer are supported by the context.
1.0 = fully supported. 0.0 = hallucinated. Output only a float.""")


def score_answer_relevancy(question, answer_text):
    return _score(f"""Question: {question}
Answer: {answer_text}

Score how well the answer addresses the question.
1.0 = fully addresses it. 0.0 = completely off-topic. Output only a float.""")


def score_context_precision(question, contexts):
    numbered = "\n".join([f"Chunk {i+1}: {c[:300]}" for i, c in enumerate(contexts)])
    return _score(f"""Question: {question}
Ranked chunks:
{numbered}

Score whether the most relevant chunks appear at the top.
1.0 = best chunks ranked first. 0.0 = worst ranked first. Output only a float.""")


def score_context_recall(question, ground_truth, contexts):
    return _score(f"""Question: {question}
Ground truth: {ground_truth}
Context: {chr(10).join(contexts)}

Score whether the context contains enough info to derive the ground truth.
1.0 = context fully covers ground truth. 0.0 = missing key info. Output only a float.""")


def score_context_relevancy(question, contexts):
    return _score(f"""Question: {question}
Context: {chr(10).join(contexts)}

Score how relevant the context is to the question.
1.0 = highly relevant. 0.0 = completely irrelevant. Output only a float.""")


def score_answer_correctness(question, ground_truth, answer_text):
    return _score(f"""Question: {question}
Ground truth: {ground_truth}
Generated answer: {answer_text}

Score factual correctness of generated answer vs ground truth.
1.0 = fully correct. 0.0 = completely wrong. Output only a float.""")


# ── Full evaluation runner ─────────────────────────────────────────────────

THRESHOLDS = {
    "faithfulness":       0.80,
    "answer_relevancy":   0.75,
    "context_precision":  0.70,
    "context_recall":     0.70,
    "context_relevancy":  0.70,
    "answer_correctness": 0.75,
}


class RAGEvaluator:
    def __init__(self, session_id: str):
        self.session_id = session_id

    def run(self, test_cases: List[Dict] = None, n: int = 10) -> Dict:
        if test_cases is None:
            test_cases = generate_test_set(self.session_id, n)

        print(f"\n[Eval] Scoring {len(test_cases)} cases...")
        scored = []

        for i, case in enumerate(test_cases):
            print(f"  [{i+1}/{len(test_cases)}] {case['question'][:60]}...")

            t0      = time.time()
            result  = answer(self.session_id, case["question"])
            latency = round(time.time() - t0, 3)

            contexts = retrieve(self.session_id, case["question"])
            ctx_texts = [c["text"] for c in contexts]

            scores = {
                "faithfulness":       score_faithfulness(case["question"], result["answer"], ctx_texts),
                "answer_relevancy":   score_answer_relevancy(case["question"], result["answer"]),
                "context_precision":  score_context_precision(case["question"], ctx_texts),
                "context_recall":     score_context_recall(case["question"], case["ground_truth"], ctx_texts),
                "context_relevancy":  score_context_relevancy(case["question"], ctx_texts),
                "answer_correctness": score_answer_correctness(case["question"], case["ground_truth"], result["answer"]),
            }

            scored.append({
                **case,
                "generated_answer": result["answer"],
                "citations":        result["citations"],
                "latency_seconds":  latency,
                "scores":           scores,
            })

        # Aggregate
        metrics = {}
        for m in THRESHOLDS:
            vals = [s["scores"][m] for s in scored]
            metrics[m] = {
                "mean": round(sum(vals) / len(vals), 3),
                "min":  round(min(vals), 3),
                "max":  round(max(vals), 3),
            }

        avg_latency = round(sum(s["latency_seconds"] for s in scored) / len(scored), 3)

        return {
            "session_id":      self.session_id,
            "evaluated_at":    datetime.now().isoformat(),
            "n_test_cases":    len(scored),
            "avg_latency_sec": avg_latency,
            "metrics":         metrics,
            "per_case":        scored,
        }

    def print_report(self, report: Dict):
        print("\n" + "═"*54)
        print("  RAG Evaluation Report — Research Paper RAG")
        print("═"*54)
        print(f"  Session    : {report['session_id']}")
        print(f"  Test cases : {report['n_test_cases']}")
        print(f"  Avg latency: {report['avg_latency_sec']}s")
        print()
        print(f"  {'Metric':<24} {'Mean':>6}  {'Min':>6}  {'Max':>6}  Status")
        print(f"  {'-'*52}")
        for m, thresh in THRESHOLDS.items():
            s      = report["metrics"][m]
            status = "PASS" if s["mean"] >= thresh else "FAIL"
            flag   = "  ← hallucination risk!" if m == "faithfulness" and s["mean"] < thresh else ""
            print(f"  {m:<24} {s['mean']:>6.3f}  {s['min']:>6.3f}  {s['max']:>6.3f}  {status}{flag}")
        print("═"*54 + "\n")

    def save_report(self, report: Dict, path: str = "eval_report.json"):
        import json
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Eval] Report saved → {path}")
