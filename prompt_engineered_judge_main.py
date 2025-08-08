import os, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

MODEL_NAME = "gpt-4.1-mini"

sysPrompt = (
    "You are a strict Translation Judge for English to FIlipino, return ONLY JSON that matches the schema, follow the rubric EXACTLY"
)
def promptMsg(english, filipino, domain="general", reference=None):
    return f"""
You are judging an English to Filipino translation for the domain: {domain}. Follow this rubric exactly.

1. Accuracy – Does it correctly express the original meaning?
2. Fluency – Is the Filipino natural and grammatically correct?
3. Coherence – Does it flow clearly and logically?
4. Cultural Appropriateness – Does it match Filipino cultural expressions and respect (e.g., use of "po", avoiding awkward literal translations)?
5. Guideline Adherence – Does it follow the expected language style and domain rules?
6. Completeness – Are all ideas translated without missing or adding information?

Reliability, do this silently, in order:
- Check Accuracy first, then Completeness, then Guideline Adherence, then Cultural Appropriateness, then Fluency, then Coherence
- Penalize any wrong meaning, added facts, dropped facts, wrong terms, wrong tone for the domain
- If reference is given, use it to judge meaning only, do not punish valid paraphrase, do not copy the wording, do not adjust at ALL, use what's given
- Ignore length, emojis, fancy style, formatting, model names, extra words, author identity

Scoring rules:
- Each criterion is 0 or 1 with a 1-sentence reason
- raw_score = sum of the six 0/1 values, range 0–6
- final_score buckets: 5–6 == 5, 3–4 == 3, 0–2 == 1
- label mapping: 5 as "excellent", 3 as "good", 1 as "poor"
- If accuracy or completeness is 0 due to a big meaning error, do not output "excellent"

Output rules:
- Return JSON only, no extra prose
- Be concise in explanations, one sentence each, no chain-of-thought, no step lists

Here are some examples (DO NOT COPY TEXT, DO NOT OUTPUT THESE, THESE ARE STRICTLY FOR EXAMPLE ONLY):

---

**Example 1:**

English: "Take two tablets daily after meals."

Filipino: "Uminom ng dalawang tableta araw-araw pagkatapos kumain."

Evaluation:
- Accuracy: 1 – Correct meaning captured
- Fluency: 1 – Natural phrasing
- Coherence: 1 – Clear and understandable
- Cultural: 1 – Appropriate phrasing for Filipino
- Guideline: 1 – Matches medical language tone
- Completeness: 1 – No omissions

**Raw Score**: 6
**Final Score**: 5  
**Label**: Excellent  
**Explanation**: The translation is accurate, fluent, and fully aligned with medical tone and structure.

-------------

**Example 2:**

English: "Avoid using expired medicine."

Filipino: "Iwasan gamitin ang gamot."

Evaluation:
- Accuracy: 0 – “Expired” not mentioned
- Fluency: 1 – Grammatically correct
- Coherence: 1 – Sentence flows logically
- Cultural: 1 – Sounds Filipino
- Guideline: 0 – Omits critical word, not precise
- Completeness: 0 – Important info missing

**Raw Score**: 3
**Final Score**: 3  
**Label**: Good  
**Explanation**: While it sounds fluent and coherent, it lacks the crucial idea of “expired,” which affects accuracy and completeness.

-------------

Now judge this pair:

English: {english}
Filipino: {filipino}

""" + (f"\nReference (optional):\n{reference}\n" if reference else "") + """
Return JSON with this exact shape:
{
  "per_criterion": {
    "accuracy": 0 or 1,
    "fluency": 0 or 1,
    "coherence": 0 or 1,
    "cultural_appropriateness": 0 or 1,
    "guideline_adherence": 0 or 1,
    "completeness": 0 or 1
  },
  "explanations": {
    "accuracy": "one sentence",
    "fluency": "one sentence",
    "coherence": "one sentence",
    "cultural_appropriateness": "one sentence",
    "guideline_adherence": "one sentence",
    "completeness": "one sentence"
  },
  "raw_score": 0-6,
  "final_score": 1 or 3 or 5,
  "label": "poor" or "good" or "excellent",
  "overall_explanation": "2-4 sentences summarizing the decision, biggest strengths or errors, mention domain rules if relevant"
}
"""

def _bucket(raw):  # 0–6 → 1/3/5
    return 5 if raw >= 5 else 3 if raw >= 3 else 1

def judge(english, filipino, domain="general", reference=None, seed=123):
    r = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        seed=seed,
        messages=[
            {"role":"system","content": sysPrompt},
            {"role":"user","content": promptMsg(english, filipino, domain, reference)}
        ],
        response_format={"type":"json_object"}
    )
    data = json.loads(r.choices[0].message.content)

    # recomputes to be sure
    keys = ["accuracy","fluency","coherence","cultural_appropriateness","guideline_adherence","completeness"]
    raw = sum(int(data["per_criterion"][k]) for k in keys)
    data["raw_score"] = raw
    data["final_score"] = _bucket(raw)
    data["label"] = {5:"excellent", 3:"good", 1:"poor"}[data["final_score"]]

    if not data.get("overall_explanation"):
        data["overall_explanation"] = "; ".join(f"{k}: {data['explanations'][k]}" for k in keys)[:900]
    return data

if __name__ == "__main__":
    out = judge("This is a sample", "Ito ay isang sample", domain="medical")
    print(json.dumps(out, ensure_ascii=False, indent=2))