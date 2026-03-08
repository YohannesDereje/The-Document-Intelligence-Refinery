from pathlib import Path
import os
from dotenv import load_dotenv
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter

load_dotenv()
p = Path("data/TRP1 _ Week 2 Delivery - Exercise 2 - Asking Good Questions in an AI-First Work Environment.pdf")
rules = Path("rubric/extraction_rules.yaml")
tri = TriageAgent(rules_path=rules)
ext = ExtractionRouter(api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"), rules_path=rules)
profile = tri.process_pdf(p)
refined = ext.process_document(pdf_path=p, profile=profile)
out = Path(".refinery/profiles") / f"{p.stem}_single_run_refined.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(refined.model_dump_json(indent=2), encoding="utf-8")
print(f"[SINGLE_DOC_DONE] {p.name} pages={profile.total_pages} output={out}")
