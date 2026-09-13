.PHONY: test evaluate evaluate-live index docker-build docker-up

test:
	python -m pytest tests/ -v

# Reads committed eval results -- no GROQ API key needed.
# To re-run against the live LLM: make evaluate-live  (requires GROQ_API_KEY in .env)
evaluate:
	mkdir -p reports
	python scripts/report_gate_eval.py

evaluate-live:
	python scripts/evaluate_grounded_gate.py

index:
	python scripts/index_documents.py

docker-build:
	docker build -t operations-assistant .

docker-up:
	docker compose up --build
