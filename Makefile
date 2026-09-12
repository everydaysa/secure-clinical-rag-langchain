# Every target runs through .venv/bin/python, so no activation is needed.
PY := .venv/bin/python
PIP := .venv/bin/pip

ROLE ?= claims_analyst
AS_USER ?= demo.user
Q ?= When is an MRI covered for low back pain?

.PHONY: help install ingest ingest-mock ask ask-mock check phi access audit demo test eval eval-mock clean-index clean-logs

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-14s %s\n", $$1, $$2}'
	@echo
	@echo '  Variables: ROLE=<role>  AS_USER=<id>  Q="<question>"'

install:  ## install dependencies into .venv
	$(PIP) install -r requirements.txt
	$(PY) -m spacy download en_core_web_sm

check:  ## verify config and model access
	$(PY) scripts/check_setup.py

ingest:  ## build the index with Azure embeddings
	$(PY) scripts/ingest.py

ingest-mock:  ## build the offline index
	LLM_PROVIDER=mock $(PY) scripts/ingest.py

ask:  ## ask a question: make ask ROLE=clinician Q="..."
	$(PY) scripts/ask.py --role $(ROLE) --user $(AS_USER) "$(Q)"

ask-mock:  ## same, offline
	LLM_PROVIDER=mock $(PY) scripts/ask.py --role $(ROLE) --user $(AS_USER) "$(Q)"

phi:  ## show the PHI redaction demo
	$(PY) scripts/phi_demo.py

access:  ## show what each role can retrieve
	$(PY) scripts/access_demo.py

audit:  ## summarise the audit log
	$(PY) scripts/audit_review.py

demo:  ## the full walkthrough, one command
	@echo "== 1. PHI redaction =="            && $(PY) scripts/phi_demo.py
	@echo "== 2. Role-based retrieval =="     && $(PY) scripts/access_demo.py
	@echo "== 3. Restricted policy, clinician ==" && \
		$(PY) scripts/ask.py --role clinician \
		"How many outpatient behavioral health visits before prior authorization?"
	@echo "== 4. Same question, claims analyst ==" && \
		$(PY) scripts/ask.py --role claims_analyst \
		"How many outpatient behavioral health visits before prior authorization?"
	@echo "== 5. Audit review =="             && $(PY) scripts/audit_review.py

test:  ## run the test suite (offline, no Azure needed)
	$(PY) -m pytest

eval:  ## run the evaluation set against Azure
	$(PY) scripts/evaluate.py

eval-mock:  ## run the retrieval-only checks offline
	LLM_PROVIDER=mock $(PY) scripts/evaluate.py

clean-index:  ## delete the vector indexes
	rm -rf .chroma

clean-logs:  ## delete the audit log
	rm -rf logs
