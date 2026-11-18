install:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python -c "import nltk; nltk.download('wordnet')"
run:
	uvicorn src.api:app --reload
cli:
	python cli.py "$(q)"
test:
	pytest -q
eval:
	python eval/run_eval.py
ui:
	streamlit run streamlit_app.py