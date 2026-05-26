setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m ipykernel install --user --name jbook --display-name "Python (jbook)"

complete:
	jupyter-book build . --all
update:
	jupyter-book build .
open:
	open _build/html/index.html
web:
	ghp-import -n -p -f _build/html