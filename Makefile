complete:
	jupyter-book build . --all
book:
	jupyter-book build .
web:
	ghp-import -n -p -f _build/html