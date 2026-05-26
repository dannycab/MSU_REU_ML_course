.PHONY: setup slides copy-slides complete update open serve-slides web d2

MARP = marp
D2 = d2
DAYS = day-00-intro day-01-data day-02-classification day-03-regression day-04-modeling
D2_FILES = $(wildcard slides/figures/d2/*.d2)
D2_SVGS = $(patsubst slides/figures/d2/%.d2,slides/figures/%.svg,$(D2_FILES))

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m ipykernel install --user --name jbook --display-name "Python (jbook)"

d2: $(D2_SVGS)

slides/figures/%.svg: slides/figures/d2/%.d2
	@command -v $(D2) >/dev/null 2>&1 || { echo "d2 not installed. Install from https://d2lang.com"; exit 1; }
	$(D2) --sketch $< $@

slides: d2
	cp activities/figures/*.png slides/figures/
	$(MARP) --pdf  slides/day-00-intro.md           -o slides/day-00-intro.pdf
	$(MARP) --pdf  slides/day-01-data.md            -o slides/day-01-data.pdf
	$(MARP) --pdf  slides/day-02-classification.md  -o slides/day-02-classification.pdf
	$(MARP) --pdf  slides/day-03-regression.md      -o slides/day-03-regression.pdf
	$(MARP) --pdf  slides/day-04-modeling.md        -o slides/day-04-modeling.pdf
	$(MARP) --image png slides/day-00-intro.md           -o slides/figures/day-00-preview.png
	$(MARP) --image png slides/day-01-data.md           -o slides/figures/day-01-preview.png
	$(MARP) --image png slides/day-02-classification.md -o slides/figures/day-02-preview.png
	$(MARP) --image png slides/day-03-regression.md     -o slides/figures/day-03-preview.png
	$(MARP) --image png slides/day-04-modeling.md       -o slides/figures/day-04-preview.png

copy-slides:
	mkdir -p _build/html/slides
	cp slides/*.pdf _build/html/slides/

complete: slides
	jupyter-book build . --all
	$(MAKE) copy-slides

update: slides
	jupyter-book build .
	$(MAKE) copy-slides

open:
	open _build/html/index.html

serve-slides:
	$(MARP) --html --server slides/

web:
	ghp-import -n -p -f _build/html
