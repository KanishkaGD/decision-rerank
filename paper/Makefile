# Build the paper (paper/ of calibrated-decision-models-for-ir).
#
#   make            build main.pdf (pdflatex + bibtex, as many runs as needed)
#   make watch      rebuild automatically whenever a source file changes
#   make check      list LaTeX errors, undefined references/citations, overfull boxes
#   make wordcount  rough word count of the section files
#   make clean      remove auxiliary files (keeps main.pdf)
#   make distclean  remove auxiliary files and main.pdf
#
# Requires a TeX distribution with latexmk (TeX Live: texlive-latex-extra,
# texlive-bibtex-extra, latexmk). Settings for latexmk are in ./latexmkrc.

MAIN     := main
LATEXMK  := latexmk
FLAGS    := -pdf -interaction=nonstopmode -file-line-error -halt-on-error
SOURCES  := $(MAIN).tex rq-macros.tex $(wildcard sections/*.tex) $(wildcard tables/*.tex) $(wildcard *.bib) \
            llncs.cls splncs04.bst $(wildcard images/*) $(wildcard figures/*)

.PHONY: all pdf watch check wordcount clean distclean

all: pdf

pdf: $(MAIN).pdf

$(MAIN).pdf: $(SOURCES)
	$(LATEXMK) $(FLAGS) $(MAIN).tex

watch:
	$(LATEXMK) $(FLAGS) -pvc -view=none $(MAIN).tex

check: $(MAIN).pdf
	@echo "--- errors ---"
	@grep -E "^!|^\./.*:[0-9]+:" $(MAIN).log || echo "none"
	@echo "--- undefined references / citations ---"
	@grep -qE "undefined|multiply defined" $(MAIN).log \
	  && grep -E "undefined|multiply defined" $(MAIN).log | sort -u || echo "none"
	@echo "--- BibTeX warnings ---"
	@grep -E "^Warning--" $(MAIN).blg || echo "none"
	@echo "--- overfull boxes ---"
	@grep -E "^Overfull" $(MAIN).log || echo "none"
	@echo "--- pages ---"
	@grep -E "Output written" $(MAIN).log

wordcount:
	@cat sections/*.tex | sed -e 's/%.*//' | tr -s '[:space:]' '\n' \
	  | grep -vE '^\\|^[{}]' | wc -w

clean:
	$(LATEXMK) -c $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).run.xml $(MAIN).synctex.gz

distclean: clean
	rm -f $(MAIN).pdf
