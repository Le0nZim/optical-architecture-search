.PHONY: all clean
all:
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
	BIBINPUTS=.: bibtex build/main
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
clean:
	rm -f build/main.aux build/main.bbl build/main.blg build/main.log build/main.out build/main.pdf
