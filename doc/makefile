.PHONY: clean
IN = memoria
FILES = $(IN).aux $(IN).log $(IN).out $(IN).synctex.gz $(IN).toc
CXX = pdflatex

all: $(IN).pdf clean

$(IN).pdf: $(IN).tex
	$(CXX) $<
	$(CXX) $<

clean:
	@echo Limpiando archivos extra generados por LaTex...
	rm -f $(FILES)
