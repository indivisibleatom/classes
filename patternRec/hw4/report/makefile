.PHONY: pdf all clean

all: pdf

pdf: report.tex
	rm report.pdf
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make report.tex

clean:
	latexmk -CA

