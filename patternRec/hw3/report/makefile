.PHONY: pdf all clean

all: pdf

pdf: report.tex
	latexmk -pdf -pdflatex="pdflatex -interactionmode=nonstopmode" -use-make report.tex

clean:
	latexmk -CA

