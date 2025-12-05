# "Improved" Makefile for LaTeX documents
# INTPUFILE This defines the name of the file you want to latex
# the make file will automatically look for INPUT.tex as the latex file
INPUTFILE=main

default:
	latexmk -pdf ${INPUTFILE}
	@echo '****************************************************************'
	@echo '******** Did you spell-check the paper? ********'
	@echo '****************************************************************'

clean:
	latexmk -c ${INPUTFILE}
	rm -f ${INPUTFILE}.bbl

allclean:
	latexmk -C ${INPUTFILE}
	rm -f ${INPUTFILE}.bbl

monitor:
	latexmk -pdf -pvc ${INPUTFILE}

cleaner: allclean

view: ${INPUTFILE}.pdf
	open ${INPUTFILE}.pdf

${INPUTFILE}.pdf: default

count:
	texcount -brief *.tex
