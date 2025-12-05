python plot.py ../../data/casestudies.tex
gnuplot make_cdfs.gnu
pdfcrop responsetime.pdf
mv responsetime-crop.pdf responsetime.pdf
