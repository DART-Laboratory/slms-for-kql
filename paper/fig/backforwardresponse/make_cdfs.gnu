set terminal pdf color enhanced font  "Helvetica,12"
set size 0.6,0.6
set bmargin 6
set style line 1 linetype 4 linewidth 3 linecolor rgb "#5ab4ac"
set style line 2 linetype 3 linewidth 3 linecolor rgb "#d8b365"
set style line 3 linetype 2 linewidth 3 linecolor rgb "#FFAA00"
set style line 4 linetype 1 linewidth 3 linecolor rgb "#7D09B2"
set style line 5 linetype 5 linewidth 3 linecolor rgb "#000000"

#####################################################


set ylabel "CDF" offset 1.5,0,0
set ytics 0.2
set xlabel "Response Time (secs)"

set output "responsetime.pdf"


set key at graph 0.95, 0.3


#set xrange [0:0.5]

plot "backvalues.txt" using 1:2 smooth freq with line ls 1  title "Prov. Tracker (Backward)", \
     "forwardvalues.txt" using 1:2 smooth freq with line ls 2 title "Prov. Tracker (Forward)"
