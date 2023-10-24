set term x11 dashed enhanced background '#000000' 
set border lw 2 lc rgb '#00bb00'
set key tc rgb '#00bb00'
set xtic tc rgb '#00bb00'                        # set xtics automatically
set ytics tc rgb '#00bb00'                       
set y2tics tc rgb '#00bb00'		
set title tc rgb '#00bb00' 
set xlabel tc rgb '#00bb00'
set ylabel tc rgb '#00bb00'
set y2label tc rgb '#00bb00'
set grid lc rgb '#00bb00'

### Start multiplot (3x3 layout)

set multiplot layout 4,3 rowsfirst
#set lmargin 2.5
#set rmargin 2.5
#set tmargin 2.5
#set bmargin 2.5

# --- GRAPH a
call 'gui-plots/plotMdmAll.p' "#00bb00"
# --- GRAPH b

# --- GRAPH c
#call 'gui-plots/plotMda-DetectionLatency.p'
#call 'gui-plots/plotV1vsV2Faulty.p' "#00bb00"
# --- GRAPH d
call 'gui-plots/plotV1vsV2AttackAndFaulty.p' "#00bb00"
# --- GRAPH e

call 'gui-plots/plotDensity.p' "#00bb00"

call 'gui-plots/plotV1vsV2Recall.p' "#00bb00"
# --- GRAPH f
call 'gui-plots/plotV1vsV2Precision.p' "#00bb00"
# --- GRAPH g
call 'gui-plots/plotV1vsV2Accuracy.p' "#00bb00"
# --- GRAPH h
call 'gui-plots/plotV1vsV2F1Score.p' "#00bb00"
# --- GRAPH i
call 'gui-plots/plotV1vsV2Informedness.p' "#00bb00"
# --- GRAPH j
call 'gui-plots/plotV1vsV2Markedness.p' "#00bb00"
# --- GRAPH k
call 'gui-plots/plotV1vsV2MCC.p' "#00bb00"
# --- GRAPH j
call 'gui-plots/plotV1vsV2Kappa.p' "#00bb00"


# --- GRAPH k
#call 'gui-plots/plotMda-DetectionRate.p' "#00bb00"


unset multiplot
unset output

set term pdfcairo dashed enhanced background '#FFFFFF'
set output "stats.pdf" 
set border lw 2 lc rgb '#000000'
set key tc rgb '#000000'
set xtic tc rgb '#000000'                        # set xtics automatically
set ytics tc rgb '#000000'                       
set y2tics tc rgb '#000000'		
set title tc rgb '#000000' 
set xlabel tc rgb '#000000'
set ylabel tc rgb '#000000'
set y2label tc rgb '#00000'
set grid lc rgb '#000000'

## Start multiplot (3x3 layout)

#set lmargin 2.5
#set rmargin 2.5
#set tmargin 2.5
#set bmargin 2.5

# --- GRAPH a
call 'gui-plots/plotMdmAll.p' "#000000"
# --- GRAPH b

# --- GRAPH c
#call 'gui-plots/plotMda-DetectionLatency.p' "#000000"
# call 'gui-plots/plotV1vsV2Faulty.p' "#000000"
# --- GRAPH d
call 'gui-plots/plotV1vsV2AttackAndFaulty.p' "#000000"
# --- GRAPH e

call 'gui-plots/plotDensity.p' "#000000"

call 'gui-plots/plotV1vsV2Recall.p' "#000000"
# --- GRAPH f
call 'gui-plots/plotV1vsV2Precision.p' "#000000"
# --- GRAPH g
call 'gui-plots/plotV1vsV2Accuracy.p' "#000000"
# --- GRAPH h
call 'gui-plots/plotV1vsV2F1Score.p' "#000000"
# --- GRAPH i
call 'gui-plots/plotV1vsV2Informedness.p' "#000000"
# --- GRAPH j
call 'gui-plots/plotV1vsV2Markedness.p' "#000000"
# --- GRAPH k
call 'gui-plots/plotV1vsV2MCC.p' "#000000"
# --- GRAPH j
call 'gui-plots/plotV1vsV2Kappa.p' "#000000"


# --- GRAPH k
#call 'gui-plots/plotMda-DetectionRate.p' "#000000"

unset output
### End multiplot  


