set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto 			# set ytics automatically
set yrange [0:1] 
# set yrange [-1:1]
#unset yrange 
set title "Markedness"
set xlabel "Time (Min)"
set ylabel "Markedness (Abs)"
set grid 
show grid 
set key right bottom
funcAcc(x,y) = 100* x/(x+y)
funcT(x) = x/60

#TP 26
#FP 14
#FN 27-26
#TN 15-14

#sensitivity TPR=TP/(TP+FN)
TPR(x,y) = x/(x+y)

#precision PPV=TP/(TP+FP)
PPV(x,y) = x/(x+y)

#F1 score F1=2 TP/(2 TP+ FP+ FN)
F(x,y,z) = 2*x/(2*x+y+z)

#F1 score 
FB(x,y) = (2*(x*y))/(x+y)

# Info BM=TP/TP+FN + TN/TN+FP − 1
# Mark MK=TP/TP+FP + TN/TN+FN − 1
MK(a,b,c,d) = (a)/(a+d)+c/(c+b) - 1

stats "AppV1.dat" using (lastV1=MK($6,($7-$6),($5-$4),$4)) nooutput
# stats "AppV2.dat" using (lastV2=MK($6,($7-$6),($5-$4),$4)) nooutput
set label sprintf("Last = %3.5g", lastV1) at graph 0.02,0.05 tc rgb ARG1

# "AppV2.dat" using (funcT($2)):(MK($6,($7-$6),($5-$4),$4)) title "V2 Markedness" with linespoints linestyle 2 lw 2 pi 30 ps 0.75 lc rgb "blue"
plot "AppV1.dat" using (funcT($2)):(MK($6,($7-$6),($5-$4),$4)) title "Markedness" with lines lw 2 lc rgb "red"


