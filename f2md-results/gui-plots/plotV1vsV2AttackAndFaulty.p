set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto 			# set ytics automatically
set title "True Positives and False Positives"
set xlabel "Time (Min)"
set ylabel "Messages (Num)"
set grid 
show grid 
set key right bottom
funcT(x) = x / 60
#func3(x,y) = y<=0  ? 0 : 100*x / y
func3(x,y) = x

stats "AppV1.dat" using (lastV1=func3($6,$7)) nooutput
# stats "AppV2.dat" using (lastV2=func3($6,$7)) nooutput
stats "AppV1.dat" using (FNV1=($7-$6)) nooutput
# stats "AppV2.dat" using (FNV2=($7-$6)) nooutput
set label sprintf("TotalTP = %3.10g-%3.10gmsg",lastV1, FNV1) at graph 0.02,0.95 tc rgb ARG1

stats "AppV1.dat" using (lastV1=func3($4,$5)) nooutput
#stats "AppV2.dat" using (lastV2=func3($4,$5)) nooutput
stats "AppV1.dat" using (FNV1=($5-$4)) nooutput
#stats "AppV2.dat" using (FNV2=($5-$4)) nooutput
set label sprintf("TotalFP = %3.10g-%3.10gmsg",lastV1, FNV1) at graph 0.02,0.9 tc rgb ARG1

# "AppV2.dat" using (funcT($2)):(func3($4,$5)) title 'V2 False Positive' with linespoints linestyle 2 lw 2 pi 30 ps 0.75 lc rgb "#4dbeee"
plot "AppV1.dat" using (funcT($2)):(func3($6,$7)) title 'True Positive' with lines lw 2 lc rgb "#CD853F", \
"AppV1.dat" using (funcT($2)):(func3($4,$5)) title 'False Positive' with lines lw 2 lc rgb "magenta"