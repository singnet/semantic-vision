
tag=99999

M=score_$tag
P=""
for i in `seq 0 9`;
do
P="$P ${M}_${i}.txt $i"
done    
sp_plot_files_2c_nice.pl angle score $P 1 2 all_scores_$tag
