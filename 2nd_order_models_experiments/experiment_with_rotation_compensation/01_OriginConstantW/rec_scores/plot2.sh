for i in `seq 0 9`;
do
sp_plot_files_2c_fn.pl score_?????_$i.txt 1 2 out_$i
done    
montage -geometry +0+0 out_*.png all_plots.png
rm -f out_*.png
