#!/usr/bin/perl -w
# parameters: file1 file2 .... fileN first_colum_num 
#             second_colunm_num1 second_column_num2 .. outputname

$info = "parameters MUST be: [xrange] [yrange] xlabel ylabel file1 file2 ... fileN first_colum_num \n" .
  "              second_colunm_num outputname\n";

die $info if ($#ARGV < 5 );

$x_range = "";
$y_range = "";
$x_range = shift @ARGV if ($ARGV[0] =~ /^\[/);
$y_range = shift @ARGV if ($ARGV[0] =~ /^\[/);
$xlabel  = shift @ARGV;
$ylabel  = shift @ARGV;

while (-f "$ARGV[0]")
{
    push(@fns,    shift @ARGV);
    push(@titles, shift @ARGV);
}

die $info  if ($#ARGV != 2);

$fst_col = shift @ARGV;
$snd_col = shift @ARGV;
$out     = shift @ARGV;

$run = "";
for ($i = 0 ; $i <= $#fns ; $i++)
{
    $fn = $fns[$i];
    $ti = $titles[$i];
    $run = $run . "\"$fn\" using (\$$fst_col):(\$$snd_col) title \"$ti\" with lines lw 2, ";
}

$run = "set terminal png ; set xlabel \"$xlabel\" ; set ylabel \"$ylabel\"; set output \"${out}.png\"; plot $x_range $y_range $run";

system "echo \'$run\' | gnuplot";
