#!/bin/bash
# this file works very similar to run_some_dt.sh, doing the same but varyinf f factor instead of a fixed dt
# f facors requested 
my_array=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)
STR="32_f/energy_32_ABCFL_f"
dum=1000
cd SA3
source env.sh
cd ..


for element in "${my_array[@]}"
do
    echo "run_some calling f = $element"
    cp auxiliar/first_microTGV.txt microTGV.sa3
    echo -e "\nfCFL $element\n" >> microTGV.sa3
    cat auxiliar/second_microTGV.txt >> microTGV.sa3
    mpirun -np 8 sa3 2 2 2 0 $PWD/microTGV $PWD/kk
    grep "TGV_Energy" kk/stdout00.txt > prep.csv
    num=$(echo $element*$dum | bc)
    num2=${num%.*}
    sed -e 's/\<TGV_Energy\>//g' prep.csv > "$STR$num2.csv"
    grep "cr_info name solveStep00" kk/stdout00.txt >> 32/f32_ABCFL_solveStep.txt
done

rm -rf kk
rm prep.csv
rm microTGV.sa3


