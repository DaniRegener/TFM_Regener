#!/bin/bash
LC_NUMERIC="en_US.UTF-8"
my_array=($(jot 18 0.01 0.18)) # dt requested 
STR="32_t/energy_32_AB_dt"     # preffix of files to save "(mesh)_t/energy_(mesh)_(sch)_dt
dum=1000		       # factor to multiply de dt in order to have it in the name of the file
# please add also the second directory file in line 21
# Remember also that this file runs a merging of first_microTGV_dt and second_microTGV_dt. If you want
# to change the set up of the case, you need to change those files

for element in "${my_array[@]}"
do
    echo "run_some calling dt = $element"
    cp auxiliar/first_microTGV_dt.txt microTGV.sa3
    echo -e "\nfdt $element\n" >> microTGV.sa3
    cat auxiliar/second_microTGV_dt.txt >> microTGV.sa3
    mpirun -np 8 sa3 2 2 2 0 $PWD/microTGV $PWD/kk
    grep "TGV_Energy" kk/stdout00.txt > prep.csv
    num=$(echo $element*$dum | bc)S
    num2=${num%.*}
    sed -e 's/\<TGV_Energy\>//g' prep.csv > "$STR$num2.csv"
    # create and save the solveP file, similar format
    grep "cr_info name ins_solveP00" kk/stdout00.txt >> 32_t/32_AB_dt_solveP.txt 
done
rm -rf kk
rm prep.csv
rm microTGV.sa3


