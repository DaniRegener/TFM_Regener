#!/bin/bash
# please remember to change "NS_test_schemes_f.py" before launching
# this file. There you should set the scheme and the range of f in study
# at the end, all the setup is there, this only does some greps

STR="library_f/DUM"
SUFF0="_results"
SUFF1="_avgdt"
SUFF2="_f"
SUFF3="_ites"
SUFF4="_MSE"

python3 NS_test_schemes_f.py > $STR$SUFF0.txt
grep "Average dt: "            $STR$SUFF0.txt > $STR$SUFF1.txt
grep "f: "          	       $STR$SUFF0.txt > $STR$SUFF2.txt
grep "Number of iterations:"   $STR$SUFF0.txt > $STR$SUFF3.txt
grep "MSE:"  		       $STR$SUFF0.txt > $STR$SUFF4.txt
