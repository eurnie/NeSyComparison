for sum in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18"
do
    python3 -m constraints_to_cnf -i constraints/sum_$sum/constraints.txt -o constraints/sum_$sum/dimacs.txt;
    pysdd -c constraints/sum_$sum/dimacs.txt -W constraints/sum_$sum/constraint.vtree -R constraints/sum_$sum/constraint.sdd;
done