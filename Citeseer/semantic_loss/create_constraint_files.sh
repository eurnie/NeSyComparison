for dataset in "CiteSeer" "Cora" "PubMed"
do
    python3 -m constraints_to_cnf -i constraints/$dataset/constraints.txt -o constraints/$dataset/dimacs.txt;
    pysdd -c constraints/$dataset/dimacs.txt -W constraints/$dataset/constraint.vtree -R constraints/$dataset/constraint.sdd;
done