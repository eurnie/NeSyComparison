python3 -m constraints_to_cnf -i constraints_generated.txt -o dimacs.txt
pysdd -c dimacs.txt -W constraint.vtree -R constraint.sdd