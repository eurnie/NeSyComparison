python3 -m constraints_to_cnf -i constraints.txt -o dimacs.txt;
pysdd -c dimacs.txt -W constraint.vtree -R constraint.sdd;
python3 -m constraints_to_cnf -i constraints_one_hot_encoding.txt -o dimacs_one_hot_encoding.txt;
pysdd -c dimacs_one_hot_encoding.txt -W constraint_one_hot_encoding.vtree -R constraint_one_hot_encoding.sdd;