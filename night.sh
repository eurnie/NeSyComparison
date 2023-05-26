source /opt/anaconda3/etc/profile.d/conda.sh;
cd CiteSeer;
cd deepstochlog;
conda activate deepstochlog_citeseer;
python3 documents_final.py;
cd ..;
cd logic_tensor_networks;
conda activate logic_tensor_networks;
python3 documents_final.py;
cd ..;
cd ..;
cd MNIST_addition;
cd deepproblog;
conda activate deepproblog;
python3 addition_final_app.py;
python3 addition_final.py