test
source /opt/anaconda3/etc/profile.d/conda.sh;
cd MNIST_addition;
cd logic_tensor_networks;
conda activate logic_tensor_networks;
python3 addition_final.py;
cd ..;
cd ..;
cd Citeseer;
cd neurasp;
conda activate neurasp;
python3 documents_final.py;
cd ..;
cd logic_tensor_networks;
conda activate logic_tensor_networks;
python3 documents_param.py;
