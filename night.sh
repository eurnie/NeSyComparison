source /opt/anaconda3/etc/profile.d/conda.sh;
cd MNIST_addition;
cd logic_tensor_networks;
conda activate logic_tensor_networks;
python3 addition_final.py;
cd ..;
cd semantic_loss;
conda activate semantic_loss;
python3 addition_final.py;