# source /home/arne/miniconda3/etc/profile.d/conda.sh;
source /opt/anaconda3/etc/profile.d/conda.sh;
cd nn;
python3 documents_final.py;
cd ..;
cd semantic_loss;
conda activate semantic_loss;
python3 documents_final.py;
cd ..;
#cd logic_tensor_networks;
#conda activate logic_tensor_networks;
#python3 documents_final.py;
#cd ..;
# cd neurasp;
# conda activate neurasp;
# python3 documents_final.py;
# cd ..;
cd deepstochlog;
conda activate deepstochlog;
python3 documents_final.py;
# cd ..;
# cd deepproblog;
# conda activate deepproblog;
# python3 documents_final.py;