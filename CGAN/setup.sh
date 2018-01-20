apt update
apt install python3-pip
echo "[install]" >> ~/.pip/pip.conf
echo "trusted-host=pypi.douban.com" >> ~/.pip/pip.conf
pip3 install --upgrade pip
pip3 install tensorflow-gpu

cd ../..
git clone https://github.com/zsdonghao/tensorlayer.git
cd tensorlayer
pip3 install -e .

pip3 install opencv-python
apt update && apt install -y libsm6
apt update && apt install -y libxrender1
