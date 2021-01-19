sudo yum update -y
sudo yum install git gcc g++ polkit -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo docker pull alband/crayon
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/ec2-user/miniconda
export PATH=/home/ec2-user/miniconda/bin:$PATH
conda create -n PokerAI python=3.6 -y
source activate PokerAI
pip install requests
conda install pytorch=0.4.1 -c pytorch -y
pip install PokerRL[distributed]

git clone https://github.com/mohamedun/Deep-CFR.git
