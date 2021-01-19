sudo service docker start
sudo docker inspect -f {{.State.Running}} crayon || sudo docker run -d -p 8888:8888 -p 8889:8889 --name crayon alband/crayon
sudo docker start crayon

export OMP_NUM_THREADS=1
source activate PokerAI
source deactivate
source activate PokerAI
