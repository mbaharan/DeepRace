#!/bin/bash
python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 0 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 1 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 4 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 9 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh

python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 0 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 1 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 4 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 9 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh

python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 0 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 1 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 4 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 9 --dropout 0.1  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh

python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 0 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 1 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 4 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 9 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh

python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 0 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 1 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 4 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 9 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh

python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 0 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 1 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 4 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 9 --dropout 0.2  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh

python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 0 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 1 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 4 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 10 --ksize 7 --levels 3 --testset 9 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh

python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 0 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 1 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 4 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 15 --ksize 7 --levels 3 --testset 9 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh

python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 0 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 1 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 4 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
python3 train.py --nhid 20 --ksize 7 --levels 3 --testset 9 --dropout 0.3  --batch-size 1
sh /home/steve/Research_Projects/DeepRace/scripts/inference.sh
