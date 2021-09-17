echo Single thread
OMP_NUM_THREADS=1 python3 matrixMultiply.py --size 500
echo Two threads
OMP_NUM_THREADS=2 python3 matrixMultiply.py --size 500
echo Four threads
OMP_NUM_THREADS=4 python3 matrixMultiply.py --size 500
echo Eight threads
OMP_NUM_THREADS=8 python3 matrixMultiply.py --size 500
