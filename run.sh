
# script="trt.py"
script="pytorch.py"
jit="--jit"

python $script --batch-size 256 -l 50 --csv $jit
python $script --batch-size 128 -l 100 --csv $jit
python $script --batch-size 64 -l 200 --csv $jit
python $script --batch-size 32 -l 400 --csv $jit
python $script --batch-size 16 -l 800 --csv $jit
python $script --batch-size 8 -l 1600 --csv $jit
python $script --batch-size 4 -l 3200 --csv $jit
python $script --batch-size 2 -l 6400 --csv $jit
python $script --batch-size 1 -l 12800 --csv $jit



