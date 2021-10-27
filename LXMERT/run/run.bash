# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name

if [ -d $output ] 
then
    echo "Directory $output exists."
    # exit 2
else
    mkdir -p $output/src
fi

cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python -u src/tasks/main.py \
    --train train --valid minival  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --batchSize 32 --optim bert --lr 1e-5 --epochs 10 \
    --tqdm --output $output ${@:3}