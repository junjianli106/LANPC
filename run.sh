##BRCA
for((FOLD=0;FOLD<4;FOLD++));
do
    python train.py --stage='train' --config='BRCA/surformer.yaml' --gpus=2 --fold=$FOLD
    python train.py --stage='test' --config='BRCA/surformer.yaml'  --gpus=2 --fold=$FOLD
done
