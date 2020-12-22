conda activate py36

export PICKLE_DIR='pickle'
export OUTPUT_DIR='output'
export ALGORITHM='GCN'
# export ALGORITHM='GCN_FIN'
# export ALGORITHM='GAT'
# export ALGORITHM='GAT_FIN'
# export ALGORITHM='GCH'
# export ALGORITHM='GCH_FIN'

pickle_list=`ls $PICKLE_DIR`
for DIR_NAME in $pickle_list
do
    INPUT_DIR=$PICKLE_DIR/$DIR_NAME
    python run.py \
        --batch_size=64 \
        --epoch=100 \
        --dim_m=128 \
        --dim_p=100 \
        --algorithm=$ALGORITHM \
        --input_dir=$INPUT_DIR \
        --output_dir=$OUTPUT_DIR
done