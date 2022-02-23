
gpu='0,1,2,3'

######################
# loss weight params #
######################
lr=1e-5
momentum=0.99
lambda_d=1
lambda_g=0.1

################
# train params #
################
max_iter=100000
# crop=128
snapshot=5000
batch=16

weight_share='weights_shared'
discrim='discrim_score'

########
# Data #
########
src='stratix'
tgt='cyclone'
datadir='./datasets'


resdir="results/${src}_to_${tgt}/adda_sgd/${weight_share}_nolsgan_${discrim}"

# init with pre-trained cyclegta5 model
model='fcn8s'
# baseiter=115000
#model='fcn8s'
#baseiter=100000


# base_model="base_models/${model}-${src}-iter${baseiter}.pth"
outdir="${resdir}/${model}/lr${lr}_ld${lambda_d}_lg${lambda_g}_momentum${momentum}"

# Run python script #
CUDA_VISIBLE_DEVICES=${gpu} python train_fcn_adda.py \
    ${outdir} \
    --dataset ${src} --dataset ${tgt} --datadir ${datadir} \
    --lr ${lr} --momentum ${momentum} --gpu ${gpu} \
    --lambda_d ${lambda_d} --lambda_g ${lambda_g} \
    --model ${model} \
    --"${weight_share}" --${discrim} --no_lsgan \
    --max_iter ${max_iter}  --batch ${batch} \
    --snapshot $snapshot
    # --weights_init ${base_model} --crop_size ${crop} \