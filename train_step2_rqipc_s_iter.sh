##!/usr/bin/env bash
#!/userhome/anaconda/bin/env bash
#PYTHON=${PYTHON:-"python"}
#PYTHON=${PYTHON:-"/userhome/anaconda/envs/pytorch/bin/python"}


#GPUS=$2
python cl_step2_rqipc_s_iter.py  > "/userhome/chengyl/UDA/multi-source/MFSAN/cl_domainnet/logs/$(date +"%Y_%m_%d_%H_%M")_cl_step2__101_rqipc.log" 2>&1