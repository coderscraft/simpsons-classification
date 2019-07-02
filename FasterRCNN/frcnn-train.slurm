#!/bin/sh

## Specify the name for your job, this is the job name by which Slurm will
## refer to your job.  This can be different from the name of your executable
## or the name of your script file.
#SBATCH --job-name FRCNN-Simpsons-train

#SBATCH --qos normal  # normal,cdsqos,phyqos,csqos,statsqos,hhqos,gaqos,esqos
#SBATCH -p gpuq       # partition (queue): all-LoPri, all-HiPri,
                      #   bigmem-LoPri, bigmem-HiPri, gpuq, CS_q, CDS_q, ...

## Deal with output and errors.  Separate into 2 files (not the default).
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayMain, %a=arraySub
#SBATCH -o /scratch/%u/FasterRCNN/log/%x-%N-%j.out    # Output file
#SBATCH -e /scratch/%u/FasterRCNN/log/%x-%N-%j.err    # Error file
#SBATCH --mail-type=BEGIN,END,FAIL     # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=rrane@gmu.edu   # Put your GMU email address here

## Specifying an upper limit on needed resources will improve your scheduling
## priority, but if you exceed these values, your job will be terminated.
## Check your "Job Ended" emails for actual resource usage info.
##SBATCH --mem=5G          # Total memory needed for your job (suffixes: K,M,G,T)
##SBATCH --time=0-00:60    # Total time needed for your job: Days-Hours:Minutes

## These options are more useful when running parallel and array jobs
#SBATCH --nodes 1         # Number of nodes (computers) to reserve
#SBATCH --tasks 1         # Number of independent processes per job
#SBATCH --gres=gpu:1      # Reserve 1 GPU

## Load the relevant modules needed for the job
module load keras/2.2.0-py36

## Setup the environment
export PYTHONPATH=$HOME/simpsons/python
export PYTHONPATH=$HOME/simpsons/simpsons-classification:$PYTHONPATH

## Start the job
python3 train.py -p /scratch/rrane/FasterRCNN/data/