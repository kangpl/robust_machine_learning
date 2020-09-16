#!/bin/bash

# Script to start a tensorboard from a local computer on Euler/Leonhard Server
# Peilin Kang, Sep. 2020 @ETH Zurich

# function to print usage instructions
function print_usage {
        echo -e "Usage:\t start_tensorboard.sh CLUSTER NETHZ_USERNAME RUN_TIME LOG_DIR\n"
        echo -e "Arguments:\n"
        echo -e "CLUSTER\t\t\t Name of the cluster on which the jupyter notebook should be started (Euler or LeoOpen)"
        echo -e "NETHZ_USERNAME\t\tNETHZ username for which the notebook should be started"
        echo -e "RUN_TIME\t\tRun time limit for the jupyter notebook on the cluster (HH:MM)"
        echo -e "LOG_DIR\t\tThe directory where tensorboard log locates\n"
        echo -e "Example:\n"
        echo -e "./start_tensorboard.sh LeoOpen pekang 04:00 robust_machine_learning/jupyter_notebook/runs\n"
}

# if number of command line arguments is different from 4 or if $1==-h or $1==--help
if [ "$#" !=  4 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    print_usage
    exit
fi

# find out in which directory the script is located
SCRIPTDIR=$(dirname $(realpath -s $0))

# Parse and check command line arguments (cluster, NETHZ username, run time limit, log directory)

# check on which cluster the script should run and load the proper python module
CLUSTERNAME="$1"

if [ "$CLUSTERNAME" == "Euler" ]; then
    CHOSTNAME="euler.ethz.ch"
    PCOMMAND="new gcc/4.8.2 r/3.6.0 python/3.6.1 eth_proxy"
elif [ "$CLUSTERNAME" == "LeoOpen" ]; then
    CHOSTNAME="login.leonhard.ethz.ch"
    PCOMMAND="gcc/6.3.0 python_cpu/3.7.4 eth_proxy"
else
    echo -e "Incorrect cluster name. Please specify Euler or LeoOpen as cluster and and try again.\n"
    print_usage
    exit
fi

echo -e "\nCluster: $CLUSTERNAME"

# no need to do checks on the username. If it is wrong, the SSH commands will not work
USERNAME="$2"
echo -e "NETHZ username: $USERNAME"

# run time limit
RUN_TIME="$3"

# check if RUN_TIME is provided in HH:MM format
if ! [[ "$RUN_TIME" =~ ^[0-9][0-9]:[0-9][0-9]$ ]]; then
    echo -e "Incorrect format. Please specify runtime limit in the format HH:MM and try again\n"
    print_usage
    exit
else
    echo -e "Run time limit set to $RUN_TIME"
fi

LOG_DIR="$4"

# check if some old files are left from a previous session and delete them
echo -e "Checking for left over files from previous sessions"
if [ -f $SCRIPTDIR/reconnect_info_tb ]; then
        echo -e "Found old reconnect_info_tb file, deleting it ..."
        rm $SCRIPTDIR/reconnect_info_tb
fi
ssh -T $USERNAME@$CHOSTNAME <<ENDSSH
if [ -f /cluster/home/$USERNAME/tbinfo ]; then
        echo -e "Found old tbinfo file, deleting it ..."
        rm /cluster/home/$USERNAME/tbinfo
fi
if [ -f /cluster/home/$USERNAME/tbip ]; then
	echo -e "Found old tbip file, deleting it ..."
        rm /cluster/home/$USERNAME/tbip
fi 
ENDSSH

# run the tensorboard job on Euler/Leonhard Open and save ip and port
# in the files tbip and tbinfo in the home directory of the user on Euler/Leonhard Open
echo -e "Connecting to $CLUSTERNAME to start tensorboard in a batch job"
ssh $USERNAME@$CHOSTNAME bsub -W $RUN_TIME  <<ENDBSUB
module load $PCOMMAND
export XDG_RUNTIME_DIR=
IP_REMOTE="\$(hostname -i)"
echo "Remote IP:\$IP_REMOTE" >> /cluster/home/$USERNAME/tbip
tensorboard --logdir=/cluster/home/$USERNAME/$LOG_DIR --host "\$IP_REMOTE" &> /cluster/home/$USERNAME/tbinfo 
ENDBSUB

# wait until tensorboard has started, poll every 60 seconds to check if $HOME/tbinfo exists
# once the file exists and is not empty, the tensorboard has been startet and is listening
ssh $USERNAME@$CHOSTNAME "while ! [ -e /cluster/home/$USERNAME/tbinfo -a -s /cluster/home/$USERNAME/tbinfo ]; do echo 'Waiting for tensorboard to start, sleep for 60 sec'; sleep 60; done"

# get remote ip, port from files stored on Euler/Leonhard Open
echo -e "Receiving ip, port from tensorboard"
remoteip=$(ssh $USERNAME@$CHOSTNAME "cat /cluster/home/$USERNAME/tbip | grep -m1 'Remote IP' | cut -d ':' -f 2")
remoteport=$(ssh $USERNAME@$CHOSTNAME "cat /cluster/home/$USERNAME/tbinfo | grep -m1 'http' | cut -d '/' -f 3 | cut -d ':' -f 2")

if  [[ "$remoteip" == "" ]]; then
    echo -e "Error: remote ip is not defined. Terminating script."
    echo -e "Please login to the cluster and check with bjobs if the batch job is still running."
    exit 1
fi

if  [[ "$remoteport" == "" ]]; then
    echo -e "Error: remote port is not defined. Terminating script."
    echo -e "Please login to the cluster and check with bjobs if the batch job is still running."
    exit 1
fi


echo -e "Remote IP address: $remoteip"
echo -e "Remote port: $remoteport"

# get a free port on local computer
echo -e "Determining free port on local computer"
PORTN=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo -e "Local port: $PORTN"

# write reconnect_info file
echo -e "Restart file \n" >> $SCRIPTDIR/reconnect_info_tb
echo -e "Remote IP address: $remoteip\n" >> $SCRIPTDIR/reconnect_info_tb
echo -e "Remote port: $remoteport\n" >> $SCRIPTDIR/reconnect_info_tb
echo -e "Local port: $PORTN\n" >> $SCRIPTDIR/reconnect_info_tb
echo -e "SSH tunnel: ssh $USERNAME@$CHOSTNAME -L $PORTN:$remoteip:$remoteport -N &\n" >> $SCRIPTDIR/reconnect_info_tb
echo -e "URL: http://localhost:$PORTN\n" >> $SCRIPTDIR/reconnect_info_tb

# setup SSH tunnel from local computer to compute node via login node
echo -e "Setting up SSH tunnel for connecting the browser to the tensorboard"
ssh $USERNAME@$CHOSTNAME -L $PORTN:$remoteip:$remoteport -N &

# SSH tunnel is started in the background, pause 5 seconds to make sure
# it is established before starting the browser
sleep 5

# save url in variable
nburl=http://localhost:$PORTN
echo -e "Starting browser and connecting it to tensorboard"
echo -e "Connecting to url "$nburl

if [[ "$OSTYPE" == "linux-gnu" ]]; then
	xdg-open $nburl
elif [[ "$OSTYPE" == "darwin"* ]]; then
	open $nburl
else
	echo -e "Your operating system does not allow to start the browser automatically."
        echo -e "Please open $nburl in your browser."
fi
