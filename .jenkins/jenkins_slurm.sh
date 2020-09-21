#!/bin/bash

# some global variables
maxsleep=9000

# check presence of env directory
envloc=`pwd`"/.jenkins"

# Download the env
module load git
git submodule update

# setup module environment and default queue
if [ ! -f ${envloc}/buildenv/machineEnvironment.sh ] ; then
    echo "Error 1201 test.sh ${LINENO}: Could not find ${envloc}/buildenv/machineEnvironment.sh"
    exit 1
fi
. ${envloc}/buildenv/machineEnvironment.sh

# load machine dependent functions
if [ ! -f ${envloc}/buildenv/env.${host}.sh ] ; then
    exitError 1202 ${LINENO} "could not find ${envloc}/buildenv/env.${host}.sh"
fi
. ${envloc}/buildenv/env.${host}.sh

# load slurm tools
if [ ! -f ${envloc}/buildenv/slurmTools.sh ] ; then
    exitError 1203 ${LINENO} "could not find ${envloc}/buildenv/slurmTools.sh"
fi
. ${envloc}/buildenv/slurmTools.sh

# set up virtual env, if not already set up
python3 -m venv venv
. ./venv/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -r requirements.txt

# check if SLURM script exists
script="${envloc}/buildenv/submit.${host}.slurm"
test -f ${script} || exitError 1252 ${LINENO} "cannot find script ${script}"

# some global variables
action="$1"
optarg="$2"

script="${root}/actions/${action}.sh"
test -f "${script}" || exitError 1301 ${LINENO} "cannot find script ${script}"
# define command
cmd="${script} ${optarg}"

# setup SLURM job
out=job.out
/bin/sed -i 's|<NAME>|jenkins|g' ${script}
/bin/sed -i 's|<NTASKS>|1|g' ${script}
/bin/sed -i 's|<NTASKSPERNODE>|'"${nthreads}"'|g' ${script}
/bin/sed -i 's|<CPUSPERTASK>|1|g' ${script}
/bin/sed -i 's|<OUTFILE>|'"${out}"'|g' ${script}
/bin/sed -i 's|<CMD>|'"export MV2_USE_CUDA=0\n${cmd}"'|g' ${script}
/bin/sed -i 's|<PARTITION>|'"cscsci"'|g' ${script}

# submit SLURM job
launch_job ${script} ${maxsleep}
if [ $? -ne 0 ] ; then
  exitError 1251 ${LINENO} "problem launching SLURM job ${script}"
fi

# echo output of SLURM job
cat job.out