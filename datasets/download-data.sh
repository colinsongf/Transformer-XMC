#!/bin/bash

dataset=$1

function gdrive-get() {
    fileid=$1
    filename=$2
    if [[ "${fileid}" == "" || "${filename}" == "" ]]; then
        echo "gdrive-curl gdrive-url|gdrive-fileid filename"
        return 1
    else
        if [[ ${fileid} = http* ]]; then
            fileid=$(echo ${fileid} | sed "s/http.*drive.google.com.*id=\([^&]*\).*/\1/")
        fi
        echo "Download ${filename} from google drive with id ${fileid}..."
        cookie="/tmp/cookies.txt"
        curl -c ${cookie} -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
        confirmid=$(awk '/download/ {print $NF}' ${cookie})
        curl -Lb ${cookie} "https://drive.google.com/uc?export=download&confirm=${confirmid}&id=${fileid}" -o ${filename}
        rm -rf ${cookie}
        return 0
    fi
}

if [ ${dataset} == 'Eurlex-4K' ]; then
	gdrive-get 1iPGbr5-z2LogtMFG1rwwekV_aTubvAb2 ${dataset}.tar.gz
elif [ ${dataset} == 'Wiki10-31K' ]; then
	gdrive-get 1Tv4MHQzDWTUC9hRFihRhG8_jt1h0VhnR ${dataset}.tar.gz
elif [ ${dataset} == 'AmazonCat-13K' ]; then
	gdrive-get 1VwHAbri6y6oh8lkpZ6sSY_b1FRNnCLFL ${dataset}.tar.gz
elif [ ${dataset} == 'Amazon-670K' ]; then
	gdrive-get 1Xd4BPFy1RPmE7MEXMu77E2_xWOhR1pHW ${dataset}.tar.gz
elif [ ${dataset} == 'Wiki-500K' ]; then
	gdrive-get 1bGEcCagh8zaDV0ZNGsgF0QtwjcAm0Afk ${dataset}.tar.gz
elif [ ${dataset} == 'Amazon-3M' ]; then
	gdrive-get 187vt5vAkGI2mS2WOMZ2Qv48YKSjNbQv4 ${dataset}.tar.gz
else
	echo "unknown dataset [Eurlex-4K | Wiki10-31K | AmazonCat-13K | Amazon-670K | Wiki-500K | Amazon-3M]"
	exit
fi

tar -zxvf ${dataset}.tar.gz

