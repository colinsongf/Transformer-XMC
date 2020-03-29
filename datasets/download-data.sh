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
	gdrive-get 1JdkkIM4hpQyOiiHX7Rp1ymu2L10r1xYg ${dataset}.tar.gz
elif [ ${dataset} == 'Wiki10-31K' ]; then
	gdrive-get 1y2FnzPE_3zLsQkKbtA4uGTgFGFwkjtN2 ${dataset}.tar.gz
else
	echo "unknown dataset [ Eurlex-4K | Wiki10-31K ]"
	exit
fi

tar -zxvf ${dataset}.tar.gz

