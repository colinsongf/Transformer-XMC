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
	gdrive-get 13bF53xYimNYsdHGc_Jzgs6Tr094O96wM ${dataset}.tar.bz2
elif [ ${dataset} == 'Wiki10-31K' ]; then
	gdrive-get 13EBKsc8JDe5KecMiruLBGw9osQXQZlvR ${dataset}.tar.bz2
else
	echo "unknown dataset [ Eurlex-4K | Wiki10-31K ]"
	exit
fi

tar -xjvf ${dataset}.tar.bz2
