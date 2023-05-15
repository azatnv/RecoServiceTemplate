#!/bin/bash

download() {
    folder=$1
    file_name=$2
    gdrive_file_id=$3

    file_path="$folder/$file_name"

    mkdir -p "$folder"
    if [ ! -f "$file_path" ]
    then
        gdown "$gdrive_file_id" -O "$file_path"
    fi
    echo "Downloaded" "$file_path"
}

download "models/" "light_fm.dill" "1NjTwM9hMveiV8twsfmcElswkj6iAZnsx"

download "models/lightfm/" "user_embeddings.dill" "107WdHB8Ka6Hqupw4g3iALLEG2w3N52y9"

download "models/popular_in_category/" "popular_in_category_model.dill" "1-0jA4-8wYMlzD1G4CxFQvjDVuwhRPtfm"

download "models/dssm" "index.hnsw" "1-rPxufL3hepnqZADsVO1DqLwft1L_Yb9"
download "models/dssm" "dssm_user_vectors.pickle" "1IK9DileuZhbZVdQHf3aZRWSu2fIFN9eI"
download "models/dssm" "user_id_to_uid.json" "1-PDhIqsvx9XsCw4l0Oio4RjMye6E8o_a"
download "models/dssm" "iid_to_item_id.json" "1-TrGCS_YmRWQkIeuhSXKsN7Xg3Nk_pEn"
download "models/dssm" "uid_to_watched_iids.json" "1-QtArop7useHil5pIeAM2t-d9J1nKOhS"
