# make csv for voxceleb1&2 dev audio (train_dir)
python3 scripts/build_datalist.py \
        --extension wav \
        --dataset_dir /ocean/projects/cis220031p/sdixit1/espnet/egs2/voxceleb_full/voxceleb1/dev/wav/ \
        --data_list_path data/train.csv