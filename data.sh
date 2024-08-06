rm -rf data; mkdir data
wget -P data/ https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
python3 scripts/format_trials.py \
            --voxceleb1_root /ocean/projects/cis220031p/sdixit1/espnet/egs2/voxceleb_full/voxceleb1/test/ \
            --src_trials_path data/veri_test2.txt \
            --dst_trials_path data/vox1_test.txt