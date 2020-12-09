set -e
tmp_dir=$(mktemp -d -t inference-XXXXXXXXXX)
cp $1 $tmp_dir  # Copy audio

docker run \
	-it \
	--memory=8g \
	--memory-swap=2g \
	--shm-size=8g \
	--cpuset-cpus=0-11 \
	--gpus '"device=0"' \
	--volume $(pwd)/config:/home/user/config \
	--volume $(pwd)/outputs:/home/user/outputs \
	--volume $(pwd)/inferenced:/home/user/inferenced \
	--volume $tmp_dir:/home/user/inference_files \
	wavenet-tamerlan-tabolov \
	bash -c "
    python ./src/audio_to_spec.py \
    audio_to_spec.audio_path=inference_files/$(basename $1) \
	" || rm -rf $tmp_dir
