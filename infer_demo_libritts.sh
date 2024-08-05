export PYTHONPATH="./"

work_dir="./" # Amphion root folder
echo work_dir: $work_dir

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

speechtokenizer_path='./egs/tts/VALLE_V2/ckpts/speechtokenizer_hubert_avg'
ar_model_path='./egs/tts/VALLE_V2/ckpts/valle_ar_mls_196000.bin'  # huggingface-cli download amphion/valle valle_ar_mls_196000.bin valle_nar_mls_164000.bin --local-dir ckpts
nar_model_path='./egs/tts/VALLE_V2/ckpts/valle_nar_mls_164000.bin'
manifest_path='/mnt/samsung-t7/yuekai/asr/lifeiteng.github.com/valle/libritts.txt'

# python3 infer.py --ar-model-path $ar_model_path --nar-model-path $nar_model_path --device cpu --demo-manifest-path $manifest_path --speechtokenizer-path $speechtokenizer_path

python3 infer.py