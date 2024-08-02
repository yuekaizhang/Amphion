import os
os.chdir('../../..')
print(os.getcwd()) # Ensure this is you Amphion root path, otherwise change the above path to you amphion root path
assert os.path.isfile('./README.md') # make sure the current path is Amphion root path
import sys
sys.path.append('.')


ar_model_path = './egs/tts/VALLE_V2/ckpts/valle_ar_mls_196000.bin'  # huggingface-cli download amphion/valle valle_ar_mls_196000.bin valle_nar_mls_164000.bin --local-dir ckpts
nar_model_path = './egs/tts/VALLE_V2/ckpts/valle_nar_mls_164000.bin'
speechtokenizer_path = './egs/tts/VALLE_V2/ckpts/speechtokenizer_hubert_avg' 


device='cpu'
from models.tts.valle_v2.valle_inference import ValleInference
# change to device='cuda' to use CUDA GPU for fast inference
# change "use_vocos" to True would give better sound quality
# If you meet problem with network, you could set "use_vocos=False", though would give bad quality
model = ValleInference(ar_path=ar_model_path, nar_path=nar_model_path, speechtokenizer_path=speechtokenizer_path, device=device)


# prepare inference data
import librosa
import torch
wav, _ = librosa.load('./egs/tts/VALLE_V2/example.wav', sr=16000)
wav = torch.tensor(wav, dtype=torch.float32)

# The transcript of the prompt part
prompt_transcript_text = 'and keeping eternity before the eyes'

# Here are the words you want the model to output
target_transcript_text = 'It is a good GPU for deep learning'
from models.tts.valle_v2.g2p_processor import G2pProcessor
g2p = G2pProcessor()
prompt_transcript = g2p(prompt_transcript_text, 'en')[1]
target_transcript = g2p(target_transcript_text, 'en')[1]

print(g2p(target_transcript_text, 'en'))


prompt_transcript = torch.tensor(prompt_transcript).long()
target_transcript = torch.tensor(target_transcript).long()
transcript = torch.cat([prompt_transcript, target_transcript], dim=-1)
batch = {
        'speech': wav.unsqueeze(0),
        'phone_ids': transcript.unsqueeze(0),
}


print(batch)


configs = [dict(
    top_p=0.9,
    top_k=5,
    temperature=0.95,
    repeat_penalty=1.0,
    max_length=2000,
    num_beams=1,
)] # model inference hyperparameters
output_wav = model(batch, configs)
print(output_wav.shape, output_wav)

print(f'prompt_transcript : {prompt_transcript_text}')
print(f'target_transcript : {target_transcript_text}')


import torchaudio
torchaudio.save('./egs/tts/VALLE_V2/out.wav', output_wav.squeeze(0), 16000)