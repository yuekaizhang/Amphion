import os
import argparse
import logging
import librosa
import torch
import torchaudio
from models.tts.valle_v2.valle_inference import ValleInference
from models.tts.valle_v2.g2p_processor import G2pProcessor

LIBRITTS_R="/workspace/libritts-r/"
LIBRITTS="/workspace/libritts/"

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--ar-model-path",
        type=str,
        default="/workspace/Amphion/ckpt/VALLE_V2/ar_libritts/checkpoint/epoch-0039_step-0069000_loss-1.933620/model.safetensors",
        help="Path to the ar model file",
    )

    parser.add_argument(
        "--nar-model-path",
        type=str,
        default="/workspace/Amphion/ckpt/VALLE_V2/nar_libritts/checkpoint/epoch-0041_step-0073000_loss-1.445112/model.safetensors",
        help="Path to the nar model file",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model",
    )

    parser.add_argument(
        "--demo-manifest-path",
        type=str,
        default="/workspace/Amphion/libritts.txt",
        help="Path to the valle demo manifest file",
    )

    parser.add_argument(
        "--speechtokenizer-path",
        type=str,
        default=None,
        help="Path to the speechtokenizer model file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/Amphion/audio_samples",
        help="Path to the valle demo files",
    )

    return parser

def infer_audio(model, g2p, prompt_text, text, prompt_audio, device):
    wav, _ = librosa.load(prompt_audio, sr=16000) # TODO: check this??
    wav = torch.tensor(wav, dtype=torch.float32)
    prompt_transcript = g2p(prompt_text, 'en')[1]
    target_transcript = g2p(text, 'en')[1]
    prompt_transcript = torch.tensor(prompt_transcript).long()
    target_transcript = torch.tensor(target_transcript).long()
    transcript = torch.cat([prompt_transcript, target_transcript], dim=-1)

    batch = {
            'speech': wav.unsqueeze(0).to(device),
            'phone_ids': transcript.unsqueeze(0).to(device),
    }
    configs = [dict(
        top_p=0.9,
        top_k=5,
        temperature=0.95,
        repeat_penalty=1.0,
        max_length=2000,
        num_beams=1,
    )] # model inference hyperparameters
    output_wav = model(batch, configs)
    return output_wav

def main():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = get_parser()
    args = parser.parse_args()
    logging.info(vars(args))

    use_speechtokenizer = True if args.speechtokenizer_path else False
    model = ValleInference(
        ar_path=args.ar_model_path,
        nar_path=args.nar_model_path,
        use_speechtokenizer=use_speechtokenizer,
        speechtokenizer_path=args.speechtokenizer_path,
        device=args.device,
    )
    logging.info("model loaded")
    g2p = G2pProcessor()

    with open(args.demo_manifest_path, "r") as f:
        i = 0
        # break after 4 lines
        for line in f:
            fields = line.strip().split("\t")
            assert len(fields) == 4
            prompt_text, prompt_audio, text, _ = fields

            # LibriTTS_R/test-clean/908/157963/908_157963_000020_000000.wav + LI
            if "LibriTTS_R/" in prompt_audio:
                prompt_audio = LIBRITTS_R + prompt_audio
            elif "LibriTTS/" in prompt_audio:
                prompt_audio = LIBRITTS + prompt_audio
            else:
                raise ValueError(f"Unknown dataset: {prompt_audio}")

            # speaker id is 908
            spk_id = prompt_audio.split("/")[-3]
            dataset_name = prompt_audio.split("/")[-5]
            subdir = dataset_name + "_" + spk_id
            # audio_path is args.output_dir/dataset_name_spk_id/amphion_valle.wav
            audio_path = os.path.join(args.output_dir, subdir, "amphion_valle.wav")
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            logging.info(f"synthesize text: {text}")
            output_wav = infer_audio(model, g2p, prompt_text, text, prompt_audio, args.device)

            logging.info(f"save to {audio_path}, audio shape: {output_wav.shape}")
            # store
            torchaudio.save(audio_path, output_wav.squeeze(0).cpu(), 16000)
            i += 1
            if i >= 4:
                break

if __name__ == "__main__":
    main()