import requests
import transformers

import mlflow
mlflow.set_tracking_uri("https://tracking-server.a2n.finance")


# Sets the current active experiment to the "Apple_Models" experiment and
# returns the Experiment metadata
apple_experiment = mlflow.set_experiment(experiment_id="146986237246779625")

# Acquire an audio file
resp = requests.get(
    "https://storage.googleapis.com/kagglesdsdata/datasets/829978/1417968/harvard.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240709%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240709T030155Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=6c171315e958f49e8f7cb268b9a5c65c8f2de5bd9b9c05f69e231e6f70811a2dbb47bc23f994f1a3af06b8853c49939779cb9be1e1dd7a089e4c3f2112ded20310d022ba72611ab71fccfa10532bd84e1963715efd10d1d31ce92cbfe4197a28a8931871f883a76c0b94fa0b365beb098332fc5d2f66402ade22b8eaa0a9894653b318932ed524124a42324535555613e53ddabf1e5f61abe666e0c264e01379100e12ced6cf98ab4b722c058a6ac3da3233245767e8eeb298f598e8f7d66f0d58e48166352d7416de5a742873849b5206beb979c1b77a3060044ae03684d86875f77903fbbd798a6f904402cbf3186f3a9478af896c771580373bedfacc6eb0"
)
resp.raise_for_status()
audio = resp.content

task = "automatic-speech-recognition"
architecture = "openai/whisper-tiny"

model = transformers.WhisperForConditionalGeneration.from_pretrained(architecture)
tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture)
feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture)
model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
audio_transcription_pipeline = transformers.pipeline(
    task=task, model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
)

# Note that if the input type is of raw binary audio, the generated signature will match the
# one created here. For other supported types (i.e., numpy array of float32 with the
# correct bitrate extraction), a signature is required to override the default of "binary" input
# type.
signature = mlflow.models.infer_signature(
    audio,
    mlflow.transformers.generate_signature_output(audio_transcription_pipeline, audio),
)

inference_config = {
    "return_timestamps": "word",
    "chunk_length_s": 20,
    "stride_length_s": [5, 3],
}

# Log the pipeline
with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=audio_transcription_pipeline,
        artifact_path="whisper_transcriber",
        extra_pip_requirements=["ffmpeg"],
        signature=signature,
        input_example=audio,
        inference_config=inference_config,
    )

# Load the pipeline in its native format
loaded_transcriber = mlflow.transformers.load_model(model_uri=model_info.model_uri)

transcription = loaded_transcriber(audio, **inference_config)

print(f"\nWhisper native output transcription:\n{transcription}")

# Load the pipeline as a pyfunc with the audio file being encoded as base64
pyfunc_transcriber = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

pyfunc_transcription = pyfunc_transcriber.predict([audio])

# Note: the pyfunc return type if `return_timestamps` is set is a JSON encoded string.
print(f"\nPyfunc output transcription:\n{pyfunc_transcription}")