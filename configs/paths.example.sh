# Copy to configs/paths.local.sh. Paths may be absolute or relative to repo root.
export PYTHON_BIN=python
export BASE_MODEL=CompVis/stable-diffusion-v1-4
export DATA_ROOT=data
export OUTPUT_ROOT=outputs
export STYLE_CLASSIFIER=checkpoints/checkpoint-2800
export NUDENET_ONNX_PATH=checkpoints/best.onnx
export NUDENET_ONNX_PROVIDERS=CPUExecutionProvider
# Optional offline ResNet; unset to download microsoft/resnet-50 automatically.
# export TINA_RESNET50_MODEL_PATH=/path/to/resnet-50
export GCD_PYTHON_BIN=/path/to/gcd-environment/bin/python
export GCD_ROOT=/path/to/celeb-detection-oss
export GCD_DATA_DIR=/path/to/celeb-detection-oss/examples/resources
# Each experiment accepts CKPT=/path/to/checkpoint and CHECKPOINT_TYPE=unet
# (or text_encoder for AdvUnlearn). Resources are documented in docs/resources.md.
