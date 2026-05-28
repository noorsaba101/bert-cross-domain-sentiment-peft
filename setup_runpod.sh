#!/usr/bin/env bash
set -e

apt update
apt install -y tmux

pip install -r requirements_runpod.txt

python -c "import torch, numpy, pandas, transformers, datasets, peft, sklearn; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('numpy', numpy.__version__); print('pandas', pandas.__version__); print('transformers', transformers.__version__); print('datasets', datasets.__version__); print('peft', peft.__version__)"