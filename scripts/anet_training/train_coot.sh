#!/usr/bin/zsh -eu
scripts/train.sh anet mart coot seperate 10 0.5 vonly
scripts/train.sh anet mart coot seperate 10 0.5 multimodal
scripts/train.sh anet mart coot joint 10 0.5 vonly
scripts/train.sh anet mart coot joint 10 0.5 multimodal
