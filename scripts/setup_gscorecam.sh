#!/usr/bin/env bash
set -euo pipefail
if [[ ! -d gScoreCAM ]]; then
  git clone https://github.com/anguyen8/gScoreCAM.git
fi
python -m pip install -r gScoreCAM/colab_requirement.txt
python -m pip install validators
