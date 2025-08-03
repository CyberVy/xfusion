 pip install -q git+https://github.com/CyberVy/xfusion.git & transformers-cli env 2> /dev/null & wait

for arg in "$@"; do
  if [[ $arg == --env=* ]]; then
    pair="${arg#--env=}"
    key="${pair%%=*}"
    value="${pair#*=}"
    export "$key=$value"
  fi
done

python3 ./start.py
