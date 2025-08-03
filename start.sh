if python3 -m pip show xfusion > /dev/null 2>&1; then
  echo "xfusion is installed."
else
  pip install -q git+https://github.com/CyberVy/xfusion.git & transformers-cli env 2> /dev/null & wait
fi


for arg in "$@"; do
  if [[ $arg == --env=* ]]; then
    pair="${arg#--env=}"
    key="${pair%%=*}"
    value="${pair#*=}"
    export "$key=$value"
  fi
done

curl -Lso start.py "https://raw.githubusercontent.com/CyberVy/xfusion/refs/heads/main/start.py"
python3 ./start.py
