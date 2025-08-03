if python3 -m pip show xfusion > /dev/null 2>&1; then
  echo "xfusion is installed."
else
  pip install -q git+https://github.com/CyberVy/xfusion.git & transformers-cli env 2> /dev/null & wait
fi

