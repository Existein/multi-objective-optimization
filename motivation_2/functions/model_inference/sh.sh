
sudo find . -type f -name "*.so" -exec strip {} + 2>/dev/null
sudo find . -type f -name "*.pyc" -exec rm {} +
sudo find . -type f -name "*.npz" -exec rm {} +
sudo find . -type f -name "*.dat" -exec rm {} +
sudo find . -type d -name "__pycache__" -exec rm -r {} +
sudo find . -type d -name "*.dist-info" -exec rm -r {} +
sudo find . -type d -name "tests" -exec rm -r {} +