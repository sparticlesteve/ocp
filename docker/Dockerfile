FROM nvcr.io/nvidia/pytorch:20.12-py3

# PyG
RUN pip install --no-cache-dir \
    -f https://data.pyg.org/whl/torch-1.8.0+cu111 \
    torch-scatter torch-sparse torch-geometric

# MLPerf logging
RUN pip install --no-cache-dir git+https://github.com/mlcommons/logging.git

# Other packages
RUN pip install --no-cache-dir ray submitit
