#!/bin/bash

# Start TinyLLaMA with llama.cpp
./llama.cpp/server -m model/tinyllama.gguf --port 1234 --host 0.0.0.0 &

# Wait a few seconds to ensure model starts
sleep 10

# Launch the Streamlit app
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
