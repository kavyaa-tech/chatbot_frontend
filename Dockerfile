FROM python:3.10

# System dependencies
RUN apt-get update && apt-get install -y \
    git build-essential cmake wget curl libopenblas-dev libsqlite3-dev

# Install llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && mkdir build && cd build && \
    cmake .. && make -j && cd ../..

# Copy app and model
COPY . /app
WORKDIR /app

# Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Make script executable
RUN chmod +x start.sh

# Expose Streamlit port
EXPOSE 7860

# Start script
CMD ["./start.sh"]
