FROM python:3.10-slim-bullseye

WORKDIR /app

# Ensure we have required system dependencies (especially since Docling requires ONNX Runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and rules
COPY src/ src/
COPY rubric/ rubric/
COPY tests/ tests/
COPY run_*.py ./

# Create output directories mapped via volume
RUN mkdir -p .refinery/profiles
RUN mkdir -p .refinery/extractions
RUN mkdir -p .refinery/ldus
RUN mkdir -p .refinery/pageindex
RUN mkdir -p .refinery/chroma_db

# The default command will just run the unit tests or keep it alive,
# relying on `docker exec` to run the specific batch or single-doc scripts
CMD ["python3", "-m", "pytest", "tests/"]
