FROM python:3.14-alpine AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && find /install \( -name "test" -o -name "tests" \) -type d \
       -exec rm -rf {} + 2>/dev/null; true

FROM python:3.14-alpine
COPY --from=builder /install /usr/local
WORKDIR /app
COPY run.py .
COPY src/ src/
COPY config.yml .

# Mount config.yml and input file at runtime
# Example: docker run --rm -v $(pwd):/data myimage --config /data/config.yml
ENTRYPOINT ["python", "run.py"]
CMD ["--config", "config.yml"]
