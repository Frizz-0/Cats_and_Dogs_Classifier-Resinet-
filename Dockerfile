# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# # Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

# ARG PYTHON_VERSION=3.12.10
# FROM python:${PYTHON_VERSION}-slim as base

# # Prevents Python from writing pyc files.
# ENV PYTHONDONTWRITEBYTECODE=1

# # Keeps Python from buffering stdout and stderr to avoid situations where
# # the application crashes without emitting any logs due to buffering.
# ENV PYTHONUNBUFFERED=1

# WORKDIR /app

# # Create a non-privileged user that the app will run under.
# # See https://docs.docker.com/go/dockerfile-user-best-practices/
# ARG UID=10001
# RUN adduser \
#     --disabled-password \
#     --gecos "" \
#     --home "/nonexistent" \
#     --shell "/sbin/nologin" \
#     --no-create-home \
#     --uid "${UID}" \
#     appuser

# # Download dependencies as a separate step to take advantage of Docker's caching.
# # Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# # Leverage a bind mount to requirements.txt to avoid having to copy them into
# # into this layer.
# RUN --mount=type=cache,target=/root/.cache/pip \
#     --mount=type=bind,source=requirements.txt,target=requirements.txt \
#     python -m pip install -r requirements.txt

# # Switch to the non-privileged user to run the application.
# USER appuser

# # Copy the source code into the container.
# COPY . .

# # Expose the port that the application listens on.
# EXPOSE 8000

# # Run the application.
# CMD python -m streamlit run main.py



# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime  
FROM python:3.10-slim AS builder

WORKDIR /catsndogs

COPY requirements.txt .

# COPY main.py .
# COPY CatsnDogs.pt .

# RUN pip install --no-cache-dir --target=/Lib/site-packages -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir streamlit torchvision torch --index-url https://download.pytorch.org/whl/cpu

RUN pip list


FROM python:3.10-slim

WORKDIR /catsndogs

# COPY --from = builder /Lib/site-packages

COPY . .

EXPOSE 8501

RUN rm -rf /root/.cache

CMD ["python","-m","streamlit", "run", "main.py"]