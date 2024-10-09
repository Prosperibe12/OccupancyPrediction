FROM python:3.10.12

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Install dependencies using the virtual environment's pip
RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt && \
    chmod +x entrypoint.sh

# Use the entrypoint script to run the app
CMD [ "/app/entrypoint.sh" ]