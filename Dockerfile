FROM python:3.x

# Set the working directory in the container to /app
WORKDIR /app

# add the current directory to the container at /app
ADD . /app

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt