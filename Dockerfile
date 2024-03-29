FROM python:3.9

# Set the working directory in the container to /app
WORKDIR /app

# add the current directory to the container at /app
ADD . /app

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pytest

# Run the tests when the container launches
CMD ["pytest"]
