# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the code files from the current directory into the container
COPY . /app

# Install the required packages
RUN pip install pandas openai streamlit plotly

# Expose the port that Streamlit is running on
EXPOSE 8501

# Set the entrypoint to run the Streamlit application
CMD ["streamlit", "run", "streamlit_app.py"]
