# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (adjust if you're using something other than Flask)
EXPOSE 8080

# Run the chatbot
CMD ["python", "app.py"]
