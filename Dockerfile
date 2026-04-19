# 1. Start with a lightweight, official Python operating system
FROM python:3.11-slim

# 2. Set the working directory inside the box
WORKDIR /app

# 3. Copy our shopping list into the box
COPY requirements.txt .

# 4. Install the packages (silently)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy ALL our code, our mlruns folder (with the model!), and everything else into the box
COPY . .

# 6. Tell the box to listen on port 8000
EXPOSE 8000

# 7. The command to start the server when the box turns on
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]