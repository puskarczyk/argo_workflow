FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy matplotlib seaborn
COPY 06_plot_results.py .