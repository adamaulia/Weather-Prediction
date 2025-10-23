# 🌦️ Weather Prediction Pipeline

A modular weather prediction system built with **Flask**, and **Docker**, designed to automate data ingestion, preprocessing, model inference, and result visualization. Ideal for learning and deploying end-to-end ML workflows.

---

## 🔑 Getting Your Visual Crossing Weather API Key

To retrieve your API key:

1. Go to [Visual Crossing Sign-Up](https://www.visualcrossing.com/sign-up) and create an account.
2. After logging in, visit your [Account Dashboard](https://www.visualcrossing.com/account).
3. Your API key will be displayed there. You can regenerate it anytime by clicking **"Change Key"**.
4. Use this key in your `.env` file or wherever the project expects it.

---

## ⚙️ Running the Project

### 🧪 Option A: Run via Conda (Local Environment)

1. **Clone the repo**
   ```bash
   git clone https://github.com/adamaulia/Weather-Prediction.git
   cd Weather-Prediction


2. **Create and activate Conda environment**
   ```bash
   conda create -n weather-env python=3.10
   conda activate weather-env
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   VISUAL_CROSSING_API_KEY=your_api_key_here
   ```

5. **Run the Flask app**
   ```bash
   cd app
   python app.py
   ```

6. **Access the API**
   Visit `http://localhost:5000` or use `curl` to test endpoints.

---

### 🐳 Option B: Run via Docker (Containerized)

1. **Ensure Docker and Docker Compose are installed**
   - On Ubuntu: `sudo apt install docker.io docker-compose`
   - On Mac/Windows: Use Docker Desktop

2. **Clone the repo**
   ```bash
   git clone https://github.com/adamaulia/Weather-Prediction.git
   cd Weather-Prediction
   ```

3. **Create `.env` file**
   ```
   VISUAL_CROSSING_API_KEY=your_api_key_here
   ```

4. **Build and run containers**
   ```bash
   docker-compose up --build
   ```

5. **Access services**
   - Flask API: `http://localhost:5000`


---

## 📡 API Usage

Trigger a prediction via POST request:
```bash
curl -X POST http://127.0.0.1:5000/predict_weather 
```

---



## 🧠 Model Info

- Model type: skforecast
- Input features: Temperature
- Output: Predicted weather conditions (temperature)

---

## 📝 Contributing

Feel free to fork, improve, and submit pull requests. Suggestions for better modularity, performance, or documentation are welcome!

---

## 📄 License

This project is licensed under the MIT License.
