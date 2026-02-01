# ♟️ Chess Piece Image Classifier (Flask + TensorFlow)

Невеликий ML-проєкт на Python Flask, який приймає зображення шахової фігури та повертає, яка саме фігура зображена (pawn, rook, knight, bishop, queen, king) разом з ймовірністю передбачення.

Проєкт використовує **заздалегідь натреновану CNN-модель TensorFlow** та REST API на Flask.

---

## 📦 Dataset & Training

- 📂 **Dataset:**  
  👉 *https://www.kaggle.com/datasets/niteshfre/chessman-image-dataset*

- 📓 **Google Colab (training notebook):**  
  👉 *https://colab.research.google.com/drive/13lb1e9h99Qg-h48vpv08AHnlY4CV_jT7#scrollTo=HfiOsa2i_m_W*

> ⚠️ Порядок класів у `classifier.py` **має відповідати порядку класів під час тренування**.

---

## 🧠 Підтримувані класи

Модель класифікує 6 шахових фігур:

- pawn
- rook
- knight
- bishop
- queen
- king

---

## 🏗️ Структура проєкту

``` text
.
├── app.py # Flask API
├── classifier.py # Image preprocessing + inference
├── static/
│ ├── models/
│ │ └── chess_model.h5 # Натренована модель
│ └── uploads/ # Тимчасові завантажені зображення
├── requirements.txt
└── README.md

```


---

## 🚀 Запуск проєкту локально

### 1 Клонувати репозиторій
```bash
git clone <repo_url>
cd ml_learning_chess
```
### 2 Створити та активувати virtual env
```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
```
### 3 Встановити залежності
```bash
pip install -r requirements.txt
```
### 4 Запустити Flask серевер
```bash
python ap.py
```
🔌 Використання API
POST /classify
📥 Запит

Тип: multipart/form-data

Поле: file (зображення шахової фігури)

### Приклад через curl:
```bash
curl -X POST http://127.0.0.1:5000/classify \
  -F "file=@queen.png"
```
### 📤 Відповідь:
```json
{
  "figure": "queen",
  "confidence": 92.47
}
```

## 🛠️ Технології

Python 3.10+

Flask

TensorFlow / Keras

NumPy

## ⚠️ Важливі нотатки

Preprocessing під час inference повинен збігатися з preprocessing під час training

Класи у CLASS_NAMES мають бути в правильному порядку

API очікує одну фігуру на зображенні

## Приклад тесту через Postman
![DEMO](assets/demo.png)