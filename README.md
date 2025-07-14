# 🧠 Toxic Comment Detector (NLP Project)

This project is a **Natural Language Processing (NLP)** tool that classifies toxic comments. It features **two different models**:

1. ✅ **TF-IDF + Logistic Regression** – A lightweight traditional machine learning model.
2. 🤖 **DistilBERT (Transformer)** – A deep learning model fine-tuned on the Jigsaw Toxic Comment dataset.

Users can train and compare both models and interact with the BERT version through a web interface.

---

## 📂 Project Structure

- `main.py` – Implements the classic ML pipeline (TF-IDF + Logistic Regression)
- `bert_toxic.py` – Trains a DistilBERT model on toxic comment data
- `bert_gradio.py` – Launches a Gradio-based UI for the DistilBERT model
- `train.csv` – Dataset used for training (download separately)
- `toxic-bert/` – Directory containing saved BERT model weights and tokenizer
- `requirements.txt` – List of required Python packages
- `.gitignore` – File/folder ignore rules for version control

---

## 📊 Dataset

This project uses the **Jigsaw Toxic Comment Classification Challenge** dataset.

Due to GitHub file size limits, you must manually download the dataset:

🔗 [Download train.csv from Kaggle](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)

Place the `train.csv` file in the root directory before running the code.

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
