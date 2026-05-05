# Movie Recommendation Using BERT + CNN-BiLSTM

A sentiment-driven movie recommendation system that combines **BERT embeddings**, a **CNN-BiLSTM classifier**, and **Gemini LLM query rewriting** to recommend movies based on free-text user input.

\---

## Overview

This project uses critic reviews from the Rotten Tomatoes dataset to build a recommendation pipeline. Given a natural language query (e.g., *"A heartfelt masterpiece that beautifully captures the complexity of love and loss"*), the system:

1. Rewrites the query into critic-style language using Gemini LLM
2. Encodes it into a BERT embedding
3. Predicts its sentiment (Fresh / Rotten) using a CNN-BiLSTM model
4. Retrieves the most semantically similar movies via cosine similarity on BERT embeddings

\---

## Dataset

**Source:** [Rotten Tomatoes Movies and Critic Reviews Dataset](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset)

Two CSV files are used:

* `rotten\_tomatoes\_critic\_reviews.csv` — critic reviews with sentiment labels
* `rotten\_tomatoes\_movies.csv` — movie metadata (title, genres, etc.)

> \*\*Note:\*\* The dataset is large (\~270 MB zipped). It is stored via Git LFS in this repository.

\---

## Pipeline

### 1\. Data Preprocessing

* Remove unnecessary columns from both datasets
* Deduplicate reviews (keeping rows with fewest missing values)
* Merge on `rotten\_tomatoes\_link`
* Drop rows with null values
* Lowercase text fields
* Strip leading `...` and `\[]` artifacts from review content
* Label encode: `fresh → 1`, `rotten → 0`

### 2\. Dataset Splitting

|Split|Size|
|-|-|
|Train|80% of dataset|
|Validation|10% of train set|
|Test|20% of dataset|

Stratified splitting is used to maintain class balance.

### 3\. Tokenization \& BERT Embeddings

* Tokenizer: `bert-base-uncased`
* Max token length: 100
* Pooling: mean pooling over the last hidden state (masked)
* Embeddings saved as `.npy` files for train, validation, and test splits

### 4\. CNN-BiLSTM Sentiment Classifier

Built in TensorFlow/Keras with the following components:

* `Conv1D` layers for local feature extraction
* `MaxPooling1D` and `BatchNormalization`
* `Bidirectional LSTM` for sequential context
* `Dense` output with sigmoid activation (binary classification)
* **Loss:** Focal Loss (γ=2, α=0.25) to handle class imbalance
* **Optimizer:** Adam
* **Metrics:** Precision, Recall, AUC, F1-Score

Training uses `EarlyStopping` and `ModelCheckpoint` callbacks.

### 5\. Evaluation

* Train vs. Validation loss and AUC plots
* Precision-Recall curves for both classes (Fresh / Rotten)
* Confusion matrix and classification report on the test set

### 6\. Movie Recommendation

Given a user query:

* BERT embedding of the query is computed
* CNN-BiLSTM predicts sentiment (Fresh / Rotten)
* Cosine similarity is computed against all test-set movie embeddings
* Top-K most similar movies matching the predicted sentiment are returned

### 7\. LLM-Enhanced Query (Gemini)

* Uses `gemini-2.5-flash` via Google Generative AI SDK
* Rewrites casual/free-form user input into Rotten Tomatoes critic-style language before embedding
* Improves alignment between query and review embedding space

\---

## Requirements

```bash
pip install torch transformers spacy pandas numpy matplotlib scikit-learn tensorflow tabulate google-generativeai
```

Developed and run on **Google Colab** (GPU recommended for BERT embedding extraction).

\---

## File Structure

```
Movie\_recommendation/
│
├── BERT\_CNN\_Movie\_Recommendation.ipynb   # Main notebook
├── Dataset\_ML.zip                        # Raw + processed datasets (Git LFS)
└── README.md
```

\---

## Usage

1. Open `BERT\_CNN\_Movie\_Recommendation.ipynb` in Google Colab
2. Mount your Google Drive and update the dataset paths
3. Add your Gemini API key where indicated:

```python
   os.environ\["GOOGLE\_API\_KEY"] = "YOUR API KEY"
   ```

4. Run all cells sequentially

When prompted:

```
Enter your movie-related preference
(e.g. A heartfelt masterpiece that beautifully captures the complexity of love and loss.)
```

The system will return top recommended movies with predicted sentiment and confidence score.

\---

## Author

**Maisha Maimuna**

