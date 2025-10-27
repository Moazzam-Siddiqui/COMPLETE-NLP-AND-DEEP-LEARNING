# 🧠 Natural Language Processing (NLP) — Text Preprocessing & Vectorization

---

## 🧹 CLEANING THE INPUT

### 🧾 Text Preprocessing
- **Tokenization:** Technique to change paragraph → sentence → words (called as *token*).  
- **Lemmatization:** Technique to find lemma (context) of a word.  
- **Stemming:** Technique to find stem (root) of a word.

---

## 🔡 CONVERTING TEXT TO VECTORS

### 🧩 Text Preprocessing — Part 2
Converting text data into numerical vectors.  
**Techniques Used:**  
- Bag of Words (BOW)  
- TF-IDF  
- Unigram  
- Bigram  

### 🧠 Text Preprocessing — Part 3
- Word2Vec  
- AvgWord2Vec  

---

## 🤖 DEEP LEARNING TECHNIQUES

### 🧬 Neural Networks
- RNN  
- LSTM-RNN  
- GRU-RNN  

### 🧱 Word Embeddings, Transformer, BERT  

**Libraries:**  
- NLTK, SpaCy for ML  
- TensorFlow, PyTorch for DL  

---

## 💡 USE CASES OF NLP
- Auto Correct  
- Auto-Generated text for email  
- Auto-Reply (LinkedIn, WhatsApp, etc.)  
- Google Translate  
- Image Search (Google Images)  
- Hugging Face Models  

---

## ✂️ TOKENIZATION

**Definition:**  
Breaking a corpus (paragraphs) into documents (sentences), then into words (tokens).  

**Concepts:**
1. Corpus (paragraphs)  
2. Documents (sentences)  
3. Vocabulary (unique words in documents)  
4. Words (all words in corpus)

**Vocabulary Count:**  
Unique words are counted only once — repeated words are ignored.

---

### ⚙️ Example: Spam/Ham Classification
To classify emails (spam/ham) or reviews (good/bad):
1. Tokenize corpus → documents → words.  
2. Apply Lemmatization / Stemming to clean data.  
3. Use ML models for classification.  

**Note:** Stemming can sometimes misclassify words, so Lemmatization is often preferred.

---

## 📚 WORDNET LEMMATIZER

**Definition:**  
Lemmatization gives the valid *root form* (lemma) of a word.  
Unlike stemming, output words are valid and meaningful.

**Library:**  
`WordNetLemmatizer` from **NLTK** uses the WordNet corpus via `morphy()`.

**Use Cases:**  
- Chatbots  
- Text Summarization  
- Q&A Systems  

---

## 🧍‍♂️ NAMED ENTITY RECOGNITION (NER)

NER identifies **names, organizations, locations, and companies** using POS tags.  
Example: “Apple” → *Organization*, “India” → *Location*

---

## 🔧 BASIC STEPS OF PREPROCESSING

**Step 1:** Dataset  
**Step 2:**  
- Tokenization  
- Lowercasing  
- Removing special characters (regex)

**Step 3:**  
- Stemming  
- Lemmatization  
- Stopword removal

**Step 4:**  
Convert text to vectors (numerical representations of text).

---

## 🔢 ONE-HOT ENCODING (OHE)

**Example:**

Vocabulary → [The, Food, Is, Good, Bad, Pizza, Amazing]

D1 = The food is good
D2 = The food is bad
D3 = Pizza is amazing


**Matrix Representation:**

| Word | One-Hot Vector |
|------|----------------|
| The | [1,0,0,0,0,0,0] |
| Food | [0,1,0,0,0,0,0] |
| Is | [0,0,1,0,0,0,0] |
| Good | [0,0,0,1,0,0,0] |

**Advantages:**
- Easy to implement (`sklearn.OneHotEncoder`, `pandas.get_dummies()`)

**Disadvantages:**
- Sparse matrix → Overfitting  
- No semantic meaning  
- Out of vocabulary (OOV) issue  
- Fixed size input required  

---

## 🧺 BAG OF WORDS (BoW)

**Example:**

S1 = "He is a good boy" → "good boy"
S2 = "She is a good girl" → "good girl"
S3 = "Boy and girl are good"


| Word | Frequency |
|------|------------|
| good | 3 |
| boy | 2 |
| girl | 2 |

**Advantages:**
- Simple and intuitive  
- Fixed-size input for ML  

**Disadvantages:**
- Sparse matrix → Overfitting  
- Loses word order  
- Out of vocabulary  
- No semantic meaning  

---

## 🔗 N-GRAMS

Combines words to form pairs/triples for context.

Example:

S1: The food is good
S2: The food is not good


**Bigrams:** food-good, food-not, not-good  
**Trigrams:** The-food-is, food-is-good

**Why use N-Grams?**
- Captures contextual and semantic meaning better than BoW.

---

## 📊 TF-IDF (Term Frequency–Inverse Document Frequency)

### Term Frequency (TF)

### TF = (No. of times word appears) / (Total words)

### Inverse Document Frequency (IDF)

###  IDF = log_e(Total Documents / Documents containing word)


| Word | S1 | S2 | S3 | IDF |
|------|----|----|----|-----|
| good | 1/2 | 1/2 | 1/3 | log(3/3)=0 |
| boy | 1/2 | 0 | 1/3 | log(3/2) |
| girl | 0 | 1/2 | 1/3 | log(3/2) |

**TF-IDF = TF × IDF**

**Advantages:**
- Captures word importance  
- Fixed-size input  

**Disadvantages:**
- Sparse matrix  
- Out of vocabulary  

---

## 🧭 WORD EMBEDDINGS

**Definition:**  
Word embeddings represent words as vectors in continuous vector space — similar words are close together.

**Types:**
1. **Count/Frequency-based:** OHE, BoW, TF-IDF  
2. **Deep Learning-based:** Word2Vec  

---

## 🧩 WORD2VEC

**Types:**
1. **CBOW (Continuous Bag of Words)**  
2. **Skip-Gram**

Trained using a neural network (2013).  
Finds word associations — synonymous or related words.

**Vocabulary Example:**

Boy, Girl, King, Queen, Apple, Mango


| Feature | Boy | Girl | King | Queen | Apple | Mango |
|----------|-----|------|------|-------|--------|--------|
| Gender | 1 | -1 | -0.92 | 0.93 | -0.01 | 0.01 |
| Royal | 0.01 | 0.02 | -0.95 | 0.96 | -0.01 | 0.01 |
| Age | 0.03 | 0.02 | 0.75 | 0.68 | -0.95 | 0.96 |
| Food | - | - | - | - | -0.99 | 0.99 |

**Famous Relation Example:**

King - Man + Queen = Woman
King - Boy + Queen = Girl


These relationships define vector-space semantics.

**Google Word2Vec:**  
Trained on 3 billion words, usually 300 dimensions per vector.

---

## 📐 COSINE SIMILARITY

**Formula:**
Distance = 1 - cos(θ)


**Interpretation:**
- Smaller distance → higher similarity  
- 0° → completely similar (distance = 0)  
- 90° → unrelated (distance = 1)

Used to measure similarity between feature vectors like movie genres, text, etc.

---

## 🧮 CBOW (Continuous Bag of Words)

**Definition:**  
A neural network model predicting a word based on its context (surrounding words).

**Example Corpus:**

[iNeuron company is related to Data Science]


**Window Size = 5**

| Input | Output |
|-------|---------|
| iNeuron, company, related, to | is |
| company, related, to, Data | related |
| related, to, Data, Science | to |

**Key Notes:**
- Choose **odd** window sizes (e.g., 5) for balanced context.  
- Larger window → better performance, more context.  
- CBOW is a fully connected neural network.  

---

For Architectural Working:
<p align="center">
  <img src="./Images/image.png" alt="NLP Diagram" width="600">
</p>

## 🧠 Skip-Gram

**Definition:**  
Skip-Gram is the inverse of CBOW. Given a **target word**, Skip-Gram predicts the **surrounding context words**.

**Window Size:** `5`

| **Output (Context Words)**                | **Input (Target Word)** |
|-------------------------------------------|--------------------------|
| iNeuron, company, related, to             | is                       |
| company, related, to, Data                | related                  |
| related, to, Data, Science                | to                       |

### ⚖️ When to use
- **CBOW** → better for **small datasets** (faster training).  
- **Skip-Gram** → better for **large datasets** and for learning **good vectors for rare words**.

### 📈 Notes
1. Increasing the **training data** typically improves both CBOW and Skip-Gram.  
2. Increasing the **window size** gives the model more context and often increases the vector dimensionality/representational capacity.

---

## 🚀 Advantages of Word2Vec

1. Produces **dense** word vectors (not sparse).  
2. Captures **semantic** relationships (similar words are close in vector space).  
3. Uses a **fixed vector dimensionality** (e.g., 100, 300).  
4. Only a small portion of words may be **OOV** depending on training.

---

## 🤖 Deep Learning & Neural Networks

### ANN (Artificial Neural Network)

**Overview:**  
An ANN is composed of layers of connected neurons (input layer → one or more hidden layers → output layer). Each connection has a weight; activations travel forward and errors propagate backward during training.

**Basic Components:**
- **Neuron:** computes weighted sum + bias → activation function.  
- **Layers:** Input, Hidden, Output.  
- **Activation functions:** ReLU, Sigmoid, Tanh, Softmax (for classification).  
- **Loss functions:** MSE, Cross-Entropy.  
- **Optimization:** Gradient Descent variants (SGD, Adam).

**Typical Workflow:**
1. Prepare and normalize input data.  
2. Define architecture (layers, units, activations).  
3. Choose loss and optimizer.  
4. Train with mini-batches and validate.  
5. Evaluate on test set and tune hyperparameters.

---


