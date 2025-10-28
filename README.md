<p align="center">
  <img src="https://img.shields.io/badge/release-v0.0.1-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/github/issues/Moazzam-Siddiqui/COMPLETE-NLP-AND-DEEP-LEARNING" />
  <img src="https://img.shields.io/github/stars/Moazzam-Siddiqui/COMPLETE-NLP-AND-DEEP-LEARNING?style=social" />
  <img src="https://img.shields.io/github/forks/Moazzam-Siddiqui/COMPLETE-NLP-AND-DEEP-LEARNING?style=social" />
  <img src="https://img.shields.io/github/watchers/Moazzam-Siddiqui/COMPLETE-NLP-AND-DEEP-LEARNING?style=social" />
</p>



Natural Language Processing (NLP) — Text Preprocessing & Vectorization

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

### 🧱 Word Embeddings: 
Transformer, BERT , GPT  

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
A process in which we take either Corpus or Documents and convert them into Tokens,which basically do Corpus to Documents,Documents to words.

**Concepts:**
1. Corpus (paragraphs)  
2. Documents (sentences)  
3. Vocabulary (unique words in documents)  
4. Words (all words in corpus)

**FOR UNIQUE WORDS OR VOCABULARY COUNT:-**  
 Also counting Unique words(words that aren't repeated,if it is written once and appears again we will no count that word again in our word)

So if I were to do classifications of words for like checking spam or ham,good reviews or bad reviews,etc I can use Tokenization,Stemming and Lemmatization combining them to make our model understand betterly, like if I were to classify a email in spam or ham the first step is to look out for corpus or paragraphs then convert them to either docs(sentences) or words then correcting the words using Lemmatization or Stemming and finally smartly allow our model to classify them using ML models,there are some disadvantages of using Stemming cuz sometimes it classify words badly increasing error chances so we use Lemmatization.

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

Lemmatization technique is like stemming. The output we will get after lemmatization is called
lemma', which is a root word rather than root stem, the output of stemming. After lemmatization,
we will be getting a valid word that means the same thing.

NLTK provides WordNetLemmatizer class which is a thin wrapper around the wordnet corpus. This
class uses morphy() function to the WordNet CorpusReader class to find a lemma.

lemmatizer takes more time as compared to stemming cuz it uses morphy() function,it's use cases can be Q&A,chatbot,text summarization,etc.



**Library:**  
`WordNetLemmatizer` from **NLTK** uses the WordNet corpus via `morphy()`.

**Use Cases:**  
- Chatbots  
- Text Summarization  
- Q&A Systems  

---

## 🧍‍♂️ NAMED ENTITY RECOGNITION (NER)

A way of NLTK of RECOGNITION names,organization,company,location,etc using pos tags or NER identifies **names, organizations, locations, and companies** using POS tags.  
Example: “Apple” → *Organization*, “India” → *Location*

---

## 🔧 BASIC STEPS OF PREPROCESSING

**Step 1:** Dataset

**Step 2:**  
    1.Tokenize.
    2.lowercase all value to sort out common words not making them Unique(for example:- "The" || "the")
    3.Regular Expression(removing special characters)

**Step 3:**  
- Stemming  
- Lemmatization  
- Stopword removal

**Step 4:**  
After these we try to convert our text data to vectors,VECTORS are basically our numerical representation of our text data,it can be word it can be sentence,it can be paragraph.

---

## 🔢 ONE-HOT ENCODING (OHE)

One of the Text to Numeric Techniques

**Example:**

Let's say our Vocabulary has → [The, Food, Is, Good, Bad, Pizza, Amazing]

D1 = The food is good
D2 = The food is bad
D3 = Pizza is amazing

now if we make matrix of each sentence

**Matrix Representation:**

D1:

| Word | One-Hot Vector |
|------|----------------|
| The |  [1,0,0,0,0,0,0] |
| Food | [0,1,0,0,0,0,0] |
| Is |   [0,0,1,0,0,0,0] |
| Good | [0,0,0,1,0,0,0] |

D2:

| Word | One-Hot Vector |
|------|----------------|
| The | [1,0,0,0,0,0,0]|
| Food | [0,1,0,0,0,0,0]|
| Is |  [0,0,1,0,0,0,0]|
| Good | [0,0,0,0,1,0,0]|

D3:

| Word    | Vector              |
|:--------|:--------------------|
| Pizza   | `[0,0,0,0,0,1,0]`   |
| is      | `[0,1,0,0,0,0,0]`   |
| Amazing | `[0,0,0,0,0,0,1]`   |

**Advantages:**
- Easy to implement (`sklearn.OneHotEncoder`, `pandas.get_dummies()`)

**Disadvantages:**
-.Sparse Matrix - Overfitting ('Very good accuracy with training data but not with new data')
- For ML algorithm we need fixed size but in OHE we can't get it here
- No semantic meaning is getting captured
- Out of Vocabulary

---

## 🧠 Bag of Words (BOW) Representation

**Definition:**
Bag of Words (BOW) is a simple and widely used text representation technique in Natural Language Processing (NLP).  
It converts text into numerical form by representing each sentence or document as a **set of words** (ignoring grammar and word order), and using either **binary values** (1 or 0) or **word frequencies** to indicate the presence or count of each word.

First step is to lower all the words and then use Stopwords.

When we words like basic Vocabulary (he, she, a, is, the, are, etc) it will be ignored cause it doesn't have any sentimental analysis.

For example:-

| Original Sentence | After Removing Stopwords |
|--------------------|--------------------------|
| He is a good boy | good boy |
| She is a good girl | good girl |
| Boy and Girl are Good | Boy Girl Good |

---

### Vocabulary and Frequency

| Vocabulary | Frequency |
|-------------|------------|
| good | 3 |
| boy | 2 |
| girl | 2 |

---

### Sentence Representation

| Sentence | Vector (good, boy, girl) | O/P |
|-----------|---------------------------|-----|
| S1 -> good boy | [1, 1, 0] | 1 |
| S2 -> good girl | [1, 0, 1] | 1 |
| S3 -> Boy Girl Good | [1, 1, 1] | 1 |

---
### Advantages of BOW :-

1.Simple and Intuitive
2.Fixed Size I/P - For ML algorithm

### Disadvantages of BOW:-

1.Sparse Matrix or array -> Overfitting
2.Ordering of the word is getting changed
3.Out of Vocabulary
4.semantic meaning is still not getting captured



### Binary BOW
Binary BOW: {1 and 0}

### Frequency BOW
BOW: {count will be updated based on frequency}


---

## 🔗 N-GRAMS

Combines words to form pairs/triples for context.

Example:


| Vocabulary | Note |
|-------------|------|
| food, not, good | *(“the” and “is” are not present because we have removed them using stopwords)* |

| Sentence | Vector (food, not, good) |
|-----------|--------------------------|
| S1 → The food is good | [1, 0, 1] |
| S2 → The food is not good | [1, 1, 1] |


let's say from S1 we are going to make combination :-
**Bigrams (combination of two words):** food-good, food-not, not-good  
You can clearly see the difference between the vectors.  
Wherever the combination matches, we can represent it with `1`.

### For S1:
| Combination | Vector |
|--------------|---------|
| food not good | 1 |
| food good | 0 |
| food not | 1 |
| not good | 0 |

---

### For S2:
| Combination | Vector |
|--------------|---------|
| food not good | 1 |
| food good | 1 |
| food not | 0 |
| not good | 1 |

sklearn → n-gram= (1,1) → unigrams
  = (1,2) → unigram, bigram
  = (1,3) → unigram, bigram,trigram
  = (2,3) → Bigram, trigram.


**Why use N-Grams?**
-   cuz it is giving us better contextial and semantic meaning of the words, better than it's predecessor BOW.

---

## 🧮 TF–IDF (Term Frequency – Inverse Document Frequency)

S1 → good boy  
S2 → good girl  
S3 → boy girl good  

**TF (Term Frequency):** No. of repetitions of words / No. of words in the sentence  
**IDF (Inverse Document Frequency):** logₑ(No. of sentences / No. of sentences containing the word)

---

### 📊 Term Frequency (TF) and Inverse Document Frequency (IDF)

**TF–IDF (Term Frequency – Inverse Document Frequency)** is a numerical statistic used in **Natural Language Processing (NLP)** and **Information Retrieval** to measure how important a word is to a document in a collection or corpus.




| Word | S1 | S2 | S3 | IDF |
|------|----|----|----|-----|
| good | 1/2 | 1/2 | 1/3 | logₑ(3/3) = 0 |
| boy  | 1/2 | 0 | 1/3 | logₑ(3/2) |
| girl | 0 | 1/2 | 1/3 | logₑ(3/2) |

---

### ⚙️ TF × IDF (Final TF–IDF Matrix)

| Sentence | good | boy | girl |
|-----------|------|-----|------|
| S1 | 0 | 1/2 × logₑ(3/2) | 0 |
| S2 | 0 | 0 | 1/2 × logₑ(3/2) |
| S3 | 0 | 1/3 × logₑ(3/2) | 1/3 × logₑ(3/2) |

---

### ✅ Advantages of TF–IDF
1. Intuitive  
2. Inputs are fixed size → Vocabulary size  
3. Word importance is captured effectively  

---

### ⚠️ Disadvantages of TF–IDF
1. Sparse matrix  
2. Out-of-vocabulary issue  


---

# 🧠 Word Embeddings

### 📖 Definition (Wikipedia)
In **Natural Language Processing (NLP)**, **word embedding** is a term used for representing words for text analysis — typically in the form of a **real-valued vector** that encodes the meaning of a word.  
Words that are **closer in the vector space** are expected to be **similar in meaning**.

---

## 🧩 Word Embedding Techniques

1. **Count or Frequency-based Methods:**  
   - OHE (One Hot Encoding)  
   - BOW (Bag of Words)  
   - TF-IDF  

2. **Deep Learning-based Methods:**  
   - Word2Vec  

---

## 🧠 Word2Vec Types

1. **CBOW (Continuous Bag of Words)**  
2. **Skip-Gram**

---

## 🔍 About Word2Vec

**Word2Vec** is a technique for NLP introduced in 2013.  
It uses a **neural network model** to learn word associations from a large corpus of text.  

Once trained, the model can:
- Detect **synonymous words**
- Suggest additional words for incomplete sentences  

Each word is represented by a list of numbers called a **vector**.

---

## 🧾 Example Vocabulary

**Vocabulary:**  
Boy, Girl, King, Queen, Apple, Mango  

Each word in the vocabulary is converted into a **feature representation vector** based on attributes (like Gender, Age, Royal, Food, etc.).  
If we assume there are `n` such features, each word will have `n` dimensions (e.g., 300).

| Feature | Boy | Girl | King | Queen | Apple | Mango |
|----------|-----|------|------|--------|--------|--------|
| Gender | 1 | -1 | -0.92 | 0.93 | -0.01 | 0.01 |
| Royal | 0.01 | 0.02 | -0.95 | 0.96 | -0.01 | 0.01 |
| Age | 0.03 | 0.02 | 0.75 | 0.68 | -0.95 | 0.96 |
| Food | - | - | - | - | -0.99 | 0.99 |
| ... | ... | ... | ... | ... | ... | ... |
| nth | - | - | - | - | - | - |

> Feature representations can extend to hundreds or thousands of dimensions.

---

## ✴️ Famous Word2Vec Calculation

A well-known example from Google’s research:


---

## 🔄 SkipGram

**Definition:**  
SkipGram is just the **opposite of CBOW**.  
While CBOW predicts the **target word** based on **context words**,  
SkipGram predicts **context words** based on a **target word**.

---

### 🧩 Example
**Window Size = 5**

| O/P (Context Words) | I/P (Target Word) |
|----------------------|-------------------|
| iNeuron, company, related, to | is |
| company, related, to, Data | related |
| related, to, Data, Science | to |

---

### ⚙️ When to Use CBOW vs SkipGram

| Dataset Size | Recommended Model |
|---------------|-------------------|
| Small dataset | **CBOW** |
| Large dataset | **SkipGram** |

---

### 💡 Notes

1. To improve CBOW or SkipGram performance, **increase the training data**.  
2. **Increasing window size** also increases **vector dimensions**.  
   > Larger window → larger vector dimension → richer context representation

---

## ✅ Advantages of Word2Vec

1. Produces **dense matrices** instead of sparse ones.  
2. **Captures semantic information** effectively.  
3. Provides a **fixed set of dimensions** for all words.  
4. Only a few words fall **Out of Vocabulary (OOV)**.

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

For Architectural Working:
<p align="center">
  <img src="./Images/img1.jpg" alt="NLP Diagram" width="600">
  <img src="./Images/img2.jpg" alt="NLP Diagram" width="600">
  <img src="./Images/img3.jpg" alt="NLP Diagram" width="600">
  <img src="./Images/img4.jpg" alt="NLP Diagram" width="600">
</p>

## 🔁 Sequential Data

1. **Definition:**  
   Sequential data refers to data that has a **meaning or relationship** with its previous inputs.  
   It can be text, time series, or any other form of data where the order matters.  

   **Example:**  
   - Sentence: *“The food is good.”*  
     Here, “GOOD” is predicted based on the previous words (“The food is”), similar to how text generation models work.  
     Even if the next word could be “average” or “bad,” the model learns to make a prediction based on prior context.

---

### 💬 Examples of Sequential Data
2. **Language Translation**  
3. **Auto Suggestions** (LinkedIn, Gmail, etc.)  
4. **Sales Data** – especially when analyzed based on date and time for forecasting.

---

## ⚙️ Why Prefer ANN over RNN?

In short, when generating outputs:  
- The model takes inputs along with **initialized weights**.  
- Adds a **bias**.  
- Applies a **basic activation function** to produce the output.  

While ANNs work on independent data points, RNNs handle **sequential dependencies**, but ANNs are still preferred in some simpler cases where context or time-sequence is not crucial.


## **Simple RNN (Recurrent Neural Network)**

### **Definition:**
A **Recurrent Neural Network (RNN)** is a type of neural network designed to handle **sequential data** — data where the current output depends on previous inputs.  
It has a special structure where the **output from one step is fed back as input to the next step**, allowing the network to remember information over time.

For example:  
In a sentence like *“The food is good”*, the meaning of the word **“good”** depends on the previous words — *“The food is”*.  
RNNs are used for tasks like **text generation**, **speech recognition**, **machine translation**, and **time series forecasting**.

**Key idea:**  
RNN = ANN + Memory (it remembers past information while processing new data)

---

### **Dataset Example**

| Text | O/P |
|------|-----|
| S1: The food is good | 1 |
| S2: The food is bad | 0 |
| S3: The food is not good | 0 |

The yellow words represent the **vocabulary** or **unique words** in our dataset.

---

### **One-Hot Encoding Representation**

Before sending text data to an RNN, we first convert words into **numerical vectors** using a simple encoding method called **One-Hot Encoding**.

| Word | [The, Food, Good, Bad, Not] |
|------|------------------------------|
| the  | [1, 0, 0, 0, 0] |
| food | [0, 1, 0, 0, 0] |
| good | [0, 0, 1, 0, 0] |
| bad  | [0, 0, 0, 1, 0] |
| not  | [0, 0, 0, 0, 1] |

---

### **Forward Propagation in RNN**

Once the words are encoded, they are fed **one at a time** into the RNN.  
At each **time step**, the RNN processes one word and passes its **hidden state** (memory) to the next step.

This means the network understands **context** and **sequence**, not just isolated words.

Example flow for:  
**“The food is good”**

For BlackBox Understanding:

#### RNN:

<p align="center">
  <img src="./Images/RNN1.jpg" alt="NLP Diagram" width="600">
</p>

#### RNN Parameters Calculations:

<p align="center">
  <img src="./Images/RNN2.jpg" alt="NLP Diagram" width="600">
</p>

#### Forward Propagation:

<p align="center">
  <img src="./Images/ForwardPropagation.jpg" alt="NLP Diagram" width="600">
</p>

**Why prefer ANN over RNN:**

So in short, whatever output we are getting after,we probably take the inputs along with the weights that are initialized, then we add a bias then we add a basic activation function

*y = f(x1w1 + x2w2 + … xnwn + b1 … bn)*

y = output  
X = inputs  
W = weights  
B = bias  
F = activation function  

There are many activation functions like sigmoid for binary output, softmax for multiclass classification and ReLu for CNN.

## RNN BACK PROPAGATION WITH TIME:

We've seen how we have calculated our forward propagation but still there is lose function which was  
loss = y - y`.  
So ,to reduce the all that loss we tends to approach backward propagation and we have to update all the weights that are there

[wi, wh ,wo]

And this will happen when our loses will be really really less  or our global minima will be entirely down.

### How we will do that:

We will use the weight update formula ,which is :-

### Weight Updation Formula

We update the weights during backpropagation using the following formula:

w_new = w_old - η * (∂h / ∂w_old)

where,  
- w_new → updated weight  
- w_old → previous weight  
- η (eta) → learning rate  
- ∂h / ∂w_old → gradient of the loss with respect to the old weight

### Simple ANN and RNN Projects

Now we are going to do some simple ANN and RNN projects using libraries like **Keras** and **TensorFlow**.

#### We will do:

1. Take **Churn Modelling Dataset**  
2. Perform **Classification** with basic **Feature Engineering**  
3. Convert **Categorical variables** into **Numerical**  
4. Apply **Standardization**  
5. Then try to **create an ANN**  
6. We will also use **Dropout** (disabling some of the nodes while doing forward and backward propagation)  
   - This helps our model **not to overfit**, as some weights will not be updated.  
7. Use **Optimizers** and **save model files** in `.pickle` or `.h5` formats  

Finally, we will use **Streamlit** for the **deployment** of our web model.
### Building a Neural Network

1. **Sequential Network (N/W)**  
   - Used to build a model layer by layer.

2. **Dense Layer**  
   - Example: 64 neurons  

3. **Activation Functions**  
   - `sigmoid`, `tanh`, `ReLU`, `leaky ReLU`

4. **Optimizer**  
   - Performs **Backpropagation** → updates the weights.

5. **Loss Function**  
   - Measures how far the predicted output is from the actual output.

6. **Metrics**  
   - Example: `[accuracy]`, `[mse, mae]`

7. **Training Process**  
   - Generate **logs** → store in **folders** → visualize with **TensorBoard**

# 🧠 Some Keywords within Neural Nets

### Epochs
In machine learning, an **epoch** means **one full pass through the entire training dataset by the learning algorithm**.

Here’s the breakdown:

- Imagine you have **1,000 training samples**.

- In 1 epoch, your model sees all 1,000 samples once, adjusts weights, and completes a full training cycle.

- If you set epochs=10, the model will see all 1,000 samples 10 times (weights updated after each batch/iteration).

**🔹 Why not just one epoch?**

One pass isn’t enough for the model to learn patterns well. Multiple epochs help it refine and improve accuracy.

Too many epochs → **overfitting** (model memorizes training data instead of generalizing).

### Validation Split:

validation_split is a parameter in Keras/TensorFlow that reserves a fraction of your training data for validation.

Example:

    model.fit(X, y, epochs=10, validation_split=0.2)

This means:

- 80% → training

- 20% → validation

**Purpose:**

Model trains on training part

After each epoch, it tests on validation part

Helps track **overfitting/underfitting**.

**Important Notes:**

Works only if dataset fits in memory (NumPy array / TensorFlow dataset).

Split happens **before shuffling**, unless data is already shuffled.


# 🧮 Loss Function

A **loss function** is the mathematical formula a model tries to minimize during training.  
It measures how far off predictions are from actual target values.

> 💡 **Lower loss → better model fit**

The **optimizer** uses the loss to adjust model weights during each training iteration.

---

## 📘 Common Loss Functions

| **Problem Type**              | **Loss Function**                | **When to Use** |
|-------------------------------|----------------------------------|-----------------|
| **Binary Classification**     | `binary_crossentropy`            | Output layer has 1 neuron (sigmoid). |
| **Multi-class Classification**| `categorical_crossentropy`       | Labels are one-hot encoded, output uses softmax. |
|                               | `sparse_categorical_crossentropy`| Labels are integers (not one-hot). |
| **Regression**                | `mean_squared_error (MSE)`       | Predicting continuous values. |
|                               | `mean_absolute_error (MAE)`      | Continuous values; less sensitive to outliers. |
| **Special Cases**             | `hinge`                          | SVM-style classification. |
|                               | `kl_divergence`                  | Comparing probability distributions. |

---

### 🧠 Summary
- **Goal:** Minimize loss to improve model accuracy.  
- **Optimizer:** Uses gradients from the loss to update weights.  
- **Choice:** Depends on problem type (classification, regression, etc.).


**Example in Keras**
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # loss function
    metrics=['accuracy']         # for monitoring
)
