# Fraudulent Job Posting Detection

## Overview

This repository contains the ML pipeline I developed for GovTech submission. The primary objective is to detect fraudulent job postings accurately without creating operational friction for legitimate employers.

Because scams make up only about 4.85% of this dataset, it is easy to build a model that appears highly accurate simply by guessing that every job is legitimate. The challenge lies in identifying the rare scams while keeping false alarms to a minimum. To achieve this, I designed a system that combines text analysis with metadata heuristics, utilises a custom confidence threshold, and relies on a HITL routing system for borderline cases.

## Methodology

### Beyond Text

Scammers tend to leave metadata footprints such as omitting a company logo or skipping screening questions. Instead of relying purely on a text-reading algorithm, I built a Multi-Input Neural Network using the Keras Functional API. One branch of the network processes the job description text using a TF-IDF matrix, whilst a second branch processes the structured yes/no metadata. The network merges these two branches to make a far more informed final decision.

### Class Imbalance

To handle the heavily imbalanced data, I used stratified sampling during the train-test split to guarantee the exact scam ratio was maintained (4.85%). I also applied class weights during training. This mathematically penalises the neural network whenever it misses a scam, forcing it to pay attention to the minority class rather than taking the easy route of optimising for the 95% of legitimate posts.

<img width="295" height="88" alt="image" src="https://github.com/user-attachments/assets/3717b57b-cd92-4430-a745-ff7bb01e0a53" />

### Prioritising User Experience

A standard ML model blocks anything with a scam probability above 50%. I found this caused too many false positives, wrongfully penalising real companies. I calibrated the system to use a 0.75 confidence threshold. By demanding higher certainty, I sacrificed an amount of overall scam detection in exchange for a 37.6% reduction in false alarms.

### Routing System

Instead of forcing the system to make a binary guess on ambiguous postings, I implemented a three-tier system to simulate a real environment:

* **Auto-Allow (Score < 0.40):** Clearly legitimate jobs are published instantly, causing zero friction.
* **Auto-Block (Score >= 0.85):** Obvious scams are blocked immediately.
* **Manual Review (0.40 to 0.84):** Borderline jobs trigger an alert containing the risk score and missing heuristics. This routes the ambiguous cases for human review, avoiding wrongful automated bans.

<img width="212" height="139" alt="image" src="https://github.com/user-attachments/assets/5e49d634-14cc-49b8-8bff-f9eed1403d38" />

## Tech Stack

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy, Regular Expressions (re)
* **Feature Engineering:** Scikit-learn (TfidfVectorizer)
* **Deep Learning:** TensorFlow, Keras
* **Evaluation & Visuals:** Matplotlib, Seaborn

## Performance Results

Evaluated on a strictly isolated 20% test set, the tuned model and routing logic achieved:

* **Overall Accuracy:** 98%
* **Minority Class (Scam) F1-Score:** 0.77
* **Net Impact:** Successfully caught the vast majority of scams whilst drastically reducing the operational friction placed on legitimate users.

<img width="517" height="610" alt="image" src="https://github.com/user-attachments/assets/bdc7e3fa-6e9d-4bde-ac9f-eb7f3976fdfe" />


## How to Run the Code

1. Clone this repository and ensure you have the required libraries installed:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn requests

```


2. Open and run the `Scam_Detection.py` script or its corresponding notebook.


3. When prompted in the first cell, upload your dataset (CSV or Excel). The script will automatically detect the file type and load it into memory.
4. The program will execute sequentially, displaying the data shapes, the class imbalance breakdown, the neural network training progress, the custom threshold confusion matrix, and the final routing summary for the moderation queue.

## Future Enhancements

- Replace the TF-IDF vectoriser with dense word embeddings (such as Word2Vec or an LLM embedding API) so the network understands contextual nuance rather than merely counting word frequencies
- Engineer network-level features to track whether a specific phone number or email address is being reused across multiple separate job postings within a 24-hour window
- Expand the metadata to include external validation APIs, cross-referencing company names against corporate registries or government databases
- Integrate advanced methodologies, combining the deep learning with tree-based classifiers (such as LightGBM or Random Forest) to capture complex, non-linear dependencies more effectively
- Package and deploy the model as a real-time fraud detection API, allowing integration with employment platforms
