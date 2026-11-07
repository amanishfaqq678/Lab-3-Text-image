# Lab 03 — Discussion Answers and Viva Notes

## Part A — Text Feature Engineering

**1. What does each column represent?**  
Each column represents one unique word (feature) found in the corpus. A cell contains the number of times that word occurs in that document.

**2. What is the machine-understandable format?**  
It is a numerical document-term matrix. Each row is a document and each column is a vocabulary word. The numbers are word occurrence counts.

**3. How is Bag-of-Words an oversimplification?**  
It ignores word order, grammar, sentence structure, context, and meaning. Two sentences with the same word counts can have different meanings. It also treats words as separate tokens and does not understand synonyms.

## Part B — Image Feature Engineering

**1. How is grayscale conversion feature engineering?**  
An RGB pixel has three channel values (red, green, blue). Grayscale converts these three values into one intensity value. This reduces the number of features and removes colour information.

**2. If an image has dimensions H × W, what is the flattened vector length after grayscale?**  
The length is H × W because grayscale has one value per pixel. For a 100 × 100 image, the vector contains 10,000 values.

**3. Why is flattening necessary for Logistic Regression or SVM?**  
Flattening converts a two-dimensional pixel matrix into a one-dimensional feature vector. Traditional models generally expect each sample to be represented as a fixed-length feature vector.

**4. Why resize images to a consistent size such as 64 × 64?**  
Different image dimensions produce different vector lengths. Resizing makes every image the same dimensions, so every flattened vector has the same length: 64 × 64 = 4,096. This allows the images to be stacked into one feature matrix and supplied to a model.

## In-Lab Exercises

**Exercise 1 — n-grams**  
`ngram_range=(1,1)` uses single words (unigrams). `ngram_range=(1,2)` uses single words and adjacent two-word sequences (bigrams). The vocabulary becomes larger because phrases such as `sun is` and `is shining` become additional features. The exact feature count is printed by the script.

**Exercise 2 — repeated word**  
The fourth document contains repeated occurrences of `sun`. Its count in the `sun` column increases. Raw counts can favour longer documents because longer documents often contain more total word occurrences.

**Exercise 3 — resizing**  
A 64 × 64 grayscale image produces 4,096 features. A 32 × 32 grayscale image produces 1,024 features. Resizing to smaller dimensions loses fine visual detail.

**Exercise 4 — reusable function**  
`image_to_features(path, size=(64,64))` loads an image, converts it to RGB, resizes it, converts it to grayscale, and flattens it. Three different source image sizes all return a vector of shape `(4096,)`.

## Home Assignment

The notebook asks for 20 labelled images across two classes. With 64 × 64 grayscale images, the matrix has shape `(20, 4096)` and the label vector has shape `(20,)`.

Twenty samples for 4,096 features is not a strong basis for training a reliable model. There are far more features than samples, so a model can overfit and may not generalize to new images. More labelled examples, feature reduction, or a suitable image model would be needed.

## Short Viva Questions

**Q: What is feature engineering?**  
Transforming or creating useful numerical features from raw data for a machine learning algorithm.

**Q: Why can traditional ML algorithms not directly use raw text?**  
They generally require numerical input, so text must be converted into numerical features.

**Q: What is a corpus?**  
A collection of text documents.

**Q: What is Bag-of-Words?**  
A method that represents each document by counting vocabulary word occurrences, while ignoring word order.

**Q: What does CountVectorizer do?**  
It tokenizes text, builds a vocabulary, and converts documents into word-count vectors.

**Q: What does `fit_transform()` do?**  
It learns the vocabulary from the documents and transforms the documents into a feature matrix.

**Q: What does `get_feature_names_out()` return?**  
The vocabulary feature names.

**Q: What does the shape `(3, 11)` mean?**  
Three documents and eleven unique vocabulary features.

**Q: Why is the word `a` absent from the default vocabulary?**  
The default CountVectorizer token pattern selects tokens with at least two word characters, so a single-character token is ignored.

**Q: What is an RGB image?**  
An image with three colour channels: red, green, and blue.

**Q: What is grayscale?**  
A single-channel image in which each pixel represents intensity rather than colour.

**Q: What does PIL stand for?**  
Python Imaging Library; Pillow is its modern maintained fork.

**Q: What is NumPy used for here?**  
Representing and processing image pixels as numerical arrays.

**Q: What does `flatten()` do?**  
It converts a multi-dimensional array into a one-dimensional array.

**Q: Why is a 100 × 100 RGB image 30,000 values?**  
100 × 100 pixels × 3 colour channels = 30,000 values.

**Q: Why is the grayscale flattened vector 10,000 values?**  
100 × 100 pixels × 1 grayscale channel = 10,000 values.

**Q: What is the advantage of resizing?**  
It gives every image the same dimensions and therefore the same feature-vector length.
