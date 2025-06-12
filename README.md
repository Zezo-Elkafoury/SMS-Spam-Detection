# SMS Spam Detection

This project aims to build a machine learning model to detect SMS spam messages. The project involves data loading, cleaning, exploratory data analysis (EDA), text preprocessing, and training a classification model.

## Project Steps

1.  **Data Loading and Initial Exploration:**
    *   Loaded the dataset (`spam.csv`) and performed initial checks on its structure and content using `df.head()`, `df.info()`, and `df.describe()`.

2.  **Data Cleaning:**
    *   Removed irrelevant columns ('Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4').
    *   Handled missing values (checked using `df.isnull().sum()`).
    *   Removed duplicate entries (checked using `df.duplicated().sum()` and removed with `df.drop_duplicates()`).
    *   Renamed columns for clarity ('v1' to 'type', 'v2' to 'text').
    *   Encoded the target variable ('type') using `LabelEncoder`.

3.  **Exploratory Data Analysis (EDA):**
    *   Visualized the distribution of message types (spam vs. non-spam) using a pie plot.
    *   Calculated and added new features:
        *   `num_chars`: Number of characters in each message.
        *   `num_words`: Number of words in each message (using `nltk.word_tokenize`).
        *   `num_sentences`: Number of sentences in each message (using `nltk.sent_tokenize`).
    *   Analyzed the descriptive statistics of these new features for both spam and non-spam messages.
    *   Visualized the distribution of message lengths (characters and sentences) for spam and non-spam messages using histograms.
    *   Explored correlations between the numerical features using a heatmap.

4.  **Data Preprocessing for Machine Learning:**
    *   Created a `transform_text` function to preprocess the text data:
        *   Convert text to lowercase.
        *   Tokenize the text.
        *   Remove non-alphanumeric characters.
        *   Remove English stop words and punctuation.
        *   Apply stemming using `PorterStemmer`.
    *   Applied the `transform_text` function to the 'text' column to create a new 'final\_text' column.
    *   Generated a word cloud for spam messages to visualize the most frequent words.

5.  **Machine Learning Model Building:**
    *   Transformed the 'final\_text' data into numerical features using `TfidfVectorizer` with `max_features=3000`.
    *   Split the data into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`).
    *   Trained a `RandomForestClassifier` model.
    *   Evaluated the model's performance on the test set using:
        *   Accuracy score
        *   Precision score
        *   Confusion matrix (visualized using a heatmap).
    *   Performed 5-fold cross-validation to assess model generalization, reporting mean accuracy, precision, and recall.

6.  **Model Saving:**
    *   Saved the trained `RandomForestClassifier` model and the `TfidfVectorizer` to pickle files (`model.pkl` and `vectorizer.pkl`) for later use.

## Technologies Used

*   Python
*   Pandas
*   NumPy
*   Matplotlib
*   Seaborn
*   NLTK
*   Scikit-learn
*   WordCloud

## How to Run

1.  Ensure you have the necessary libraries installed (`pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud`).
2.  Download the `spam.csv` dataset.
3.  Run the Python code in a Jupyter Notebook or Google Colab environment.

## Files

*   `spam.csv`: The dataset containing SMS messages and their labels (spam/non-spam).
*   `model.pkl`: The saved trained `RandomForestClassifier` model.
*   `vectorizer.pkl`: The saved trained `TfidfVectorizer`.
*   Notebook code: The Python code used for the project steps.
