# Multiple choice quiz program - similar to flashcards but with multiple choice questions
import numpy as np

# qa_mc = ["question","a", "b", "c", "d", "answer","category","source"]
qa_mc = [
 ("question","a","b","c","d","answer","category","source"),

 ("Which one of the following is the largest and fastest growing sector for AI-related global investment (2018-2019)?","Facial Recognition","Autonomous driving","Drug, cancer study","Robotic automation","Autonomous driving","AI industry trends","eksamen 2021"),

 ("Which country got the most private investments (for startups) in Artificial Intelligence (in 2018) in terms of per capita?","United States","China","Singapore","Israel","Israel","AI industry trends","eksamen 2021"),

 ("Many people consider Artificial Intelligence as the","Sixth industrial revolution","Fifth industrial revolution","Fourth industrial revolution","Third industrial revolution","Fourth industrial revolution","AI history","eksamen 2021"),

 ("Which of the following is true for General Artificial Intelligence?","Takes knowledge from one domain and transfers it to other domain","Dedicated to assist with or take over specific tasks","Machines which are an order of magnitude as intelligent or more intelligent than humans","Machines that rely on human input","Takes knowledge from one domain and transfers it to other domain","AI concepts","eksamen 2021"),

 ("Chatbots and Voice assistants (Siri, Alexa, Google assistant) are examples of","General AI","Narrow AI","Super AI","None","Narrow AI","AI concepts","eksamen 2021"),

 ("What is a Turing test in Artificial Intelligence?","A method for determining whether or not a computer is capable of thinking like a human being.","A method for determining whether or not a computer is capable of thinking like Super AI","A method for determining whether or not a computer is capable of thinking like General AI","A benchmark for AI speed","A method for determining whether or not a computer is capable of thinking like a human being.","AI fundamentals","eksamen 2021"),

 ("While working with creating Artificial Intelligence applications, in which area do AI programmers spend most of their time","Model deployment","Data processing (cleaning, labeling etc)","A.I programming","Model development","Data processing (cleaning, labeling etc)","Data engineering","eksamen 2021"),

 ("A data point which differs significantly from other observed data points is called","Labeled data","Synthetic data","Outlier","Noise","Outlier","Data processing","eksamen 2021"),

 ("What is the process of manually adding tags or categories to data points called?","Synthetic data generation","Data anonymization","Feature engineering","Data labeling","Data labeling","Data engineering","eksamen 2021"),

 ("The process of using domain knowledge of a data set to create new attributes from existing data points/attributes is called","Feature engineering","Synthetic data generation","Data labeling","Data cleaning","Feature engineering","Data engineering","eksamen 2021"),

 ("Which type of machine learning uses labeled training data with input-output pairs?","Recommender systems","Reinforcement learning","Unsupervised learning","Supervised learning","Supervised learning","Machine learning types","eksamen 2021"),

 ("In a specific kind of machine learning, an agent can learn in an interactive environment by trial and error using feedback from its own actions and experiences. This is","Supervised learning","Unsupervised learning","Recommender systems","Reinforcement learning","Reinforcement learning","Machine learning types","eksamen 2021"),

 ("What kind of algorithm is Logistic regression?","Clustering algorithm","Regression algorithm","Association algorithm","Classification algorithm","Classification algorithm","Machine learning algorithms","eksamen 2021"),

 ("The output of a sigmoid function (for classification algorithms) has a range from","0 to 10","0 to 1","0 to 1000","0 to 100","0 to 1","Mathematical functions","eksamen 2021"),

 ("Suppose that you are given the previous tax information of all individuals and you now have to develop an algorithm which predicts how much tax they will submit next year. Which type of algorithm would you use?","Clustering","Classification","Association","Regression","Regression","Machine learning applications","eksamen 2021"),

 ("What kind of algorithm assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature","Naive Bayes algorithm","Polynomial regression","Linear regression","Logistic regression","Naive Bayes algorithm","Machine learning algorithms","eksamen 2021"),

 ("What is the maximum number of hyperplanes one can use","10 dimensional","n dimensional","2 dimensional","3 dimensional","n dimensional","SVM / geometry","eksamen 2021"),

 ("Suppose you are given a data set of student complaints from OsloMet's customer service center. The data set is labelled. You are now given a task to understand how angry or happy the students are in those complaints. What kind of algorithms would you use?","Regression","Clustering","Classification","Association","Classification","NLP / sentiment analysis","eksamen 2021"),

 ("Suppose you are given a data set of X ray images of Covid patients. The data set is not labelled and you do not have the opportunity to label it. You are now given the task to identify if the patient has covid or not. What kind of algorithm would you use?","Clustering","Classification","Regression","Reinforcement","Clustering","Unsupervised learning","eksamen 2021"),

 ("Suppose you operate a successful eCommerce store. You want to boost your sales and think you can encourage people to buy more based on their previous purchases. What kind of algorithm would you use to show customers what should they buy?","Association","Clustering","Classification","Regression","Association","Recommender systems","eksamen 2021"),

 ("An equation that describes a relationship between two quantities that show a constant rate of change is called","Support vector machine","Linear regression","Naive Bayes","Logistic regression","Linear regression","Regression","eksamen 2021"),

 ("A regression model where the relationship between variables follows a curved line (like y = ax² + bx + c) is an example of","Linear regression","Polynomial regression","Exponential regression","None","Polynomial regression","Regression","eksamen 2021"),

 ("A recommendation system (e.g. used by social media companies) usually belongs to the following category of AI:","Super A.I.","Narrow A.I.","General A.I.","None","Narrow A.I.","AI categories","eksamen 2021"),

 ("Has there been any software which claims to have passed the Turing test?","Yes","No","Maybe","Unknown","No","AI history","eksamen 2021"),

 ("Suppose you are given the task to predict your income for the next year. You need data for the last 15 years and you only have data for the last 5 years. How will you get that missing data?","Data warehousing","Data anonymization","Feature engineering","Synthetic data","Synthetic data","Data handling","eksamen 2021"),

 ("In what kind of algorithms do we need to use data labeling?","Unsupervised learning","Reinforcement learning","Supervised learning","Semi-supervised learning","Supervised learning","Machine learning types","eksamen 2021"),

 ("In Machine learning, Linear Regression falls within the category of:","Unsupervised learning","Recommender systems","Supervised learning","Reinforcement learning","Supervised learning","Machine learning types","eksamen 2021"),

 ("Regression models are used with","Random data","Continuous data","None of the above","Categorical data","Continuous data","Regression","eksamen 2021"),

 ("What is NOT valid for a hyperplane?","They are boundaries that help classify data points","Hyperplanes work with support vector machines","We can only use maximum 2 hyperplanes for any number of features","They separate data linearly","We can only use maximum 2 hyperplanes for any number of features","SVM","eksamen 2021"),

 ("Which statement is true about outliers?","The nature of the problem determines how outliers are used","Outliers should be part of the training data set but not test data","Outliers should be identified and removed from the data set","Outliers should be part of the test data set but not training data","The nature of the problem determines how outliers are used","Data processing","eksamen 2021"),

 ("The correlation between the number of years an employee has worked for a company and the salary of the employee is 0.75. What can be said about employee salary and years worked?","Individuals that have worked for the company the longest have lower salaries","There is no relationship between salary and years worked","Individuals that have worked for the company the longest have higher salaries","The majority of employees have been with the company a long time","Individuals that have worked for the company the longest have higher salaries","Statistics","eksamen 2021"),

 ("What is TRUE for a machine learning algorithm?","It is harder to train the first 90% than the remaining 10%","None of the above","It is harder to train the remaining last 10% than the first 90%","Training complexity is uniform","It is harder to train the remaining last 10% than the first 90%","Machine learning","eksamen 2021"),

 ("'You may also like' or 'recommended for you' kind of applications (used primarily in Amazon, Facebook etc) can be implemented by using algorithms such as","Neural network algorithms","Apriori algorithm","K-Means algorithm","Decision tree","Apriori algorithm","Recommender systems","eksamen 2021"),

 ("What kind of problem does this statement highlight in your data: Most facial recognition systems today use a higher proportion of white faces as training data (study by IBM in 2019)","Clustered data","Data Bias","Unlabeled data","None of the above","Data Bias","Ethics / data bias","eksamen 2021"),

 ("If the software follows a logical series of steps to reach a conclusion, is easy to explain and the programmer has complete control over the code, then what kind of programming is it?","Conventional programming","Artificial Intelligence programming","Machine learning","Neural networks","Conventional programming","Programming paradigms","eksamen 2021"),

 ("The major reason behind the increased use of Artificial Intelligence today is due to","Powerful processors","Increased connectivity between devices and Cloud computing","Powerful processors","All of the choices","All of the choices","AI trends","eksamen 2021"),

 ("What is the preferred way to work with an A.I. algorithm?","Identify the problem -> prepare data -> choose algorithms -> train the algorithm -> run the algorithm","Identify the problem -> choose algorithms -> run the algorithm -> prepare data -> train the algorithm -> export data to algorithms","Identify the problem -> choose algorithms -> train the algorithm -> run the algorithm -> prepare data -> export data to algorithms","All of the above","Identify the problem -> prepare data -> choose algorithms -> train the algorithm -> run the algorithm","AI workflow","eksamen 2021"),

 ("What is a sigmoid function?","A function that creates a linear relationship","A mathematical function that produces an S-shaped curve used in ML","A function only used in statistics","A function that always outputs zero or one","A mathematical function that produces an S-shaped curve used in ML","GLM","flashcards"),

 ("What is the difference between a generative and discriminative model?","No difference, they are the same","Discriminative learns joint probability, generative learns conditional","Generative learns joint probability, discriminative learns conditional","Both learn only conditional probability","Generative learns joint probability, discriminative learns conditional","ML","flashcards"),



 ("What is gradient descent?","A data preprocessing technique","An optimization algorithm to minimize cost functions","A type of neural network","A feature selection method","An optimization algorithm to minimize cost functions","ML","flashcards"),

 ("What is a decision boundary?","The edge of a dataset","A surface that separates different classes","A type of cost function","A data validation technique","A surface that separates different classes","ML","flashcards"),

 ("What does 96 percent accuracy mean?","The model is 96 percent confident","Model correctly predicted 96 percent of output labels","Training took 96 percent of expected time","96 percent of features were used","Model correctly predicted 96 percent of output labels","ML","flashcards"),

 ("What is the logit function used for in logistic regression?","To increase model complexity","To transform probabilities to linear combinations","To validate input data","To reduce computational cost","To transform probabilities to linear combinations","GLM","flashcards"),

 ("What is Naive Bayes?","A complex neural network","A probabilistic classifier using Bayes' theorem with independence assumptions","A regression algorithm","A clustering method","A probabilistic classifier using Bayes' theorem with independence assumptions","ML","flashcards"),

 ("What is Naive Bayes particularly good at?","Image recognition","Spam detection and text sentiment analysis","Weather prediction","Stock price prediction","Spam detection and text sentiment analysis","ML","flashcards"),

 ("What is Deep Learning?","Basic machine learning","Subset of ML using neural networks with many layers","Only for image processing","A type of database","Subset of ML using neural networks with many layers","ML","flashcards"),

 ("What is Support Vector Machine (SVM)?","Unsupervised clustering algorithm","Supervised algorithm finding optimal hyperplane for classification","Only for text processing","A data preprocessing tool","Supervised algorithm finding optimal hyperplane for classification","ML","flashcards"),

 ("When is SVM typically used?","Large datasets only","Small datasets requiring high accuracy","Only for regression","Real-time applications","Small datasets requiring high accuracy","ML","flashcards"),

 ("What is K-means?","Supervised learning algorithm","Unsupervised clustering algorithm partitioning data into K clusters","Classification algorithm","Regression technique","Unsupervised clustering algorithm partitioning data into K clusters","ML","flashcards"),

 ("What is the purpose of feature scaling?","To reduce dataset size","To normalize the range of features for better model performance","To add more features","To remove outliers","To normalize the range of features for better model performance","ML","flashcards"),

 ("What is the curse of dimensionality?","Having too few features","Problems arising when analyzing high-dimensional data","Having too much data","Network connectivity issues","Problems arising when analyzing high-dimensional data","ML","flashcards"),

 ("What is Lasso regression?","Basic linear regression","Linear regression with L1 regularization","Non-linear regression","Clustering algorithm","Linear regression with L1 regularization","ML","flashcards"),

 ("Adding a new column based on data available is considered creating a","Feature","Label","Cost Function","Function","Feature","Data engineering","Lab"),

 ("What is Accuracy in ML","Percentage of correctly predicted instances out of the total instances","TP / (TP + FP)","TP / (TP + FN)","TN / (TN + FP)","Percentage of correctly predicted instances out of the total instances","ML evaluation","Lab"),

 ("What is Precision in ML","TP + TN / Total","TP / (TP + FP)","TP / (TP + FN)","TN / (TN + FP)","TP / (TP + FP)","ML evaluation","Lab"),

 ("What is Recall in ML","TP + TN / Total","TP / (TP + FP)","TP / (TP + FN)","TN / (TN + FP)","TP / (TP + FN)","ML evaluation","Lab"),

 ("What is F1 Score in ML","2 * (Precision * Recall) / (Precision + Recall)","(Precision * Recall) / (Precision + Recall)","Precision / (Precision + Recall)","2 * (Precision + Recall) / (Precision * Recall)","2 * (Precision * Recall) / (Precision + Recall)","ML evaluation","Lab"),

 ("What is correct about Log Loss?","It is used for regression","It is used for classification","It is not used in ML","It is a type of regularization","It is used for classification","ML evaluation","Lab"),

 ("What does Log Loss represent?","How close the predicted probabilities are to the actual class labels","The accuracy of the model","The precision of the model","The recall of the model","How close the predicted probabilities are to the actual class labels","ML evaluation","Lab"),

 ("What is the ideal value for Log Loss?","0","1","100","-1","0","ML evaluation","Lab"),

 ("What is a ROC curve?","A plot of true positive rate vs false positive rate","A plot of precision vs recall","A plot of accuracy vs error rate","A plot of loss vs epochs","A plot of true positive rate vs false positive rate","ML evaluation","Lab"),

 ("What does the area under the ROC curve (AUC) represent?","The model's ability to distinguish between classes","The model's accuracy","The model's precision","The model's recall","The model's ability to distinguish between classes","ML evaluation","Lab"),

 ("Is there a trade-off between precision and recall","Yes","No","Sometimes","Never","Yes","ML evaluation","Lab"),

 ("What is the trade-off between precision and recall?","Increasing one decreases the other","They are independent","Increasing one increases the other","They are always equal","Increasing one decreases the other","ML evaluation","Lab"),

 ("What is regarded as the first ai software","Watson","Deep Blue","Logic Theorist","AlphaGo","Logic Theorist","AI history","DIKU 002"),

 ("What is the first chat bot and who invented it.","Siri by Apple","Alexa by Amazon","Cortana by Microsoft","ELIZA by Joseph Weizenbaum","ELIZA by Joseph Weizenbaum","AI history","DIKU 002"),

 ("When was the AI winter","1970s","1980s","1990s","Both 1970 and 1990","Both 1970 and 1990","AI history","DIKU 002"),

 ("What happened in the 1980 revival of AI","The video game industry started using AI as a test bed, Japan announces a 850 millions investment in AI, first autonomous vehicle using a neural network, focus switched to narrow ai","Sweden announced a $600 million investment in computer research, AI was used in medical imaging for the first time, and companies began developing AI-powered tractors for agriculture.","Germany launched a national robotics initiative, new AI programs were built to compose music, and researchers focused on creating fully general intelligence systems.","France funded a major AI art project, universities began using AI to automate grading, and the first AI-operated subway system was tested in London. In the United States, tech companies like IBM and AT&T launched large-scale AI labs focused on creating expert systems for business automation and defense research.","The video game industry started using AI as a test bed, Japan announces a 850 millions investment in AI, first autonomous vehicle using a neural network, focus switched to narrow ai","AI history","DIKU 002"),

 ("What is a key ethical requirement for Artificial Intelligence developed and used in Norway?","It should improve efficiency and reduce human labor needs","It should respect human rights and democracy","It should align with market and innovation priorities","It should adapt to European data-sharing frameworks","It should respect human rights and democracy","AI act","DIKU 002"),
 
 ("Which description best matches Artificial Intelligence?","Produces insights based on data, is commonly “one-off,” and usually takes the form of a report or presentation.","Automates tasks or predicts future events based on data, is commonly used “live,” and often takes the form of software.","Focuses on collecting, cleaning, and storing data for later analysis, and mainly supports database infrastructure.","Provides statistical summaries to guide strategic planning, often completed once per project and presented to management.","Automates tasks or predicts future events based on data, is commonly used “live,” and often takes the form of software.","AI","History of AI document"),
 
 ("Which description best matches Data Science?","Produces insights based on data, is commonly \"one-off,\" and usually takes the form of a report or presentation.","Builds neural networks that run continuously and make decisions without human input.","Automates tasks or predicts future events based on data, is commonly used \"live,\" and often takes the form of software.","Focuses on developing digital interfaces that gather user feedback for automated systems.","Automates tasks or predicts future events based on data, is commonly used \"live,\" and often takes the form of software.","AI","History of AI document"),

 ("what is a holdout set in machine learning?", "A portion of the dataset set aside for final model evaluation","A subset of data used for training the model","Data used for hyperparameter tuning","Data used for feature selection","A portion of the dataset set aside for final model evaluation","ML concepts","statistisk analyse mamo2100"),

 ("numpy is used for?", "Data manipulation and numerical computations","Building neural networks","Creating visualizations","Data storage","Data manipulation and numerical computations","Python libraries","shiberas"),

 ("pandas is used for?", "Data manipulation and analysis","Building machine learning models","Creating web applications","Data visualization","Data manipulation and analysis","Python libraries","shiberas"),

 ("the difference between numpy and pandas is?", "numpy is for numerical computations, pandas is for data manipulation and analysis","Both are used for numerical computations","Both are used for data visualization","Both are used for building machine learning models","numpy is for numerical computations, pandas is for data manipulation and analysis","Python libraries","shiberas"),

 ("scikit-learn is a machine learning library in python, and it is used for?", "Building and training machine learning models","Data manipulation and analysis","Creating visualizations","Statistical modeling","Building and training machine learning models","Python libraries","shiberas"),

 ("matplotlib is used for?", "Creating visualizations and plots","Building machine learning models","Data manipulation and analysis","Numerical computations","Creating visualizations and plots","Python libraries","shiberas"),

 ("Hvilken kodelinje er riktig for å utføre regresjon", "model = LinearRegression().fit(X, y)","model = LogisticRegression().fit(X, y)","model = KMeans().fit(X)","model = SVC().fit(X, y)","model = LinearRegression().fit(X, y)","GLM regresjon","shiberas"),

 ("What is the main purpose of a single-number evaluation metric?","To make code run faster","To help quickly compare different models and choose the best one","To reduce the size of the dataset","To eliminate the need for human judgment","To help quickly compare different models and choose the best one","ML strategy","Machine Learning Yearning"),

 ("What should you do if you have multiple metrics you care about?","Pick one as the optimizing metric and others as satisficing metrics","Ignore all but one metric","Average all metrics together","Only use accuracy","Pick one as the optimizing metric and others as satisficing metrics","ML strategy","Machine Learning Yearning"),

 ("What is the recommended split ratio for dev/test sets in modern ML with large datasets?","50/50 dev/test","70/30 dev/test","98/1/1 train/dev/test","60/20/20 train/dev/test","98/1/1 train/dev/test","ML strategy","Machine Learning Yearning"),

 ("What is the key principle about dev and test set distribution?","They should come from different distributions to test generalization","They should come from the same distribution as what you want to do well on","Dev set should be harder than test set","Test set should be from training data","They should come from the same distribution as what you want to do well on","ML strategy","Machine Learning Yearning"),

 ("What size should your dev and test sets be?","As large as possible","Large enough to give high confidence in the overall performance of your system","Exactly 30% of total data","At least 10,000 examples each","Large enough to give high confidence in the overall performance of your system","ML strategy","Machine Learning Yearning"),

 ("When should you change dev/test sets or metrics?","Never, keep them fixed","When your metric is no longer measuring what is most important to you","Only after deploying to production","Every week","When your metric is no longer measuring what is most important to you","ML strategy","Machine Learning Yearning"),

 ("What is the first thing to setup before starting an ML project?","Build the neural network","Set up dev/test sets and metrics","Collect more data","Deploy to production","Set up dev/test sets and metrics","ML strategy","Machine Learning Yearning"),

 ("What does 'avoidable bias' refer to?","The difference between training error and dev error","The difference between training error and human-level performance","Any bias in the training data","The difference between dev error and test error","The difference between training error and human-level performance","ML strategy","Machine Learning Yearning"),

 ("What does 'variance' indicate in ML model performance?","The model is not fitting the training set well","The model is overfitting - doing well on training but not on dev/test","The data is biased","Human-level performance is low","The model is overfitting - doing well on training but not on dev/test","ML strategy","Machine Learning Yearning"),

 ("If training error is 15 percent and dev error is 16 percent, and human-level error is 0 percent, what should you focus on?","Reducing variance","Reducing bias","Getting more data","Changing the metric","Reducing bias","ML strategy","Machine Learning Yearning"),

 ("If training error is 1 percent and dev error is 11 percent, and human-level error is 0 percent, what should you focus on?","Reducing bias","Reducing variance","Human-level error is too low","The model is perfect","Reducing variance","ML strategy","Machine Learning Yearning"),

 ("What is the most reliable way to reduce bias in your model?","Get more training data","Train a bigger model","Add regularization","Use a smaller learning rate","Train a bigger model","ML strategy","Machine Learning Yearning"),

 ("What is the most reliable way to reduce variance in your model?","Train a bigger model","Get more training data or add regularization","Use a higher learning rate","Remove features","Get more training data or add regularization","ML strategy","Machine Learning Yearning"),

 ("What is 'Bayes error' or 'Bayes optimal error'?","The error rate of the best possible function","The error rate of Bayes' theorem","Always 0%","The average human error rate","The error rate of the best possible function","ML theory","Machine Learning Yearning"),

 ("Why is human-level performance often used as a proxy for Bayes error?","Because humans are always optimal","Because it's easy to calculate","Because for tasks humans are good at, human-level performance is close to Bayes error","Because Bayes error doesn't exist","Because for tasks humans are good at, human-level performance is close to Bayes error","ML theory","Machine Learning Yearning"),

 ("When should you stop trying to reduce bias?","Never","When training error reaches human-level performance","When dev error is 0%","After 100 epochs","When training error reaches human-level performance","ML strategy","Machine Learning Yearning"),

 ("What is error analysis?","Running your model on examples it got wrong and analyzing the patterns","Deleting wrong predictions","Calculating error rates","Testing on production data","Running your model on examples it got wrong and analyzing the patterns","ML strategy","Machine Learning Yearning"),

 ("During error analysis, what should you do with misclassified examples?","Delete them from the dataset immediately","Manually examine them and categorize the types of errors","Ignore them and continue training","Retrain the model immediately without analysis","Manually examine them and categorize the types of errors","ML strategy","Machine Learning Yearning"),

 ("If you find that 5 percent of dev set errors are due to a particular category, is it worth fixing?","Yes, always fix every category","Maybe not - focus on categories that account for larger error percentages","Yes, fix all errors regardless of size","No, 5 percent is too small to matter at all","Maybe not - focus on categories that account for larger error percentages","ML strategy","Machine Learning Yearning"),

 ("What should you do if your dev and test sets come from different distributions?","Use them anyway without changes","Make sure they come from the same distribution as your target application","Always use random splits for both sets","Ignore the difference completely","Make sure they come from the same distribution as your target application","ML strategy","Machine Learning Yearning"),

 ("What is a 'training-dev set'?","Another name for the validation set","Data from the same distribution as training, but not used for training - helps detect variance","Data for developers only to use","The same as the test set completely","Data from the same distribution as training, but not used for training - helps detect variance","ML strategy","Machine Learning Yearning"),

 ("If training error is low, training-dev error is low, but dev error is high, what's the problem?","Bias problem with the model","Variance problem with the model","Data mismatch problem between sets","The model is perfect already","Data mismatch problem between sets","ML strategy","Machine Learning Yearning"),

 ("What should you do when you have a data mismatch problem?","Get more training data immediately","Try to understand the difference between training and dev sets and add similar data to training","Change the algorithm completely","Use a smaller model instead","Try to understand the difference between training and dev sets and add similar data to training","ML strategy","Machine Learning Yearning"),

 ("What is transfer learning?","Transferring data between multiple databases","Using knowledge from one task to help with another task","Moving models between different servers","Sharing code between different projects","Using knowledge from one task to help with another task","ML strategy","Machine Learning Yearning"),

 ("When does transfer learning make sense?","Always in every situation","When task A and B have the same input, you have more data for A than B, and low-level features help both","Only for image tasks specifically","Never use it at all","When task A and B have the same input, you have more data for A than B, and low-level features help both","ML strategy","Machine Learning Yearning"),

 ("What is multi-task learning?","Training multiple separate models independently","Training one model to perform multiple tasks simultaneously","Using multiple GPUs for training","Training on multiple datasets sequentially","Training one model to perform multiple tasks simultaneously","ML strategy","Machine Learning Yearning"),

 ("When does multi-task learning make sense?","Always in every scenario","When tasks share lower-level features, similar amount of data for each task, and you can train a big enough network","Only for NLP tasks specifically","Never use it in practice","When tasks share lower-level features, similar amount of data for each task, and you can train a big enough network","ML strategy","Machine Learning Yearning"),

 ("What is end-to-end deep learning?","Training only the final layer","Replacing a multi-step pipeline with a single neural network","Training from scratch every time","Using multiple models in sequence","Replacing a multi-step pipeline with a single neural network","ML strategy","Machine Learning Yearning"),

 ("What is the main advantage of end-to-end learning?","It's always faster than alternatives","It lets the model learn the optimal representation without hand-designed components","It requires less data overall","It's easier to debug issues","It lets the model learn the optimal representation without hand-designed components","ML strategy","Machine Learning Yearning"),

 ("What is the main disadvantage of end-to-end learning?","It's too slow for production","It requires a very large amount of data","It always overfits the training set","It can't work with images","It requires a very large amount of data","ML strategy","Machine Learning Yearning"),

 ("What should guide your decision to use end-to-end learning?","Always use it regardless","Whether you have enough data to learn the complexity of the mapping","Use it only for vision tasks","Never use it at all","Whether you have enough data to learn the complexity of the mapping","ML strategy","Machine Learning Yearning"),

 ("Hvilken kodelinje er riktig for å utføre klassifisering", "model = LogisticRegression().fit(X, y)","model = LinearRegression().fit(X, y)","model = KMeans().fit(X)","model = SVC().fit(X, y)","model = LogisticRegression().fit(X, y)","GLM regresjon","shiberas"),

 ("Hvilken kodelinje er riktig for å utføre klynging", "model = KMeans().fit(X)","model = LogisticRegression().fit(X, y)","model = LinearRegression().fit(X, y)","model = SVC().fit(X, y)","model = KMeans().fit(X)","GLM regresjon","shiberas"),

 ("Hvilken kodelinje er riktig for å finne den mest forekommende verdien i en kolonne i en pandas dataframe", "Grc_df['Outlet_Size'].value_counts().idxmax()","Grc_df['Outlet_Size'].max()","Grc_df['Outlet_Size'].mean()","Grc_df['Outlet_Size'].min()","Grc_df['Outlet_Size'].value_counts().idxmax()","Pandas dataframe","shiberas"),

 ("Hvilken kodelinje er riktig for å dele en kolonne i en pandas dataframe i 10 like store binner", "Grc_Concat_df['Item_Weight_Binned'] = pd.cut(Grc_Concat_df['Item_Weight'], bins=10)","Grc_Concat_df['Item_Weight_Binned'] = pd.qcut(Grc_Concat_df['Item_Weight'], bins=10)","Grc_Concat_df['Item_Weight_Binned'] = pd.split(Grc_Concat_df['Item_Weight'], bins=10)","Grc_Concat_df['Item_Weight_Binned'] = pd.bucket(Grc_Concat_df['Item_Weight'], bins=10)","Grc_Concat_df['Item_Weight_Binned'] = pd.cut(Grc_Concat_df['Item_Weight'], bins=10)","Pandas dataframe","shiberas"),

 ("Hvilken kodelinje er riktig for å fylle inn manglende verdier i en kolonne med medianen av den kolonnen", "df['Age'] = df['Age'].fillna(df['Age'].median())","df['Age'] = df['Age'].fillna(df['Age'].mean())","df['Age'] = df['Age'].fillna(df['Age'].mode())","df['Age'] = df['Age'].fillna(df['Age'].min())","df['Age'] = df['Age'].fillna(df['Age'].median())","Pandas dataframe","shiberas"),

 ("According to the AI Index Report 2025, which sector saw the highest AI private investment in 2024?","Healthcare","Data management and processing","Transportation","Retail","Data management and processing","AI industry trends","AI Index Report 2025"),

 ("What was the approximate total global AI private investment in 2024 according to the AI Index Report?","$50 billion","$97.2 billion","$150 billion","$25 billion","$97.2 billion","AI industry trends","AI Index Report 2025"),

 ("Which country led in AI private investment in 2024?","China","United Kingdom","United States","Germany","United States","AI industry trends","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what percentage of companies reported adopting at least one AI capability?","25 percent","42 percent","55 percent","72 percent","55 percent","AI adoption","AI Index Report 2025"),

 ("What is the main reason companies cited for not adopting AI according to the AI Index Report?","Too expensive","Lack of skilled personnel","No clear business case","Regulatory concerns","Lack of skilled personnel","AI adoption","AI Index Report 2025"),

 ("According to the AI Index Report 2025, which AI application area saw the most significant growth in 2024?","Robotics","Computer vision","Natural language processing","Autonomous vehicles","Natural language processing","AI applications","AI Index Report 2025"),

 ("What trend did the AI Index Report 2025 identify regarding AI model training costs?","Decreasing significantly","Increasing exponentially","Remaining stable","Becoming unpredictable","Increasing exponentially","AI development","AI Index Report 2025"),

 ("According to the AI Index Report, what is the primary ethical concern about AI in 2024?","Cost of deployment infrastructure","Bias and fairness in algorithms","Speed of processing data","Energy consumption levels","Bias and fairness in algorithms","AI ethics","AI Index Report 2025"),

 ("Which region showed the fastest growth in AI research publications according to the AI Index Report 2025?","North America region","Europe region","Asia region","South America region","Asia region","AI research","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what percentage of AI PhD graduates in the US go into industry rather than academia?","35 percent of graduates","50 percent of graduates","65 percent of graduates","80 percent of graduates","65 percent of graduates","AI workforce","AI Index Report 2025"),

 ("What does the AI Index Report 2025 say about AI's impact on job displacement?","Minimal impact observed so far","Significant displacement in manufacturing and routine cognitive tasks","Only affects low-skill jobs primarily","No measurable impact yet detected","Significant displacement in manufacturing and routine cognitive tasks","AI impact","AI Index Report 2025"),

 ("According to the AI Index Report, which application of AI in transportation saw the most investment in 2024?","Traffic management systems","Autonomous vehicles technology","Route optimization algorithms","Predictive maintenance tools","Autonomous vehicles technology","AI applications","AI Index Report 2025"),

 ("What trend did the AI Index Report 2025 identify in AI regulatory frameworks globally?","Decreasing regulation overall","Increasing fragmentation and country-specific approaches","Complete harmonization worldwide","No significant changes observed","Increasing fragmentation and country-specific approaches","AI policy","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what is the estimated growth rate of the global AI market from 2024 to 2030?","10 percent annually","20 percent annually","37 percent annually","50 percent annually","37 percent annually","AI industry trends","AI Index Report 2025"),

 ("Which AI technique showed the most improvement in benchmark performance in 2024 according to the report?","Reinforcement learning algorithms","Large language models","Computer vision systems","Speech recognition systems","Large language models","AI development","AI Index Report 2025"),

 ("According to the AI Index Report, what percentage of AI systems deployed in production experienced some form of failure or incident?","15 percent of systems","28 percent of systems","45 percent of systems","60 percent of systems","28 percent of systems","AI reliability","AI Index Report 2025"),

 ("What does the AI Index Report 2025 identify as the biggest barrier to AI adoption in developing countries?","Lack of interest overall","Infrastructure and connectivity limitations","Cultural resistance to technology","Too many regulations","Infrastructure and connectivity limitations","AI adoption","AI Index Report 2025"),

 ("According to the report, which industry sector has the highest AI adoption rate?","Healthcare industry","Finance and insurance industry","Retail industry","Manufacturing industry","Finance and insurance industry","AI adoption","AI Index Report 2025"),

 ("What trend did the AI Index Report identify regarding open-source AI models in 2024?","Declining in popularity overall","Significant increase in development and adoption","Remaining stable without change","Being replaced by proprietary models","Significant increase in development and adoption","AI development","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what is the primary driver of AI innovation?","Government funding programs","Academic research institutions","Private sector investment","International collaboration efforts","Private sector investment","AI development","AI Index Report 2025"),

 ("What does the report say about AI's energy consumption in 2024?","Decreasing due to efficiency gains","Growing concern due to training large models","Not significant enough to measure","Completely offset by renewable energy","Growing concern due to training large models","AI sustainability","AI Index Report 2025"),

 ("According to the AI Index Report, which country has the most comprehensive AI strategy?","United States of America","China","Singapore","European Union","China","AI policy","AI Index Report 2025"),

 ("What percentage of Fortune 500 companies have a dedicated AI strategy according to the AI Index Report 2025?","40 percent of companies","60 percent of companies","75 percent of companies","90 percent of companies","75 percent of companies","AI adoption","AI Index Report 2025"),

 ("According to the report, what is the average time to deploy an AI model to production in 2024?","1-3 months on average","4-6 months on average","7-12 months on average","Over 1 year on average","4-6 months on average","AI development","AI Index Report 2025"),

 ("What does the AI Index Report identify as the most promising emerging AI application?","AI for climate change mitigation","AI for drug discovery","AI for education technology","AI for cybersecurity","AI for drug discovery","AI applications","AI Index Report 2025"),

 ("What are the three main types of machine learning?","Supervised, Unsupervised, Reinforcement","Classification, Regression, Clustering","Neural networks, Decision trees, SVM","Deep learning, Shallow learning, Transfer learning","Supervised, Unsupervised, Reinforcement","ML fundamentals","studocu"),

 ("In supervised learning, what is required for training?","Only input data is needed","Only output labels are needed","Both labeled input and output pairs","No data is needed at all","Both labeled input and output pairs","ML fundamentals","studocu"),

 ("What is the main goal of unsupervised learning?","To predict future values accurately","To find patterns and structure in unlabeled data","To maximize rewards over time","To classify data into known categories","To find patterns and structure in unlabeled data","ML fundamentals","studocu"),

 ("What is the key characteristic of reinforcement learning?","Learning from labeled examples","Finding clusters in data","Learning through trial and error with rewards and penalties","Reducing dimensionality","Learning through trial and error with rewards and penalties","ML fundamentals","studocu"),



 ("What is the purpose of a validation set?","To train the model initially","To tune hyperparameters and prevent overfitting","To replace the test set","To label data manually","To tune hyperparameters and prevent overfitting","ML concepts","studocu"),





 ("What is a confusion matrix used for?","To confuse the model during training","To evaluate classification model performance by showing true/false positives and negatives","To visualize training loss over time","To select features for modeling","To evaluate classification model performance by showing true/false positives and negatives","ML evaluation","studocu"),

 ("What does True Positive (TP) mean in a confusion matrix?","Model correctly predicted negative class","Model incorrectly predicted positive class","Model correctly predicted positive class","Model incorrectly predicted negative class","Model correctly predicted positive class","ML evaluation","studocu"),

 ("What does False Positive (FP) mean?","Correctly predicted positive class","Incorrectly predicted positive (Type I error)","Correctly predicted negative class","Incorrectly predicted negative class","Incorrectly predicted positive (Type I error)","ML evaluation","studocu"),

 ("What does False Negative (FN) mean?","Correctly predicted negative class","Incorrectly predicted negative (Type II error)","Correctly predicted positive class","Incorrectly predicted positive class","Incorrectly predicted negative (Type II error)","ML evaluation","studocu"),

 ("What is gradient descent?","A clustering algorithm for data","An optimization algorithm that iteratively adjusts parameters to minimize loss","A classification method for labels","A data preprocessing technique","An optimization algorithm that iteratively adjusts parameters to minimize loss","ML algorithms","studocu"),

 ("What is a learning rate in gradient descent?","The speed of data loading","The step size for parameter updates during optimization","The accuracy of the model","The number of epochs","The step size for parameter updates during optimization","ML algorithms","studocu"),

 ("What happens if the learning rate is too high?","Training is too slow overall","The algorithm may overshoot the minimum and fail to converge","The model becomes too accurate","Nothing significant happens","The algorithm may overshoot the minimum and fail to converge","ML algorithms","studocu"),

 ("What happens if the learning rate is too low?","Perfect convergence is achieved","Training is very slow and may get stuck in local minima","Model overfits immediately","No training occurs at all","Training is very slow and may get stuck in local minima","ML algorithms","studocu"),

 ("What is batch gradient descent?","Uses one sample at a time","Uses the entire dataset to compute gradient","Uses random samples only","Uses only test data","Uses the entire dataset to compute gradient","ML algorithms","studocu"),

 ("What is stochastic gradient descent (SGD)?","Uses entire dataset at once","Uses one training example at a time to update parameters","Uses validation set only","Uses multiple datasets","Uses one training example at a time to update parameters","ML algorithms","studocu"),

 ("What is mini-batch gradient descent?","Uses entire dataset at once","Uses small random batches of training data","Uses one sample only","Uses test data only","Uses small random batches of training data","ML algorithms","studocu"),

 ("What is the purpose of an activation function in neural networks?","To slow down training processes","To introduce non-linearity so the network can learn complex patterns","To reduce overfitting problems","To normalize inputs","To introduce non-linearity so the network can learn complex patterns","Neural networks","studocu"),

 ("What is the most common activation function for hidden layers?","Sigmoid function","ReLU (Rectified Linear Unit)","Linear function","Softmax function","ReLU (Rectified Linear Unit)","Neural networks","studocu"),

 ("What activation function is typically used for binary classification output?","ReLU function","Sigmoid function","Tanh function","Linear function","Sigmoid function","Neural networks","studocu"),

 ("What activation function is used for multi-class classification output?","Sigmoid function","ReLU function","Softmax function","Tanh function","Softmax function","Neural networks","studocu"),

 ("What is backpropagation?","Forward pass through network","Algorithm for computing gradients and updating weights by propagating errors backward","Data preprocessing step","Model evaluation metric","Algorithm for computing gradients and updating weights by propagating errors backward","Neural networks","studocu"),

 ("What is an epoch in neural network training?","One forward pass only","One complete pass through the entire training dataset","One weight update only","One batch processed","One complete pass through the entire training dataset","Neural networks","studocu"),

 ("What is dropout in neural networks?","Removing bad data points","Regularization technique that randomly drops neurons during training to prevent overfitting","Stopping training early","Removing features from model","Regularization technique that randomly drops neurons during training to prevent overfitting","Neural networks","studocu"),

 ("What is the vanishing gradient problem?","Gradients explode during training","Gradients become very small in deep networks, making training difficult","Gradients disappear completely","Gradients become negative","Gradients become very small in deep networks, making training difficult","Neural networks","studocu"),

 ("What is the main difference between AI and Machine Learning?","They are the same thing essentially","AI is the broader concept, ML is a subset focused on learning from data","ML is broader than AI overall","AI only works with images","AI is the broader concept, ML is a subset focused on learning from data","AI concepts","eksamen 2022"),

 ("What is the purpose of data normalization in machine learning?","To remove outliers from data","To scale features to similar ranges for better model performance","To add more data","To label the data manually","To scale features to similar ranges for better model performance","Data preprocessing","eksamen 2022"),

 ("Which metric would be most appropriate for imbalanced classification problems?","Accuracy alone","F1-score or precision-recall","Mean squared error","R-squared coefficient","F1-score or precision-recall","ML evaluation","eksamen 2022"),

 ("What is the curse of dimensionality?","Having too much data available","Performance degradation as the number of features increases","Having too few samples","Model training is too fast","Performance degradation as the number of features increases","ML concepts","eksamen 2022"),

 ("What is the purpose of PCA (Principal Component Analysis)?","To increase dimensions in data","To reduce dimensionality while preserving variance","To classify data into categories","To cluster data into groups","To reduce dimensionality while preserving variance","ML techniques","eksamen 2022"),

 ("What does it mean when a model has high training accuracy but low test accuracy?","The model is underfitting the data","The model is overfitting the data","The model is perfect overall","The data is bad quality","The model is overfitting the data","ML concepts","eksamen 2022"),

 ("What are the three main components of AI according to the introduction lecture?","Hardware, Software, Data","Reasoning, Learning, Perception","Input, Processing, Output","Training, Testing, Deployment","Reasoning, Learning, Perception","AI fundamentals","Lecture Introduction-DAVE3625"),

 ("What is the main limitation of rule-based AI systems?","They require expensive hardware and infrastructure for deployment","They cannot handle uncertainty and require explicit programming for all scenarios","They are too slow for real-time processing applications","They use too much memory for practical applications","They cannot handle uncertainty and require explicit programming for all scenarios","AI limitations","Lecture Limitations-with-AI"),

 ("What is the main difference between strong AI and weak AI?","Strong AI has more accurate predictions than weak AI","Strong AI has general intelligence like humans, weak AI is task-specific","Strong AI processes information faster than weak AI","Strong AI uses larger datasets than weak AI","Strong AI has general intelligence like humans, weak AI is task-specific","AI concepts","Lecture Introduction-DAVE3625"),

 ("What is the primary goal of supervised learning?","To find hidden patterns in unlabeled data","To learn a mapping from inputs to outputs using labeled examples","To group similar items into clusters","To maximize rewards through trial and error","To learn a mapping from inputs to outputs using labeled examples","ML fundamentals","Lecture MachineLearning"),

 ("In the context of machine learning, what is a feature?","The output variable that we're trying to predict","An individual measurable property or characteristic of the data","The algorithm used to train the model","The training process that optimizes the model","An individual measurable property or characteristic of the data","ML fundamentals","Lecture MachineLearning"),

 ("What is the purpose of splitting data into training and test sets?","To save storage space by reducing data size","To evaluate model performance on unseen data","To speed up training by using smaller datasets","To reduce overfitting during training phase","To evaluate model performance on unseen data","ML fundamentals","Lecture MachineLearning"),

 ("What does the bias-variance tradeoff refer to?","Speed vs accuracy tradeoff","The balance between underfitting and overfitting","Training time vs testing time","Memory usage vs performance","The balance between underfitting and overfitting","ML theory","Lecture MachineLearning-p2"),

 ("What is the main advantage of decision trees?","They always achieve the most accurate predictions","They are interpretable and easy to understand","They require the least computational resources","They work without any training data required","They are interpretable and easy to understand","ML algorithms","Lecture MachineLearning-p2"),

 ("What is ensemble learning?","Using multiple datasets from different sources","Combining multiple models to improve performance","Training one model multiple times on same data","Using multiple computers to speed up training","Combining multiple models to improve performance","ML techniques","Lecture MachineLearning-p2"),

 ("What is the purpose of the kernel trick in SVM?","To reduce training time for large datasets","To map data to higher dimensions for linear separation","To reduce the number of features in data","To normalize the data before classification","To map data to higher dimensions for linear separation","SVM","Lecture MachineLearning-p2"),

 ("What is a confusion matrix used to evaluate?","Regression models and continuous predictions","Classification model performance and accuracy","Clustering quality and cluster cohesion","Data quality and missing values","Classification model performance and accuracy","ML evaluation","Lecture MachineLearning-p2"),

 ("What is cross-validation primarily used for?","To increase dataset size through replication","To get a more reliable estimate of model performance","To speed up training by using less data","To visualize data patterns and relationships","To get a more reliable estimate of model performance","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the main purpose of regularization in machine learning?","To increase model complexity and flexibility","To prevent overfitting by penalizing complex models","To speed up training process significantly","To reduce data size for faster processing","To prevent overfitting by penalizing complex models","ML techniques","Lecture MachineLearning-p2"),

 ("What is the difference between L1 and L2 regularization?","There is no difference between them","L1 can zero out coefficients (feature selection), L2 shrinks them","L2 can zero out coefficients, L1 shrinks them instead","Both zero out coefficients equally across features","L1 can zero out coefficients (feature selection), L2 shrinks them","ML techniques","Lecture MachineLearning-p2"),

 ("What is the K in K-Nearest Neighbors (KNN)?","The number of features in the dataset","The number of nearest neighbors to consider","The number of classes in classification","The number of iterations to run","The number of nearest neighbors to consider","ML algorithms","Lecture MachineLearning-p3-1"),

 ("What is a hyperparameter in machine learning?","A parameter that is learned during training","A parameter set before training that controls the learning process","The final output of the trained model","A type of activation function for neurons","A parameter set before training that controls the learning process","ML fundamentals","Lecture MachineLearning-p3-1"),

 ("What is grid search used for?","To visualize data in graphical format","To systematically search for optimal hyperparameters","To clean data and remove outliers","To reduce dimensions in feature space","To systematically search for optimal hyperparameters","ML optimization","Lecture MachineLearning-p3-1"),

 ("What is the elbow method used for in K-means clustering?","To find the optimal K value","To measure classification accuracy","To normalize data before clustering","To split data into train/test sets","To find the optimal K value","Clustering","Lecture MachineLearning-p4-unsupervised"),

 ("What is the main goal of dimensionality reduction?","To increase features for better accuracy","To reduce the number of features while preserving important information","To improve accuracy directly without data loss","To speed up data collection and storage","To reduce the number of features while preserving important information","ML techniques","Lecture MachineLearning-p4-unsupervised"),

 ("What is PCA (Principal Component Analysis) primarily used for?","Classification of data into categories","Dimensionality reduction by finding principal components","Clustering similar data points together","Regression analysis of continuous variables","Dimensionality reduction by finding principal components","ML techniques","Lecture MachineLearning-p4-unsupervised"),

 ("In unsupervised learning, what is the main difference from supervised learning?","No computer is used in the process","No labeled data is used for training","No training is needed at all","No testing is needed afterwards","No labeled data is used for training","ML fundamentals","Lecture MachineLearning-p4-unsupervised"),

 ("What is hierarchical clustering?","A single-step clustering algorithm","A method that creates a tree of clusters","A supervised learning classification technique","A type of neural network architecture","A method that creates a tree of clusters","Clustering","Lecture MachineLearning-p4-unsupervised"),

 ("What is the Silhouette score used for?","To measure classification prediction accuracy","To evaluate clustering quality and cohesion","To measure regression prediction error","To select features for model training","To evaluate clustering quality and cohesion","Clustering","Lecture MachineLearning-p4-unsupervised"),

 ("What is anomaly detection used for?","To find typical patterns in data","To identify unusual data points that don't fit normal patterns","To classify data into known categories","To reduce dimensions in feature space","To identify unusual data points that don't fit normal patterns","ML applications","Lecture MachineLearning-p4-unsupervised"),

 ("What does it mean when we say AI has a 'black box' problem?","AI is always wrong in predictions","It's difficult to understand how AI makes decisions","AI is too simple to understand","AI is too expensive to maintain","It's difficult to understand how AI makes decisions","AI limitations","Lecture Limitations-with-AI"),

 ("What is algorithmic bias in AI?","AI being too accurate in predictions","Systematic errors in AI systems due to biased training data or design","AI being too slow in processing","AI using too much memory storage","Systematic errors in AI systems due to biased training data or design","AI ethics","Lecture Limitations-with-AI"),

 ("What is meant by AI explainability or interpretability?","Making AI faster in processing","The ability to understand and explain how an AI system makes decisions","Making AI cheaper to operate","Making AI more accurate in predictions","The ability to understand and explain how an AI system makes decisions","AI concepts","Lecture Limitations-with-AI"),

 ("What is the main concern with AI systems making critical decisions in healthcare or justice?","The cost of implementation","Lack of transparency and potential for bias affecting human lives","The speed of decision making","The storage requirements for data","Lack of transparency and potential for bias affecting human lives","AI ethics","Lecture Limitations-with-AI"),

 ("What is data poisoning in the context of AI security?","Using too much data in training","Deliberately manipulating training data to compromise the model","Deleting data from the database","Encrypting data for security purposes","Deliberately manipulating training data to compromise the model","AI security","Lecture Limitations-with-AI"),

 ("What is adversarial attack in AI?","Training with more data samples","Crafting inputs designed to fool the AI model","Using better hardware for training","Improving the algorithm's efficiency","Crafting inputs designed to fool the AI model","AI security","Lecture Limitations-with-AI"),

 ("What is the main energy concern with large AI models?","They use too little energy","Training large models requires massive computational resources and energy","They only work on batteries","They can't be powered properly","Training large models requires massive computational resources and energy","AI sustainability","Lecture Limitations-with-AI"),

 ("What is transfer learning in machine learning?","Transferring data between different computers","Using knowledge from one task to improve learning on a related task","Moving models between different servers","Translating between different languages","Using knowledge from one task to improve learning on a related task","ML techniques","Lecture MachineLearning-p3-1"),

 ("What is the difference between classification and regression?","There is no difference between them","Classification predicts discrete categories, regression predicts continuous values","Regression predicts categories, classification predicts continuous values","Both predict the same types of outputs","Classification predicts discrete categories, regression predicts continuous values","ML fundamentals","Lecture MachineLearning"),

 ("What is a neural network layer?","A physical component of the computer","A collection of neurons that process inputs together","A type of data format","A training algorithm for optimization","A collection of neurons that process inputs together","Neural networks","Lecture MachineLearning-p3-1"),



 ("What is the vanishing gradient problem in deep learning?","Gradients becoming too large during training","Gradients becoming too small in early layers, preventing learning","Gradients disappearing completely from network","Loss function increasing during training","Gradients becoming too small in early layers, preventing learning","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is meant by model deployment in machine learning?","Training the model on data","Putting the trained model into production for real-world use","Collecting data for the model","Testing the model's accuracy","Putting the trained model into production for real-world use","ML lifecycle","Lecture MachineLearning"),

 ("What is A/B testing in the context of ML deployment?","Testing two different datasets","Comparing two model versions with real users to see which performs better","Testing on A grade vs B grade","Testing accuracy vs bias metrics","Comparing two model versions with real users to see which performs better","ML deployment","Lecture MachineLearning"),

 ("What is model monitoring in production?","Ignoring the model after deployment","Continuously tracking model performance to detect degradation or issues","Only checking once per month","Manual testing every few months","Continuously tracking model performance to detect degradation or issues","ML deployment","Lecture MachineLearning"),

 ("What is concept drift in machine learning?","Model getting better over time","When the statistical properties of the target variable change over time","Model staying the same forever","Training getting faster over time","When the statistical properties of the target variable change over time","ML deployment","Lecture MachineLearning"),



 ("What is the purpose of data augmentation?","To delete unnecessary data","To artificially increase training data by creating modified versions","To compress data for storage","To visualize data patterns","To artificially increase training data by creating modified versions","Data preprocessing","Lecture MachineLearning"),

 ("What is imbalanced data in classification?","Equal class distribution","When one class has significantly more samples than others","All classes are missing","Data is corrupted","When one class has significantly more samples than others","Data issues","Lecture MachineLearning-p2"),

 ("What technique can help with imbalanced datasets?","Ignore the problem completely","Oversampling minority class or undersampling majority class","Delete all data","Use only one class","Oversampling minority class or undersampling majority class","Data preprocessing","Lecture MachineLearning-p2"),

 ("What is SMOTE in machine learning?","A type of neural network","Synthetic Minority Oversampling Technique for handling imbalanced data","A clustering algorithm","A regularization method","Synthetic Minority Oversampling Technique for handling imbalanced data","Data preprocessing","Lecture MachineLearning-p2"),

 ("What is the ROC curve's x-axis and y-axis?","Precision and Recall","False Positive Rate and True Positive Rate","Accuracy and Loss","Bias and Variance","False Positive Rate and True Positive Rate","ML evaluation","Lecture MachineLearning-p2"),

 ("What does an AUC (Area Under Curve) of 0.5 indicate?","Perfect model performance","Random guessing performance","Worst possible performance","Good performance","Random guessing performance","ML evaluation","Lecture MachineLearning-p2"),

 ("What does an AUC close to 1.0 indicate?","Poor model performance","Excellent model performance","Random model performance","Biased model performance","Excellent model performance","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the purpose of max pooling in CNNs?","To increase image size","To reduce spatial dimensions while retaining important features","To add more layers","To train faster","To reduce spatial dimensions while retaining important features","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is a convolutional layer in CNNs?","A fully connected layer","A layer that applies filters to detect features in images","An output layer","A normalization layer","A layer that applies filters to detect features in images","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the main advantage of CNNs for image processing?","They are faster than others","They can automatically learn spatial hierarchies of features","They use less memory","They require less data","They can automatically learn spatial hierarchies of features","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is a recurrent neural network (RNN) primarily used for?","Image classification tasks","Sequential data like text or time series","Clustering algorithms","Dimensionality reduction","Sequential data like text or time series","Neural networks","Lecture MachineLearning-p3-1"),

 ("What problem do LSTMs solve compared to basic RNNs?","They are faster overall","They better handle long-term dependencies in sequences","They use less memory","They are simpler overall","They better handle long-term dependencies in sequences","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of attention mechanism in neural networks?","To make training faster","To allow the model to focus on relevant parts of the input","To reduce parameters","To normalize data","To allow the model to focus on relevant parts of the input","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is a loss function in machine learning?","A function that always returns zero","A function that measures the difference between predictions and actual values","A function that adds features","A function that removes outliers","A function that measures the difference between predictions and actual values","ML fundamentals","Lecture MachineLearning"),

 ("What is mean squared error (MSE) typically used for?","Classification tasks","Regression problems","Clustering tasks","Dimensionality reduction","Regression problems","ML evaluation","Lecture MachineLearning"),

 ("What is cross-entropy loss typically used for?","Regression tasks","Classification problems","Clustering tasks","Data preprocessing","Classification problems","ML evaluation","Lecture MachineLearning-p2"),

 ("What is early stopping in neural network training?","Starting training early","Stopping training when validation performance stops improving","Stopping after one epoch","Never stopping training","Stopping training when validation performance stops improving","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of a learning rate scheduler?","To maintain constant learning rate throughout","To adjust learning rate during training for better convergence","To increase learning rate only","To remove learning rate entirely","To adjust learning rate during training for better convergence","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is batch size in neural network training?","Total dataset size available","Number of samples processed before updating model parameters","Number of epochs to run","Number of layers in network","Number of samples processed before updating model parameters","Neural networks","Lecture MachineLearning-p3-1"),

 ("What happens if batch size is too large?","Training is always better overall","May not fit in memory and may lead to poor generalization","Training is faster only","Model is more accurate overall","May not fit in memory and may lead to poor generalization","Neural networks","Lecture MachineLearning-p3-1"),

 ("What happens if batch size is too small?","Perfect training results","Training becomes noisy and slow","Training is optimal","No training occurs at all","Training becomes noisy and slow","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of a validation set distinct from test set?","No specific purpose","To tune hyperparameters without touching the test set","To replace training set entirely","To reduce data size only","To tune hyperparameters without touching the test set","ML fundamentals","Lecture MachineLearning"),

 ("What is stratified sampling in train/test split?","Random sampling only","Maintaining class proportions in splits","Taking only one class","Taking all data together","Maintaining class proportions in splits","Data preprocessing","Lecture MachineLearning"),

 ("What is the difference between parameters and hyperparameters?","No difference at all","Parameters are learned during training, hyperparameters are set before","Hyperparameters are learned, parameters are set manually","Both are the same thing","Parameters are learned during training, hyperparameters are set before","ML fundamentals","Lecture MachineLearning"),

 ("What is the purpose of momentum in gradient descent?","To slow down training process","To accelerate convergence by accumulating gradients","To increase loss function","To remove features entirely","To accelerate convergence by accumulating gradients","Optimization","Lecture MachineLearning-p2"),

 ("What is Adam optimizer?","A type of neural network architecture","An adaptive learning rate optimization algorithm","A loss function type","A regularization technique method","An adaptive learning rate optimization algorithm","Optimization","Lecture MachineLearning-p3-1"),

 ("What is the main idea behind ensemble methods like Random Forest?","Use one tree model","Combine multiple models to reduce variance and improve accuracy","Use only neural networks","Avoid decision trees entirely","Combine multiple models to reduce variance and improve accuracy","ML algorithms","Lecture MachineLearning-p2"),

 ("What is bagging in ensemble learning?","Training one model only","Training multiple models on different random subsets of data","Using bags for storage only","Removing data points entirely","Training multiple models on different random subsets of data","ML techniques","Lecture MachineLearning-p2"),

 ("What is boosting in ensemble learning?","Random combination of models","Sequentially training models where each focuses on previous errors","Training in parallel only","Using only strong learners","Sequentially training models where each focuses on previous errors","ML techniques","Lecture MachineLearning-p2"),

 ("What is the difference between bagging and boosting?","No difference at all","Bagging trains in parallel, boosting trains sequentially focusing on errors","Boosting trains in parallel, bagging sequentially","Both are identical methods","Bagging trains in parallel, boosting trains sequentially focusing on errors","ML techniques","Lecture MachineLearning-p2"),

 ("What is the purpose of dropout rate in neural networks?","To keep all neurons active","To specify the fraction of neurons to randomly drop during training","To add more layers overall","To reduce epochs needed","To specify the fraction of neurons to randomly drop during training","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is weight initialization in neural networks?","Setting all weights to zero","Setting initial weights before training begins","Final weight values only","Removing weights entirely","Setting initial weights before training begins","Neural networks","Lecture MachineLearning-p3-1"),

 ("Why is proper weight initialization important?","It's not important at all","Poor initialization can lead to vanishing/exploding gradients","It only affects speed","It reduces accuracy overall","Poor initialization can lead to vanishing/exploding gradients","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of the softmax function?","To make training harder overall","To convert outputs to probability distribution for multi-class classification","To normalize inputs only","To reduce dimensions available","To convert outputs to probability distribution for multi-class classification","Neural networks","Lecture MachineLearning-p2"),



 ("What does high precision but low recall indicate?","Model is perfect overall","Model is conservative, misses many positives but rarely wrong when it predicts positive","Model is random predictions","Model always predicts positive class","Model is conservative, misses many positives but rarely wrong when it predicts positive","ML evaluation","Lecture MachineLearning-p2"),

 ("What does high recall but low precision indicate?","Model is perfect overall","Model catches most positives but has many false alarms","Model is conservative in predictions","Model predicts nothing at all","Model catches most positives but has many false alarms","ML evaluation","Lecture MachineLearning-p2"),



 ("What is a true negative (TN) in classification?","Correctly predicted positive class","Correctly predicted negative class","Incorrectly predicted positive class","Incorrectly predicted negative class","Correctly predicted negative class","ML evaluation","Lecture MachineLearning-p2"),

 ("What is specificity in classification?","True positive rate overall","True Negative divided by all actual negatives","True Positive divided by actual positives","False positive rate overall","True Negative divided by all actual negatives","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the purpose of standardization in data preprocessing?","To remove outliers completely","To scale features to have mean 0 and standard deviation 1","To delete data entirely","To add features only","To scale features to have mean 0 and standard deviation 1","Data preprocessing","Lecture MachineLearning"),

 ("What is the difference between standardization and normalization?","No difference at all","Standardization uses mean and std, normalization scales to a range like 0-1","Normalization uses mean and std values","Both use the same formula","Standardization uses mean and std, normalization scales to a range like 0-1","Data preprocessing","Lecture MachineLearning"),

 ("What is label encoding?","Encoding images into data","Converting categorical labels to numerical values","Removing labels completely","Adding more labels unnecessarily","Converting categorical labels to numerical values","Data preprocessing","Lecture MachineLearning"),

 ("What is the problem with label encoding for nominal categories?","No problem at all","It introduces ordinal relationships where none exist","It's too slow overall","It uses too much memory","It introduces ordinal relationships where none exist","Data preprocessing","Lecture MachineLearning"),

 ("What is gradient boosting?","Random combination of models","Ensemble method building models sequentially to correct previous errors using gradients","Training one model only","Removing gradients entirely","Ensemble method building models sequentially to correct previous errors using gradients","ML algorithms","Lecture MachineLearning-p2"),

 ("What is XGBoost?","A neural network architecture","An optimized implementation of gradient boosting","A clustering algorithm type","A data structure format","An optimized implementation of gradient boosting","ML algorithms","Lecture MachineLearning-p2"),

 ("What is feature importance in tree-based models?","Random values only","Measure of how useful each feature is for making predictions","Always equal for all features","Not measurable at all","Measure of how useful each feature is for making predictions","ML interpretation","Lecture MachineLearning-p2"),

 ("What is the purpose of pruning in decision trees?","To make trees larger overall","To reduce tree size and prevent overfitting","To add more branches overall","To remove all leaves completely","To reduce tree size and prevent overfitting","ML algorithms","Lecture MachineLearning-p2"),

 ("What is Gini impurity used for?","Classification accuracy only","Measuring how often a randomly chosen element would be incorrectly labeled","Regression error only","Data cleaning tasks","Measuring how often a randomly chosen element would be incorrectly labeled","ML algorithms","Lecture MachineLearning-p2"),

 ("What is information gain in decision trees?","Loss function only","Decrease in entropy after splitting on an attribute","Increase in complexity","Data augmentation method","Decrease in entropy after splitting on an attribute","ML algorithms","Lecture MachineLearning-p2"),

 ("What is K-fold cross-validation?","Using K models overall","Splitting data into K parts, training K times with different validation sets","Using K features only","Training for K epochs only","Splitting data into K parts, training K times with different validation sets","ML evaluation","Lecture MachineLearning-p2"),

 ("What is leave-one-out cross-validation?","Remove one feature only","K-fold CV where K equals the number of samples","Remove one class only","Use only one sample","K-fold CV where K equals the number of samples","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the purpose of data splitting in machine learning?","To delete data completely","To create independent sets for training, validation, and testing","To increase data size","To compress data efficiently","To create independent sets for training, validation, and testing","ML fundamentals","Lecture MachineLearning"),

 ("What is the No Free Lunch theorem in machine learning?","All algorithms are free","No algorithm is universally best for all problems","All algorithms perform equally","Free algorithms are best","No algorithm is universally best for all problems","ML theory","Lecture MachineLearning"),

 ("What is the formula for a simple linear regression model?","weight = b1 + b0 value","weight = b1 × height + b0","weight = height / b1 constant","weight = b0 - b1 value","weight = b1 × height + b0","Regression","DIKU 004 - Supervised Machine Learning"),

 ("In linear regression, what does b1 represent?","The intercept value","The slope of the line","The error term","The prediction accuracy","The slope of the line","Regression","DIKU 004 - Supervised Machine Learning"),

 ("In linear regression, what does b0 represent?","The slope of the line","The intercept value","The correlation coefficient","The standard deviation","The intercept value","Regression","DIKU 004 - Supervised Machine Learning"),

 ("What does R² (R-squared) measure in regression?","The slope steepness","The quality of fit (closer to 1 is better)","The number of data points","The training speed","The quality of fit (closer to 1 is better)","Regression","DIKU 004 - Supervised Machine Learning"),

 ("What is the best-fit line in regression?","The line that passes through all points","The line that minimizes error between prediction and data","The line with the steepest slope","The line with zero intercept","The line that minimizes error between prediction and data","Regression","DIKU 004 - Supervised Machine Learning"),

 ("What is regression used for in supervised learning?","Only for categorizing data","Prediction, interpolation, and inference of continuous values","Only for clustering data","Only for dimensionality reduction","Prediction, interpolation, and inference of continuous values","Regression","DIKU 004 - Supervised Machine Learning"),

 ("What is classification in supervised learning?","Predicting continuous numerical values","Dividing data into categories based on known labels","Finding patterns in unlabeled data","Reducing the number of features","Dividing data into categories based on known labels","Classification","DIKU 004 - Supervised Machine Learning"),

 ("What does classification require for training?","Only unlabeled data","Labeled data (training set) and testing data","Only test data","No data at all","Labeled data (training set) and testing data","Classification","DIKU 004 - Supervised Machine Learning"),

 ("What tool is used to evaluate classification model performance?","Regression line","Confusion matrix","Scatter plot","Histogram","Confusion matrix","Classification","DIKU 004 - Supervised Machine Learning"),

 ("Classification boundaries can be which of the following?","Only linear boundaries","Linear or complex (curved, multidimensional)","Only circular boundaries","Only straight vertical lines","Linear or complex (curved, multidimensional)","Classification","DIKU 004 - Supervised Machine Learning"),

 ("What is core business data?","Any random data collected","Data most directly tied to company's value-generating activities","Only financial statements","Only employee records","Data most directly tied to company's value-generating activities","Business data","DIKU 004 - Supervised Machine Learning"),

 ("What characterizes core business data?","Low dollar density overall","High dollar density with measurable financial impact per record","No connection to profit","Only historical data","High dollar density with measurable financial impact per record","Business data","DIKU 004 - Supervised Machine Learning"),

 ("What percentage of enterprise data is typically structured?","Around 80% of data","Around 20% of data","Around 50% of data","Around 5% of data","Around 20% of data","Data types","DIKU 004 - Supervised Machine Learning"),

 ("What percentage of enterprise data is typically unstructured?","Around 20% of data","Around 80% of data","Around 50% of data","Around 10% of data","Around 80% of data","Data types","DIKU 004 - Supervised Machine Learning"),

 ("Which type of data is easier to manage?","Unstructured data overall","Structured data (tabular)","Both are equally difficult","Neither can be managed","Structured data (tabular)","Data types","DIKU 004 - Supervised Machine Learning"),

 ("What type of data holds richer information but requires AI to extract value?","Structured data only","Unstructured data (images, audio, video, text)","Numerical data only","Spreadsheet data","Unstructured data (images, audio, video, text)","Data types","DIKU 004 - Supervised Machine Learning"),

 ("According to the 5 V's of Big Data, what does 'Volume' refer to?","Speed of data generation","Massive amounts of data (terabytes to petabytes)","Data accuracy and quality","Data usefulness overall","Massive amounts of data (terabytes to petabytes)","Big Data","DIKU 004 - Supervised Machine Learning"),

 ("According to the 5 V's of Big Data, what does 'Velocity' refer to?","Amount of data overall","Data is generated and processed rapidly","Data accuracy and quality","Data variety types","Data is generated and processed rapidly","Big Data","DIKU 004 - Supervised Machine Learning"),

 ("According to the 5 V's of Big Data, what does 'Variety' refer to?","Only structured data","Structured, semi-structured, and unstructured data types","Only numerical data","Only text data","Structured, semi-structured, and unstructured data types","Big Data","DIKU 004 - Supervised Machine Learning"),

 ("According to the 5 V's of Big Data, what does 'Veracity' refer to?","Data volume overall","Data accuracy and reliability","Data speed overall","Data storage types","Data accuracy and reliability","Big Data","DIKU 004 - Supervised Machine Learning"),

 ("According to the 5 V's of Big Data, what does 'Value' refer to?","Data size overall","Data usefulness for decision-making","Data speed overall","Data format types","Data usefulness for decision-making","Big Data","DIKU 004 - Supervised Machine Learning"),

 ("When did digital storage become inexpensive, marking the beginning of the digital age?","Around 1990 era","Around 2008 era","Around 2015 era","Around 2000 era","Around 2008 era","AI history","DIKU 004 - Supervised Machine Learning"),

 ("What costs businesses over $3.1 trillion per year?","Hardware costs overall","Poor data quality","Training costs overall","Storage costs overall","Poor data quality","Data quality","DIKU 004 - Supervised Machine Learning"),

 ("What is symbolic AI also known as?","Deep Learning methods","GOFAI (Good Old-Fashioned AI)","Neural networks overall","Genetic algorithms","GOFAI (Good Old-Fashioned AI)","AI types","DIKU 004 - Supervised Machine Learning"),

 ("What does symbolic AI use to represent knowledge?","Only numbers available","Symbols (nouns) and relations (verbs/adjectives)","Only images available","Only text available","Symbols (nouns) and relations (verbs/adjectives)","AI types","DIKU 004 - Supervised Machine Learning"),

 ("What logic operations does symbolic AI use?","Only multiplication operations","AND, OR, NOT","Only addition operations","Only division operations","AND, OR, NOT","AI types","DIKU 004 - Supervised Machine Learning"),

 ("What is fuzzy logic used for?","Binary true/false only","Handling uncertainty with degrees of truth (values between 0 and 1)","Only integer values","Only text processing","Handling uncertainty with degrees of truth (values between 0 and 1)","AI types","DIKU 004 - Supervised Machine Learning"),

 ("What is an example application of fuzzy logic?","Only image recognition","Home appliances and subway control systems","Only text analysis","Only speech recognition","Home appliances and subway control systems","AI types","DIKU 004 - Supervised Machine Learning"),

 ("According to George Box, what is true about models?","All models are perfect","All models are wrong, but some are useful","All models are useless","Models are always accurate","All models are wrong, but some are useful","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("What does learning mean in the context of machine learning?","Memorizing all data completely","Behavioral change from experience","Deleting old data only","Increasing storage capacity","Behavioral change from experience","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("How do machines learn?","By copying humans directly","By building models from data","By guessing randomly","By following fixed rules only","By building models from data","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("What are the main types of models?","Only mathematical models","Descriptive, predictive, mechanistic, and normative","Only statistical models","Only graphical models","Descriptive, predictive, mechanistic, and normative","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("What is a descriptive model?","Predicts future events","Represents current state","Optimizes strategies overall","Shows causal processes","Represents current state","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("What is a predictive model?","Shows current state only","Shows trends over time","Shows optimal strategies","Shows logical relations","Shows trends over time","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("What is a mechanistic model?","Shows current state overall","Shows causal processes","Shows optimal strategies","Shows trends only","Shows causal processes","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("What is a normative model?","Shows current state overall","Shows optimal strategies","Shows causal processes","Shows trends only","Shows optimal strategies","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("In supervised learning, what does 'supervised' refer to?","The algorithm supervises itself","Learning from labeled examples with known outputs","No human involvement","Random learning process","Learning from labeled examples with known outputs","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("What are the two main types of supervised learning problems?","Clustering and association","Regression and classification","Only neural networks","Only decision trees","Regression and classification","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("What type of output does regression predict?","Categories only","Continuous numerical values","Binary only","Text only","Continuous numerical values","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("What type of output does classification predict?","Continuous values","Categorical outcomes","Only numbers","Only images","Categorical outcomes","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("How is supervised learning performance measured?","Only by speed","By accuracy or error metrics","Only by cost","Only by time","By accuracy or error metrics","ML evaluation","DIKU 004 - Supervised Machine Learning"),

 ("What is machine learning according to DAVE3625?","Manual programming of all rules","Application of AI that allows systems to automatically learn and improve from experience without explicit programming","Only statistical analysis","Only data collection","Application of AI that allows systems to automatically learn and improve from experience without explicit programming","ML fundamentals","DAVE3625-MachineLearning1"),

 ("In the ML algorithm building process, what is the first step?","Test the model first","Collect data initially","Train the model first","Deploy the model first","Collect data initially","ML fundamentals","DAVE3625-MachineLearning1"),

 ("How does a reinforcement learning agent learn?","Only from labeled examples","By interacting with environment via trial and error","By clustering data together","By reducing dimensions only","By interacting with environment via trial and error","Reinforcement learning","DAVE3625-MachineLearning1"),

 ("What feedback system does reinforcement learning use?","No feedback at all","Reward if correct, penalty if wrong","Only penalties given","Only rewards given","Reward if correct, penalty if wrong","Reinforcement learning","DAVE3625-MachineLearning1"),

 ("What is the purpose of recommender systems?","To classify images only","Suggest relevant items to users and predict products likely to interest them","Only for search engines","Only for social media","Suggest relevant items to users and predict products likely to interest them","Recommender systems","DAVE3625-MachineLearning1"),

 ("When should you use machine learning?","Always for every problem","When rules are not explicitly known, but patterns can be inferred from data","Only for image processing","Only for text processing","When rules are not explicitly known, but patterns can be inferred from data","ML fundamentals","DAVE3625-MachineLearning1"),

 ("When was the Dartmouth Conference that founded AI?","1950 era","1956 era","1960 era","1970 era","1956 era","AI history","DIKU 002 - History of AI-p2"),

 ("Who were among the founding fathers of AI at Dartmouth?","Only John McCarthy","John McCarthy, Marvin Minsky, Claude Shannon, and others","Only Alan Turing","Only Marvin Minsky","John McCarthy, Marvin Minsky, Claude Shannon, and others","AI history","DIKU 002 - History of AI-p2"),

 ("What was SNARC?","The first computer ever","First neural network machine developed by Marvin Minsky in 1951","A programming language","A database system","First neural network machine developed by Marvin Minsky in 1951","AI history","DIKU 002 - History of AI-p2"),

 ("What was Logic Theorist designed to operate on?","Only numbers available","Symbols rather than numbers","Only text available","Only images available","Symbols rather than numbers","AI history","DIKU 002 - History of AI-p2"),

 ("Who developed the Perceptron?","Marvin Minsky","Frank Rosenblatt","John McCarthy","Claude Shannon","Frank Rosenblatt","AI history","DIKU 002 - History of AI-p2"),

 ("What was the Perceptron?","A programming language","An electronic device following biological principles, capable of learning","A database system","A mechanical calculator","An electronic device following biological principles, capable of learning","AI history","DIKU 002 - History of AI-p2"),

 ("What is the structure of Rosenblatt's Perceptron model?","inputs → outputs directly","inputs → weights → activation function → output","only activation function","weights only","inputs → weights → activation function → output","Neural networks","DIKU 002 - History of AI-p2"),

 ("Who developed early versions of deep learning models in the 1960s?","Frank Rosenblatt","Alexey G. Ivakhnenko","John McCarthy","Marvin Minsky","Alexey G. Ivakhnenko","AI history","DIKU 002 - History of AI-p2"),

 ("Which lambda function correctly creates a binary classification from grades (pass if >=10)?","df['passed'] = df['final_grade'].apply(lambda x: 1 if x >= 10 else 0)","df['passed'] = df['final_grade'].apply(lambda x: 1 if x > 10 else 0)","df['passed'] = df['final_grade'].apply(lambda x: True if x < 10 else False)","df['passed'] = df['final_grade'].apply(lambda x: 0 if x >= 10 else 1)","df['passed'] = df['final_grade'].apply(lambda x: 1 if x >= 10 else 0)","Lambda functions","Lab6"),

 ("Which code correctly applies StandardScaler to normalize features in scikit-learn?","scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_train)","scaler = StandardScaler(); X_scaled = scaler.transform(X_train)","scaler = StandardScaler(); X_scaled = scaler.fit(X_train)","scaler = StandardScaler(); X_scaled = scaler.normalize(X_train)","scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_train)","Feature scaling","Lab5"),

 ("What does np.where(df['quality'] >= 7, 1, 0) do?","Returns 1 where quality >= 7 and 0 otherwise","Returns 0 where quality >= 7 and 1 otherwise","Returns True where quality >= 7 and False otherwise","Returns quality value if >= 7, otherwise 0","Returns 1 where quality >= 7 and 0 otherwise","NumPy operations","Lab5"),

 ("Which lambda function filters rows where studytime > 2 in a pandas dataframe?","df_filtered = df[df['studytime'].apply(lambda x: x > 2)]","df_filtered = df[df['studytime'].map(lambda x: x > 2)]","df_filtered = df[df.apply(lambda x: x['studytime'] > 2, axis=1)]","df_filtered = df.filter(lambda x: x['studytime'] > 2)","df_filtered = df[df['studytime'].apply(lambda x: x > 2)]","Lambda functions","Lab6"),

 ("What does train_test_split(X, y, test_size=0.3, random_state=42) do?","Splits data 70% training / 30% testing with seed 42 for reproducibility","Splits data 30% training / 70% testing with seed 42 for reproducibility","Splits data 70% training / 30% testing with random splits each time","Splits data equally 50/50 with seed 42 for reproducibility","Splits data 70% training / 30% testing with seed 42 for reproducibility","Train-test split","Lab5"),

 ("Which code checks for missing values in each column of a pandas dataframe?","missing_values = df.isnull().sum()","missing_values = df.isna().count()","missing_values = df.null_count()","missing_values = df.missing().sum()","missing_values = df.isnull().sum()","Data cleaning","Lab5"),

 ("What does GridSearchCV(knn, param_grid, cv=5) do?","Tests parameter combinations with 5-fold cross-validation","Tests 5 different models with grid parameters","Validates model 5 times on same test set","Splits data into 5 equal parts for training","Tests parameter combinations with 5-fold cross-validation","Hyperparameter tuning","Lab5"),

 ("Which lambda expression correctly categorizes ages into groups?","df['age_group'] = df['age'].apply(lambda x: 'young' if x < 20 else 'old')","df['age_group'] = df['age'].map(lambda x: 'young' if x < 20 else 'old')","df['age_group'] = df.apply(lambda x: 'young' if x['age'] < 20 else 'old')","df['age_group'] = lambda x: df['age'] < 20 ? 'young' : 'old'","df['age_group'] = df['age'].apply(lambda x: 'young' if x < 20 else 'old')","Lambda functions","Lab6"),

 ("What does df.select_dtypes(include=[np.number]).columns.tolist() return?","List of numerical column names from the dataframe","List of all column names including non-numerical","List of numerical values from first row","List of column types for all columns","List of numerical column names from the dataframe","Pandas operations","Lab6"),

 ("Which code snippet correctly implements K-Nearest Neighbors with optimal K?","knn = KNeighborsClassifier(n_neighbors=5); knn.fit(X_train, y_train)","knn = KNN(neighbors=5); knn.train(X_train, y_train)","knn = KNeighborsClassifier(k=5); knn.fit(X_train, y_train)","knn = KNeighborsRegressor(n_neighbors=5); knn.fit(X_train, y_train)","knn = KNeighborsClassifier(n_neighbors=5); knn.fit(X_train, y_train)","KNN implementation","Lab5"),

 ("What does SVC(kernel='rbf') create compared to SVC(kernel='linear')?","Non-linear decision boundary vs linear hyperplane","Linear hyperplane vs non-linear decision boundary","Both create identical linear boundaries","Both create identical non-linear boundaries","Non-linear decision boundary vs linear hyperplane","SVM kernels","Lab5"),

 ("Which lambda function creates a new feature combining two columns?","df['total'] = df.apply(lambda row: row['A'] + row['B'], axis=1)","df['total'] = df.apply(lambda row: row['A'] + row['B'], axis=0)","df['total'] = df.map(lambda row: row['A'] + row['B'])","df['total'] = lambda row: df['A'] + df['B']","df['total'] = df.apply(lambda row: row['A'] + row['B'], axis=1)","Lambda functions","Lab6"),

 ("What does df.rename(columns={'G1': 'period_1_grades'}, inplace=True) do?","Renames column 'G1' to 'period_1_grades' and modifies original dataframe","Creates new dataframe with renamed column 'G1' to 'period_1_grades'","Renames row 'G1' to 'period_1_grades' and modifies original dataframe","Renames all columns to 'period_1_grades' in original dataframe","Renames column 'G1' to 'period_1_grades' and modifies original dataframe","Pandas operations","Lab6"),

 ("Which code correctly creates a Random Forest classifier with 100 trees?","rf = RandomForestClassifier(n_estimators=100); rf.fit(X_train, y_train)","rf = RandomForest(trees=100); rf.fit(X_train, y_train)","rf = RandomForestClassifier(n_trees=100); rf.train(X_train, y_train)","rf = ForestClassifier(estimators=100); rf.fit(X_train, y_train)","rf = RandomForestClassifier(n_estimators=100); rf.fit(X_train, y_train)","Random Forest","Lab6"),

 ("What does DecisionTreeClassifier(max_depth=5) limit?","Maximum depth of tree to 5 levels","Maximum number of features to 5","Maximum number of samples to 5","Maximum number of branches to 5","Maximum depth of tree to 5 levels","Decision Trees","Lab6"),

 ("Which lambda function correctly filters dataframe for absences <= 5?","filtered = df[df['absences'].apply(lambda x: x <= 5)]","filtered = df[df.apply(lambda x: x['absences'] <= 5)]","filtered = df.filter(lambda x: x['absences'] <= 5)","filtered = df[lambda x: df['absences'] <= 5]","filtered = df[df['absences'].apply(lambda x: x <= 5)]","Lambda functions","Lab6"),

 ("What does GaussianNB() assume about feature distributions?","Features follow Gaussian (normal) distribution within each class","Features follow uniform distribution across all classes","Features follow Poisson distribution within each class","Features follow exponential distribution across all classes","Features follow Gaussian (normal) distribution within each class","Naive Bayes","Lab6"),

 ("Which code snippet correctly calculates confusion matrix in scikit-learn?","cm = confusion_matrix(y_test, y_pred)","cm = confusion_matrix(y_pred, y_test)","cm = accuracy_score(y_test, y_pred)","cm = classification_report(y_test, y_pred)","cm = confusion_matrix(y_test, y_pred)","Model evaluation","Lab5"),

 ("What does df['quality'].value_counts() return?","Frequency count of each unique value in 'quality' column","Sum of all values in 'quality' column","Number of non-null values in 'quality' column","Statistical summary of 'quality' column values","Frequency count of each unique value in 'quality' column","Pandas operations","Lab5"),

 ("Which lambda expression creates age categories (child/teen/adult)?","df['category'] = df['age'].apply(lambda x: 'child' if x<13 else ('teen' if x<20 else 'adult'))","df['category'] = df['age'].map(lambda x: 'child' if x<13 else ('teen' if x<20 else 'adult'))","df['category'] = lambda x: 'child' if df['age']<13 else ('teen' if df['age']<20 else 'adult')","df['category'] = df.apply(lambda x: 'child' if x<13 else ('teen' if x<20 else 'adult'))","df['category'] = df['age'].apply(lambda x: 'child' if x<13 else ('teen' if x<20 else 'adult'))","Lambda functions","Lab6"),

 ("What does accuracy_score(y_test, y_pred) calculate?","Proportion of correct predictions out of total predictions","Sum of true positives and true negatives","Difference between predicted and actual values","Average of precision and recall","Proportion of correct predictions out of total predictions","Model evaluation","Lab5"),

 ("Which code creates a countplot to visualize class distribution?","sns.countplot(x='quality_binary', data=df)","sns.barplot(x='quality_binary', data=df)","sns.scatterplot(x='quality_binary', data=df)","sns.lineplot(x='quality_binary', data=df)","sns.countplot(x='quality_binary', data=df)","Data visualization","Lab5"),

 ("What does df.drop(['quality', 'quality_binary'], axis=1) do?","Removes columns 'quality' and 'quality_binary' from dataframe","Removes rows 'quality' and 'quality_binary' from dataframe","Removes all columns except 'quality' and 'quality_binary'","Removes all rows except 'quality' and 'quality_binary'","Removes columns 'quality' and 'quality_binary' from dataframe","Pandas operations","Lab5"),

 ("Which lambda function correctly converts Celsius to Fahrenheit?","df['fahrenheit'] = df['celsius'].apply(lambda x: (x * 9/5) + 32)","df['fahrenheit'] = df['celsius'].map(lambda x: (x * 9/5) + 32)","df['fahrenheit'] = df.apply(lambda x: (x['celsius'] * 9/5) + 32)","df['fahrenheit'] = lambda x: (df['celsius'] * 9/5) + 32","df['fahrenheit'] = df['celsius'].apply(lambda x: (x * 9/5) + 32)","Lambda functions","Lab6"),

 ("What does SVC(kernel='linear').decision_function(X_test) return?","Signed distance from samples to hyperplane","Probability predictions for each class","Binary predictions (0 or 1)","Accuracy score of the model","Signed distance from samples to hyperplane","SVM operations","Lab5"),

 ("Which code correctly splits features and target variable?","X = df.drop(columns=['passed']); y = df['passed']","X = df.remove(['passed']); y = df['passed']","X = df.drop('passed'); y = df.select('passed')","X = df.exclude(['passed']); y = df.get('passed')","X = df.drop(columns=['passed']); y = df['passed']","Data preparation","Lab6"),

 ("What does roc_curve(y_test, y_pred_proba) calculate?","False positive rate and true positive rate at various thresholds","Only accuracy at different thresholds","Only precision at different thresholds","Only recall at different thresholds","False positive rate and true positive rate at various thresholds","Model evaluation","Lab5"),

 ("Which lambda filters students with failures > 0 OR absences > 10?","filtered = df[df.apply(lambda x: x['failures'] > 0 or x['absences'] > 10, axis=1)]","filtered = df[df['failures'].apply(lambda x: x > 0 or df['absences'] > 10)]","filtered = df.filter(lambda x: x['failures'] > 0 or x['absences'] > 10)","filtered = lambda x: df[df['failures'] > 0 or df['absences'] > 10]","filtered = df[df.apply(lambda x: x['failures'] > 0 or x['absences'] > 10, axis=1)]","Lambda functions","Lab6"),

 ("What does pd.read_csv('data/wine.csv', sep=';') do?","Reads CSV file using semicolon as delimiter","Reads CSV file using comma as delimiter","Reads CSV file using space as delimiter","Reads CSV file using tab as delimiter","Reads CSV file using semicolon as delimiter","Data loading","Lab5"),

 ("Which code correctly implements Gaussian Naive Bayes?","nb = GaussianNB(); nb.fit(X_train, y_train)","nb = NaiveBayes(); nb.fit(X_train, y_train)","nb = GaussianNB(); nb.train(X_train, y_train)","nb = BayesClassifier(); nb.fit(X_train, y_train)","nb = GaussianNB(); nb.fit(X_train, y_train)","Naive Bayes","Lab6"),

 ("What does df.describe() provide for numerical columns?","Statistical summary including mean, std, min, max, and quartiles","Only mean and median values","Only count of non-null values","Only maximum and minimum values","Statistical summary including mean, std, min, max, and quartiles","Data exploration","Lab5"),

 ("Which lambda creates a grade trend feature (difference between periods)?","df['trend'] = df.apply(lambda x: x['period_2_grades'] - x['period_1_grades'], axis=1)","df['trend'] = df.apply(lambda x: x['period_2_grades'] - x['period_1_grades'], axis=0)","df['trend'] = df.map(lambda x: x['period_2_grades'] - x['period_1_grades'])","df['trend'] = lambda x: df['period_2_grades'] - df['period_1_grades']","df['trend'] = df.apply(lambda x: x['period_2_grades'] - x['period_1_grades'], axis=1)","Lambda functions","Lab6"),

 ("What parameter in KNeighborsClassifier determines the number of neighbors?","n_neighbors","k_value","num_neighbors","neighbors_count","n_neighbors","KNN parameters","Lab5"),

 ("Which code plots ROC curves for multiple models?","plt.plot(fpr, tpr, label='Model')","plt.scatter(fpr, tpr, label='Model')","plt.bar(fpr, tpr, label='Model')","plt.hist(fpr, tpr, label='Model')","plt.plot(fpr, tpr, label='Model')","Data visualization","Lab5"),

 ("What does df.info() display about a dataframe?","Column names, data types, non-null counts, and memory usage","Only column names","Only data types","Only non-null counts","Column names, data types, non-null counts, and memory usage","Data exploration","Lab6"),

 ("Which lambda function correctly bins continuous age into categories?","df['age_bin'] = df['age'].apply(lambda x: '0-20' if x<=20 else ('21-40' if x<=40 else '41+'))","df['age_bin'] = df['age'].map(lambda x: '0-20' if x<=20 else ('21-40' if x<=40 else '41+'))","df['age_bin'] = lambda x: '0-20' if df['age']<=20 else ('21-40' if df['age']<=40 else '41+')","df['age_bin'] = df.apply(lambda x: '0-20' if x<=20 else ('21-40' if x<=40 else '41+'))","df['age_bin'] = df['age'].apply(lambda x: '0-20' if x<=20 else ('21-40' if x<=40 else '41+'))","Lambda functions","Lab6"),

 ("What does RandomForestClassifier(n_estimators=100, max_depth=10) create?","Random forest with 100 trees, each limited to depth 10","Random forest with 10 trees, each limited to depth 100","Decision tree with 100 nodes and depth 10","Neural network with 100 neurons and 10 layers","Random forest with 100 trees, each limited to depth 10","Random Forest","Lab6"),

 ("Which code correctly computes feature importance from Random Forest?","importances = rf.feature_importances_","importances = rf.get_feature_importance()","importances = rf.compute_importances()","importances = rf.importance_scores()","importances = rf.feature_importances_","Random Forest","Lab6"),

 ("What does classification_report(y_test, y_pred) provide?","Precision, recall, F1-score, and support for each class","Only accuracy score","Only confusion matrix","Only ROC-AUC score","Precision, recall, F1-score, and support for each class","Model evaluation","Lab6"),

 ("Which lambda applies a discount based on quantity purchased?","df['price'] = df.apply(lambda x: x['base_price'] * 0.9 if x['qty'] > 10 else x['base_price'], axis=1)","df['price'] = df['base_price'].apply(lambda x: x * 0.9 if df['qty'] > 10 else x)","df['price'] = lambda x: df['base_price'] * 0.9 if df['qty'] > 10 else df['base_price']","df['price'] = df.map(lambda x: x['base_price'] * 0.9 if x['qty'] > 10 else x['base_price'])","df['price'] = df.apply(lambda x: x['base_price'] * 0.9 if x['qty'] > 10 else x['base_price'], axis=1)","Lambda functions","Lab6"),

 ("Why is StandardScaler important for KNN and SVM algorithms?","Features with larger scales dominate distance calculations without scaling","Scaling increases model accuracy by 100%","Scaling is required by pandas operations","Scaling reduces computation time significantly","Features with larger scales dominate distance calculations without scaling","Feature scaling","Lab5"),

 ("What does df.head(n) return?","First n rows of the dataframe","Last n rows of the dataframe","n random rows from dataframe","Summary statistics of n columns","First n rows of the dataframe","Pandas operations","Lab5"),

 ("Which lambda categorizes students by study time and absences?","df['risk'] = df.apply(lambda x: 'high' if x['studytime']<2 and x['absences']>10 else 'low', axis=1)","df['risk'] = df['studytime'].apply(lambda x: 'high' if x<2 and df['absences']>10 else 'low')","df['risk'] = lambda x: 'high' if df['studytime']<2 and df['absences']>10 else 'low'","df['risk'] = df.map(lambda x: 'high' if x['studytime']<2 and x['absences']>10 else 'low')","df['risk'] = df.apply(lambda x: 'high' if x['studytime']<2 and x['absences']>10 else 'low', axis=1)","Lambda functions","Lab6"),

 ("What is the purpose of random_state=42 in train_test_split?","Ensures reproducible splits across different runs","Sets training size to 42%","Limits random samples to 42","Creates 42 different splits","Ensures reproducible splits across different runs","Data splitting","Lab5"),

 ("Which code correctly creates binary target from continuous grades?","df['pass'] = np.where(df['grade'] >= 10, 1, 0)","df['pass'] = df['grade'].map(lambda x: 1 if x >= 10 else 0)","df['pass'] = np.if_else(df['grade'] >= 10, 1, 0)","df['pass'] = df['grade'].where(df['grade'] >= 10, 1, 0)","df['pass'] = np.where(df['grade'] >= 10, 1, 0)","Feature engineering","Lab5"),

 ("What does param_grid = {'n_neighbors': np.arange(1, 31)} create?","Dictionary with array of integers from 1 to 30 for grid search","Dictionary with array of integers from 0 to 31 for grid search","List with array of integers from 1 to 30 for testing","Dictionary with single value 31 for grid search","Dictionary with array of integers from 1 to 30 for grid search","Hyperparameter tuning","Lab5"),

 ("Which lambda calculates BMI from weight and height?","df['bmi'] = df.apply(lambda x: x['weight'] / (x['height'] ** 2), axis=1)","df['bmi'] = df.apply(lambda x: x['weight'] / (x['height'] ** 2), axis=0)","df['bmi'] = lambda x: df['weight'] / (df['height'] ** 2)","df['bmi'] = df.map(lambda x: x['weight'] / (x['height'] ** 2))","df['bmi'] = df.apply(lambda x: x['weight'] / (x['height'] ** 2), axis=1)","Lambda functions","Lab6"),

 ("What does auc(fpr, tpr) calculate?","Area under the ROC curve","Area under precision-recall curve","Total number of predictions","Average classification accuracy","Area under the ROC curve","Model evaluation","Lab5"),

 ("Which code correctly handles missing values by filling with mean?","df['age'].fillna(df['age'].mean(), inplace=True)","df['age'].replace(np.nan, df['age'].mean())","df['age'].fill(df['age'].mean(), inplace=True)","df['age'].substitute(np.nan, df['age'].mean())","df['age'].fillna(df['age'].mean(), inplace=True)","Data cleaning","Lab6"),

 ("What does df.apply(lambda x: x.max() - x.min(), axis=0) calculate?","Range (max - min) for each column","Range (max - min) for each row","Sum of max and min for each column","Mean of max and min for each row","Range (max - min) for each column","Lambda functions","Lab6"),

 ("Which visualization shows the relationship between two continuous variables?","plt.scatter(x, y)","plt.bar(x, y)","plt.hist(x)","plt.pie(x)","plt.scatter(x, y)","Data visualization","Lab5"),

 ("What does DecisionTreeClassifier(criterion='gini') use for splits?","Gini impurity to measure split quality","Entropy to measure split quality","Accuracy to measure split quality","Variance to measure split quality","Gini impurity to measure split quality","Decision Trees","Lab6"),

 ("Which lambda creates a full name column from first and last names?","df['full_name'] = df.apply(lambda x: x['first_name'] + ' ' + x['last_name'], axis=1)","df['full_name'] = df.apply(lambda x: x['first_name'] + ' ' + x['last_name'], axis=0)","df['full_name'] = lambda x: df['first_name'] + ' ' + df['last_name']","df['full_name'] = df.map(lambda x: x['first_name'] + ' ' + x['last_name'])","df['full_name'] = df.apply(lambda x: x['first_name'] + ' ' + x['last_name'], axis=1)","Lambda functions","Lab6"),

 
]




# Remove header row and empty questions
valid_questions = [q for q in qa_mc[1:] if q[0] and q[0] != "question"]

# Shuffle questions
np.random.shuffle(valid_questions)

# Quiz variables
remaining_questions = valid_questions.copy()  # Questions still available this round
answered_questions = set()  # Track unique questions answered (question text as ID)
score = 0
total_answered = 0
question_attempts = {}  # Track attempts per question
wrong_answers = {}  # Track wrong answers: {question_text: {'wrong_count': X, 'correct_answer': Y}}

def display_question(q):
    """Display a multiple choice question with options"""
    print(f"\n" + "-"*50)
    question_id = q[0]  # Use question text as ID
    attempts = question_attempts.get(question_id, 0)
    unique_answered = len(answered_questions)
    total_questions = len(valid_questions)
    print(f"\n{q[0]}")
    print(f"\na) {q[1]}")
    print(f"s) {q[2]}")
    print(f"d) {q[3]}")
    print(f"f) {q[4]}")
    
def display_answer(q, user_answer=None):
    """Display the correct answer and explanation"""
    correct_answer = q[5]
    if user_answer:
        # Find the letter for user answer and correct answer
        answer_map = {q[1]: 'a', q[2]: 's', q[3]: 'd', q[4]: 'f'}
        correct_letter = answer_map.get(correct_answer, '?')
        user_letter = answer_map.get(user_answer, '?')
        
        if user_answer.lower() == correct_answer.lower():
            print(f"Correct answer: {correct_letter} '{correct_answer}' | Your answer: {user_letter} | ✓ CORRECT!")
        else:
            print(f"Correct answer: {correct_letter} '{correct_answer}' | Your answer: {user_letter} | ✗ INCORRECT!")
    else:
        # Just showing answer without user input
        answer_map = {q[1]: 'a', q[2]: 's', q[3]: 'd', q[4]: 'f'}
        correct_letter = answer_map.get(correct_answer, '?')
        print(f"Correct answer: {correct_letter} '{correct_answer}'")

def get_user_choice():
    """Get user's choice for the multiple choice question"""
    while True:
        choice = input("\nAnswer (a/s/d/f), 'q' quit: ").strip().lower()
        if choice in ['a', 's', 'd', 'f', 'q']:
            return choice
        print("Please enter a, s, d, f, or q")



# Main quiz loop
print("Welcome to the Adaptive Multiple Choice Quiz!")

while len(remaining_questions) > 0:
    # Pick a random question from remaining questions
    question = remaining_questions[np.random.randint(0, len(remaining_questions))]
    question_id = question[0]
    
    display_question(question)
    choice = get_user_choice()
    
    if choice == 'q':
        break
    elif choice in ['a', 's', 'd', 'f']:
        # Track attempt
        question_attempts[question_id] = question_attempts.get(question_id, 0) + 1
        
        # Map choice to actual answer text
        answer_map = {'a': question[1], 's': question[2], 'd': question[3], 'f': question[4]}
        user_answer_text = answer_map[choice]
        
        display_answer(question, user_answer_text)
        
        # Update score and tracking
        total_answered += 1
        answered_questions.add(question_id)  # Track unique questions answered
        
        if user_answer_text.lower() == question[5].lower():
            score += 1
            # Remove question from remaining questions (mastered!)
            remaining_questions.remove(question)
        else:
            # Track wrong answer
            if question_id not in wrong_answers:
                wrong_answers[question_id] = {'wrong_count': 0, 'correct_answer': question[5]}
            wrong_answers[question_id]['wrong_count'] += 1
        
        # Check if all questions have been answered at least once
        if len(answered_questions) == len(valid_questions):
            remaining_questions = valid_questions.copy()
            answered_questions.clear()
        
        # Simple continue prompt
        input("Press Enter to continue...")
        # Continue to next random question


# Final score
print(f"\n" + "="*50)
if len(remaining_questions) == 0:
    print("🎉 CONGRATULATIONS! You've mastered all questions!")
else:
    print("QUIZ ENDED!")
    print(f"Questions remaining: {len(remaining_questions)}")

if total_answered > 0:
    percentage = (score / total_answered) * 100
    mastered = len(valid_questions) - len(remaining_questions)
    mastery_rate = (mastered / len(valid_questions)) * 100
    print(f"Session Score: {score}/{total_answered} ({percentage:.1f}%)")
    print(f"Overall Mastery: {mastered}/{len(valid_questions)} ({mastery_rate:.1f}%)")
else:
    print("No questions were answered.")

# Display wrong answers summary
if wrong_answers:
    print(f"\n" + "-"*50)
    print("QUESTIONS ANSWERED INCORRECTLY:")
    print("-"*50)
    for question_text, data in wrong_answers.items():
        # Truncate long questions for display
        display_question = question_text[:80] + "..." if len(question_text) > 80 else question_text
        print(f"❌ Wrong {data['wrong_count']} time(s): {display_question}")
        print(f"   Correct answer: {data['correct_answer']}")
        print()

print("Thanks for practicing!")

