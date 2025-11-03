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

 ("What is the purpose of regularization in machine learning?","To make models more complex","To prevent overfitting by adding penalty terms","To increase training speed","To reduce the number of features","To prevent overfitting by adding penalty terms","ML","flashcards"),

 ("What is the difference between L1 and L2 regularization?","No difference","L1 uses absolute values, L2 uses squared values","L2 uses absolute values, L1 uses squared values","Both use the same penalty method","L1 uses absolute values, L2 uses squared values","ML","flashcards"),

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

 ("What is the main goal of feature engineering?","To create new features that improve model performance","To reduce the number of features","To visualize the data","To select the best model","To create new features that improve model performance","Data engineering","Lab"),
 
 ("What is true about cross-validation?","It provides a more reliable estimate of model performance","It is not important","It is only used for small datasets","It always gives higher accuracy","It provides a more reliable estimate of model performance","ML evaluation","Lab"),

 ("What is L1 regularization?","A technique to reduce overfitting by adding a penalty for larger coefficients","A technique to increase model complexity","A method for feature selection","A type of neural network","A technique to reduce overfitting by adding a penalty for larger coefficients","ML","Lab"),

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

 ("During error analysis, what should you do with misclassified examples?","Delete them","Manually examine them and categorize the types of errors","Ignore them","Retrain immediately","Manually examine them and categorize the types of errors","ML strategy","Machine Learning Yearning"),

 ("If you find that 5 percent of dev set errors are due to a particular category, is it worth fixing?","Yes, always","Maybe not - focus on categories that account for larger error percentages","Yes, fix all errors","No, 5 percent is too small to matter","Maybe not - focus on categories that account for larger error percentages","ML strategy","Machine Learning Yearning"),

 ("What should you do if your dev and test sets come from different distributions?","Use them anyway","Make sure they come from the same distribution as your target application","Always use random splits","Ignore the difference","Make sure they come from the same distribution as your target application","ML strategy","Machine Learning Yearning"),

 ("What is a 'training-dev set'?","Another name for validation set","Data from the same distribution as training, but not used for training - helps detect variance","Data for developers only","The same as test set","Data from the same distribution as training, but not used for training - helps detect variance","ML strategy","Machine Learning Yearning"),

 ("If training error is low, training-dev error is low, but dev error is high, what's the problem?","Bias problem","Variance problem","Data mismatch problem","The model is perfect","Data mismatch problem","ML strategy","Machine Learning Yearning"),

 ("What should you do when you have a data mismatch problem?","Get more training data","Try to understand the difference between training and dev sets and add similar data to training","Change the algorithm","Use a smaller model","Try to understand the difference between training and dev sets and add similar data to training","ML strategy","Machine Learning Yearning"),

 ("What is transfer learning?","Transferring data between databases","Using knowledge from one task to help with another task","Moving models between servers","Sharing code between projects","Using knowledge from one task to help with another task","ML strategy","Machine Learning Yearning"),

 ("When does transfer learning make sense?","Always","When task A and B have the same input, you have more data for A than B, and low-level features help both","Only for image tasks","Never","When task A and B have the same input, you have more data for A than B, and low-level features help both","ML strategy","Machine Learning Yearning"),

 ("What is multi-task learning?","Training multiple separate models","Training one model to perform multiple tasks simultaneously","Using multiple GPUs","Training on multiple datasets sequentially","Training one model to perform multiple tasks simultaneously","ML strategy","Machine Learning Yearning"),

 ("When does multi-task learning make sense?","Always","When tasks share lower-level features, similar amount of data for each task, and you can train a big enough network","Only for NLP","Never","When tasks share lower-level features, similar amount of data for each task, and you can train a big enough network","ML strategy","Machine Learning Yearning"),

 ("What is end-to-end deep learning?","Training only the final layer","Replacing a multi-step pipeline with a single neural network","Training from scratch every time","Using multiple models in sequence","Replacing a multi-step pipeline with a single neural network","ML strategy","Machine Learning Yearning"),

 ("What is the main advantage of end-to-end learning?","It's always faster","It lets the model learn the optimal representation without hand-designed components","It requires less data","It's easier to debug","It lets the model learn the optimal representation without hand-designed components","ML strategy","Machine Learning Yearning"),

 ("What is the main disadvantage of end-to-end learning?","It's too slow","It requires a very large amount of data","It always overfits","It can't work with images","It requires a very large amount of data","ML strategy","Machine Learning Yearning"),

 ("What should guide your decision to use end-to-end learning?","Always use it","Whether you have enough data to learn the complexity of the mapping","Use it only for vision tasks","Never use it","Whether you have enough data to learn the complexity of the mapping","ML strategy","Machine Learning Yearning"),

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

 ("According to the AI Index Report, what is the primary ethical concern about AI in 2024?","Cost of deployment","Bias and fairness","Speed of processing","Energy consumption","Bias and fairness","AI ethics","AI Index Report 2025"),

 ("Which region showed the fastest growth in AI research publications according to the AI Index Report 2025?","North America","Europe","Asia","South America","Asia","AI research","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what percentage of AI PhD graduates in the US go into industry rather than academia?","35 percent","50 percent","65 percent","80 percent","65 percent","AI workforce","AI Index Report 2025"),

 ("What does the AI Index Report 2025 say about AI's impact on job displacement?","Minimal impact observed","Significant displacement in manufacturing and routine cognitive tasks","Only affects low-skill jobs","No measurable impact yet","Significant displacement in manufacturing and routine cognitive tasks","AI impact","AI Index Report 2025"),

 ("According to the AI Index Report, which application of AI in transportation saw the most investment in 2024?","Traffic management","Autonomous vehicles","Route optimization","Predictive maintenance","Autonomous vehicles","AI applications","AI Index Report 2025"),

 ("What trend did the AI Index Report 2025 identify in AI regulatory frameworks globally?","Decreasing regulation","Increasing fragmentation and country-specific approaches","Complete harmonization","No significant changes","Increasing fragmentation and country-specific approaches","AI policy","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what is the estimated growth rate of the global AI market from 2024 to 2030?","10 percent annually","20 percent annually","37 percent annually","50 percent annually","37 percent annually","AI industry trends","AI Index Report 2025"),

 ("Which AI technique showed the most improvement in benchmark performance in 2024 according to the report?","Reinforcement learning","Large language models","Computer vision","Speech recognition","Large language models","AI development","AI Index Report 2025"),

 ("According to the AI Index Report, what percentage of AI systems deployed in production experienced some form of failure or incident?","15 percent","28 percent","45 percent","60 percent","28 percent","AI reliability","AI Index Report 2025"),

 ("What does the AI Index Report 2025 identify as the biggest barrier to AI adoption in developing countries?","Lack of interest","Infrastructure and connectivity limitations","Cultural resistance","Too many regulations","Infrastructure and connectivity limitations","AI adoption","AI Index Report 2025"),

 ("According to the report, which industry sector has the highest AI adoption rate?","Healthcare","Finance and insurance","Retail","Manufacturing","Finance and insurance","AI adoption","AI Index Report 2025"),

 ("What trend did the AI Index Report identify regarding open-source AI models in 2024?","Declining in popularity","Significant increase in development and adoption","Remaining stable","Being replaced by proprietary models","Significant increase in development and adoption","AI development","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what is the primary driver of AI innovation?","Government funding","Academic research","Private sector investment","International collaboration","Private sector investment","AI development","AI Index Report 2025"),

 ("What does the report say about AI's energy consumption in 2024?","Decreasing due to efficiency gains","Growing concern due to training large models","Not significant enough to measure","Completely offset by renewable energy","Growing concern due to training large models","AI sustainability","AI Index Report 2025"),

 ("According to the AI Index Report, which country has the most comprehensive AI strategy?","United States","China","Singapore","European Union","China","AI policy","AI Index Report 2025"),

 ("What percentage of Fortune 500 companies have a dedicated AI strategy according to the AI Index Report 2025?","40 percent","60 percent","75 percent","90 percent","75 percent","AI adoption","AI Index Report 2025"),

 ("According to the report, what is the average time to deploy an AI model to production in 2024?","1-3 months","4-6 months","7-12 months","Over 1 year","4-6 months","AI development","AI Index Report 2025"),

 ("What does the AI Index Report identify as the most promising emerging AI application?","AI for climate change","AI for drug discovery","AI for education","AI for cybersecurity","AI for drug discovery","AI applications","AI Index Report 2025"),

 ("What are the three main types of machine learning?","Supervised, Unsupervised, Reinforcement","Classification, Regression, Clustering","Neural networks, Decision trees, SVM","Deep learning, Shallow learning, Transfer learning","Supervised, Unsupervised, Reinforcement","ML fundamentals","studocu"),

 ("In supervised learning, what is required for training?","Only input data","Only output labels","Both labeled input and output pairs","No data is needed","Both labeled input and output pairs","ML fundamentals","studocu"),

 ("What is the main goal of unsupervised learning?","To predict future values","To find patterns and structure in unlabeled data","To maximize rewards","To classify data into known categories","To find patterns and structure in unlabeled data","ML fundamentals","studocu"),

 ("What is the key characteristic of reinforcement learning?","Learning from labeled examples","Finding clusters in data","Learning through trial and error with rewards and penalties","Reducing dimensionality","Learning through trial and error with rewards and penalties","ML fundamentals","studocu"),

 ("What is overfitting in machine learning?","Model performs well on training data but poorly on new data","Model performs poorly on all data","Model is too simple","Model trains too quickly","Model performs well on training data but poorly on new data","ML concepts","studocu"),

 ("What is underfitting in machine learning?","Model is too complex","Model fails to capture the underlying pattern in the data","Model has perfect accuracy","Model only works on test data","Model fails to capture the underlying pattern in the data","ML concepts","studocu"),

 ("What is the purpose of a validation set?","To train the model","To tune hyperparameters and prevent overfitting","To replace the test set","To label data","To tune hyperparameters and prevent overfitting","ML concepts","studocu"),

 ("What is cross-validation used for?","To increase training speed","To assess model performance more reliably by using multiple train/test splits","To reduce dataset size","To eliminate outliers","To assess model performance more reliably by using multiple train/test splits","ML evaluation","studocu"),

 ("What is the bias-variance tradeoff?","Balance between model complexity and generalization ability","Balance between speed and accuracy","Balance between cost and performance","Balance between training and testing time","Balance between model complexity and generalization ability","ML theory","studocu"),

 ("What does high bias indicate in a model?","The model is overfitting","The model is underfitting and too simple","The model is perfect","The model is too complex","The model is underfitting and too simple","ML theory","studocu"),

 ("What does high variance indicate in a model?","The model is underfitting","The model is overfitting and too sensitive to training data","The model is balanced","The model needs more features","The model is overfitting and too sensitive to training data","ML theory","studocu"),

 ("What is a confusion matrix used for?","To confuse the model","To evaluate classification model performance by showing true/false positives and negatives","To visualize training loss","To select features","To evaluate classification model performance by showing true/false positives and negatives","ML evaluation","studocu"),

 ("What does True Positive (TP) mean in a confusion matrix?","Model correctly predicted negative class","Model incorrectly predicted positive class","Model correctly predicted positive class","Model incorrectly predicted negative class","Model correctly predicted positive class","ML evaluation","studocu"),

 ("What does False Positive (FP) mean?","Correctly predicted positive","Incorrectly predicted positive (Type I error)","Correctly predicted negative","Incorrectly predicted negative","Incorrectly predicted positive (Type I error)","ML evaluation","studocu"),

 ("What does False Negative (FN) mean?","Correctly predicted negative","Incorrectly predicted negative (Type II error)","Correctly predicted positive","Incorrectly predicted positive","Incorrectly predicted negative (Type II error)","ML evaluation","studocu"),

 ("What is gradient descent?","A clustering algorithm","An optimization algorithm that iteratively adjusts parameters to minimize loss","A classification method","A data preprocessing technique","An optimization algorithm that iteratively adjusts parameters to minimize loss","ML algorithms","studocu"),

 ("What is a learning rate in gradient descent?","The speed of data loading","The step size for parameter updates during optimization","The accuracy of the model","The number of epochs","The step size for parameter updates during optimization","ML algorithms","studocu"),

 ("What happens if the learning rate is too high?","Training is too slow","The algorithm may overshoot the minimum and fail to converge","The model becomes too accurate","Nothing significant","The algorithm may overshoot the minimum and fail to converge","ML algorithms","studocu"),

 ("What happens if the learning rate is too low?","Perfect convergence","Training is very slow and may get stuck in local minima","Model overfits immediately","No training occurs","Training is very slow and may get stuck in local minima","ML algorithms","studocu"),

 ("What is batch gradient descent?","Uses one sample at a time","Uses the entire dataset to compute gradient","Uses random samples","Uses only test data","Uses the entire dataset to compute gradient","ML algorithms","studocu"),

 ("What is stochastic gradient descent (SGD)?","Uses entire dataset","Uses one training example at a time to update parameters","Uses validation set","Uses multiple datasets","Uses one training example at a time to update parameters","ML algorithms","studocu"),

 ("What is mini-batch gradient descent?","Uses entire dataset","Uses small random batches of training data","Uses one sample","Uses test data","Uses small random batches of training data","ML algorithms","studocu"),

 ("What is the purpose of an activation function in neural networks?","To slow down training","To introduce non-linearity so the network can learn complex patterns","To reduce overfitting","To normalize inputs","To introduce non-linearity so the network can learn complex patterns","Neural networks","studocu"),

 ("What is the most common activation function for hidden layers?","Sigmoid","ReLU (Rectified Linear Unit)","Linear","Softmax","ReLU (Rectified Linear Unit)","Neural networks","studocu"),

 ("What activation function is typically used for binary classification output?","ReLU","Sigmoid","Tanh","Linear","Sigmoid","Neural networks","studocu"),

 ("What activation function is used for multi-class classification output?","Sigmoid","ReLU","Softmax","Tanh","Softmax","Neural networks","studocu"),

 ("What is backpropagation?","Forward pass through network","Algorithm for computing gradients and updating weights by propagating errors backward","Data preprocessing","Model evaluation","Algorithm for computing gradients and updating weights by propagating errors backward","Neural networks","studocu"),

 ("What is an epoch in neural network training?","One forward pass","One complete pass through the entire training dataset","One weight update","One batch","One complete pass through the entire training dataset","Neural networks","studocu"),

 ("What is dropout in neural networks?","Removing bad data","Regularization technique that randomly drops neurons during training to prevent overfitting","Stopping training early","Removing features","Regularization technique that randomly drops neurons during training to prevent overfitting","Neural networks","studocu"),

 ("What is the vanishing gradient problem?","Gradients explode during training","Gradients become very small in deep networks, making training difficult","Gradients disappear completely","Gradients become negative","Gradients become very small in deep networks, making training difficult","Neural networks","studocu"),

 ("What is the main difference between AI and Machine Learning?","They are the same thing","AI is the broader concept, ML is a subset focused on learning from data","ML is broader than AI","AI only works with images","AI is the broader concept, ML is a subset focused on learning from data","AI concepts","eksamen 2022"),

 ("What is the purpose of data normalization in machine learning?","To remove outliers","To scale features to similar ranges for better model performance","To add more data","To label the data","To scale features to similar ranges for better model performance","Data preprocessing","eksamen 2022"),

 ("Which metric would be most appropriate for imbalanced classification problems?","Accuracy","F1-score or precision-recall","Mean squared error","R-squared","F1-score or precision-recall","ML evaluation","eksamen 2022"),

 ("What is the curse of dimensionality?","Having too much data","Performance degradation as the number of features increases","Having too few samples","Model training is too fast","Performance degradation as the number of features increases","ML concepts","eksamen 2022"),

 ("What is the purpose of PCA (Principal Component Analysis)?","To increase dimensions","To reduce dimensionality while preserving variance","To classify data","To cluster data","To reduce dimensionality while preserving variance","ML techniques","eksamen 2022"),

 ("What does it mean when a model has high training accuracy but low test accuracy?","The model is underfitting","The model is overfitting","The model is perfect","The data is bad","The model is overfitting","ML concepts","eksamen 2022"),

 ("What are the three main components of AI according to the introduction lecture?","Hardware, Software, Data","Reasoning, Learning, Perception","Input, Processing, Output","Training, Testing, Deployment","Reasoning, Learning, Perception","AI fundamentals","Lecture Introduction-DAVE3625"),

 ("What is the main limitation of rule-based AI systems?","They require expensive hardware and infrastructure for deployment","They cannot handle uncertainty and require explicit programming for all scenarios","They are too slow for real-time processing applications","They use too much memory for practical applications","They cannot handle uncertainty and require explicit programming for all scenarios","AI limitations","Lecture Limitations-with-AI"),

 ("What is the main difference between strong AI and weak AI?","Strong AI has more accurate predictions than weak AI","Strong AI has general intelligence like humans, weak AI is task-specific","Strong AI processes information faster than weak AI","Strong AI uses larger datasets than weak AI","Strong AI has general intelligence like humans, weak AI is task-specific","AI concepts","Lecture Introduction-DAVE3625"),

 ("What is the primary goal of supervised learning?","To find hidden patterns in unlabeled data","To learn a mapping from inputs to outputs using labeled examples","To group similar items into clusters","To maximize rewards through trial and error","To learn a mapping from inputs to outputs using labeled examples","ML fundamentals","Lecture MachineLearning"),

 ("In the context of machine learning, what is a feature?","The output variable that we're trying to predict","An individual measurable property or characteristic of the data","The algorithm used to train the model","The training process that optimizes the model","An individual measurable property or characteristic of the data","ML fundamentals","Lecture MachineLearning"),

 ("What is the purpose of splitting data into training and test sets?","To save storage space by reducing data size","To evaluate model performance on unseen data","To speed up training by using smaller datasets","To reduce overfitting during training phase","To evaluate model performance on unseen data","ML fundamentals","Lecture MachineLearning"),

 ("What does the bias-variance tradeoff refer to?","Speed vs accuracy","The balance between underfitting and overfitting","Training time vs testing time","Memory usage vs performance","The balance between underfitting and overfitting","ML theory","Lecture MachineLearning-p2"),

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

 ("What is transfer learning in machine learning?","Transferring data between computers","Using knowledge from one task to improve learning on a related task","Moving models between servers","Translating between languages","Using knowledge from one task to improve learning on a related task","ML techniques","Lecture MachineLearning-p3-1"),

 ("What is the difference between classification and regression?","There is no difference between them","Classification predicts discrete categories, regression predicts continuous values","Regression predicts categories, classification predicts continuous values","Both predict the same types of outputs","Classification predicts discrete categories, regression predicts continuous values","ML fundamentals","Lecture MachineLearning"),

 ("What is a neural network layer?","A physical component of the computer","A collection of neurons that process inputs together","A type of data format","A training algorithm for optimization","A collection of neurons that process inputs together","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of the activation function in a neural network?","To slow training down","To introduce non-linearity and enable learning complex patterns","To reduce data size","To save memory usage","To introduce non-linearity and enable learning complex patterns","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is batch normalization in neural networks?","Removing bad data from batches","Normalizing inputs of each layer to stabilize and speed up training","Reducing batch size for efficiency","Increasing accuracy through larger batches","Normalizing inputs of each layer to stabilize and speed up training","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the vanishing gradient problem in deep learning?","Gradients becoming too large during training","Gradients becoming too small in early layers, preventing learning","Gradients disappearing completely from network","Loss function increasing during training","Gradients becoming too small in early layers, preventing learning","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is meant by model deployment in machine learning?","Training the model on data","Putting the trained model into production for real-world use","Collecting data for the model","Testing the model's accuracy","Putting the trained model into production for real-world use","ML lifecycle","Lecture MachineLearning"),

 ("What is A/B testing in the context of ML deployment?","Testing two different datasets","Comparing two model versions with real users to see which performs better","Testing on A grade vs B grade","Testing accuracy vs bias metrics","Comparing two model versions with real users to see which performs better","ML deployment","Lecture MachineLearning"),

 ("What is model monitoring in production?","Ignoring the model after deployment","Continuously tracking model performance to detect degradation or issues","Only checking once per month","Manual testing every few months","Continuously tracking model performance to detect degradation or issues","ML deployment","Lecture MachineLearning"),

 ("What is concept drift in machine learning?","Model getting better over time","When the statistical properties of the target variable change over time","Model staying the same forever","Training getting faster over time","When the statistical properties of the target variable change over time","ML deployment","Lecture MachineLearning"),

 ("What is the purpose of feature engineering?","To remove features from dataset","To create or transform features to improve model performance","To reduce training time significantly","To increase data size artificially","To create or transform features to improve model performance","ML techniques","Lecture MachineLearning"),

 ("What is one-hot encoding used for?","To encode numerical values","To convert categorical variables into binary vectors","To normalize data values","To reduce dimensions in data","To convert categorical variables into binary vectors","Data preprocessing","Lecture MachineLearning"),

 ("What is the purpose of data augmentation?","To delete unnecessary data","To artificially increase training data by creating modified versions","To compress data for storage","To visualize data patterns","To artificially increase training data by creating modified versions","Data preprocessing","Lecture MachineLearning"),

 ("What is imbalanced data in classification?","Equal class distribution","When one class has significantly more samples than others","All classes are missing","Data is corrupted","When one class has significantly more samples than others","Data issues","Lecture MachineLearning-p2"),

 ("What technique can help with imbalanced datasets?","Ignore the problem","Oversampling minority class or undersampling majority class","Delete all data","Use only one class","Oversampling minority class or undersampling majority class","Data preprocessing","Lecture MachineLearning-p2"),

 ("What is SMOTE in machine learning?","A type of neural network","Synthetic Minority Oversampling Technique for handling imbalanced data","A clustering algorithm","A regularization method","Synthetic Minority Oversampling Technique for handling imbalanced data","Data preprocessing","Lecture MachineLearning-p2"),

 ("What is the ROC curve's x-axis and y-axis?","Precision and Recall","False Positive Rate and True Positive Rate","Accuracy and Loss","Bias and Variance","False Positive Rate and True Positive Rate","ML evaluation","Lecture MachineLearning-p2"),

 ("What does an AUC (Area Under Curve) of 0.5 indicate?","Perfect model","Random guessing performance","Worst possible performance","Good performance","Random guessing performance","ML evaluation","Lecture MachineLearning-p2"),

 ("What does an AUC close to 1.0 indicate?","Poor model","Excellent model performance","Random model","Biased model","Excellent model performance","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the purpose of max pooling in CNNs?","To increase image size","To reduce spatial dimensions while retaining important features","To add more layers","To train faster","To reduce spatial dimensions while retaining important features","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is a convolutional layer in CNNs?","A fully connected layer","A layer that applies filters to detect features in images","An output layer","A normalization layer","A layer that applies filters to detect features in images","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the main advantage of CNNs for image processing?","They are faster","They can automatically learn spatial hierarchies of features","They use less memory","They require less data","They can automatically learn spatial hierarchies of features","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is a recurrent neural network (RNN) primarily used for?","Image classification","Sequential data like text or time series","Clustering","Dimensionality reduction","Sequential data like text or time series","Neural networks","Lecture MachineLearning-p3-1"),

 ("What problem do LSTMs solve compared to basic RNNs?","They are faster","They better handle long-term dependencies in sequences","They use less memory","They are simpler","They better handle long-term dependencies in sequences","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of attention mechanism in neural networks?","To make training faster","To allow the model to focus on relevant parts of the input","To reduce parameters","To normalize data","To allow the model to focus on relevant parts of the input","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is a loss function in machine learning?","A function that always returns zero","A function that measures the difference between predictions and actual values","A function that adds features","A function that removes outliers","A function that measures the difference between predictions and actual values","ML fundamentals","Lecture MachineLearning"),

 ("What is mean squared error (MSE) typically used for?","Classification","Regression problems","Clustering","Dimensionality reduction","Regression problems","ML evaluation","Lecture MachineLearning"),

 ("What is cross-entropy loss typically used for?","Regression","Classification problems","Clustering","Data preprocessing","Classification problems","ML evaluation","Lecture MachineLearning-p2"),

 ("What is early stopping in neural network training?","Starting training early","Stopping training when validation performance stops improving","Stopping after one epoch","Never stopping","Stopping training when validation performance stops improving","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of a learning rate scheduler?","To maintain constant learning rate","To adjust learning rate during training for better convergence","To increase learning rate only","To remove learning rate","To adjust learning rate during training for better convergence","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is batch size in neural network training?","Total dataset size","Number of samples processed before updating model parameters","Number of epochs","Number of layers","Number of samples processed before updating model parameters","Neural networks","Lecture MachineLearning-p3-1"),

 ("What happens if batch size is too large?","Training is always better","May not fit in memory and may lead to poor generalization","Training is faster","Model is more accurate","May not fit in memory and may lead to poor generalization","Neural networks","Lecture MachineLearning-p3-1"),

 ("What happens if batch size is too small?","Perfect training","Training becomes noisy and slow","Training is optimal","No training occurs","Training becomes noisy and slow","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of a validation set distinct from test set?","No purpose","To tune hyperparameters without touching the test set","To replace training set","To reduce data size","To tune hyperparameters without touching the test set","ML fundamentals","Lecture MachineLearning"),

 ("What is stratified sampling in train/test split?","Random sampling","Maintaining class proportions in splits","Taking only one class","Taking all data","Maintaining class proportions in splits","Data preprocessing","Lecture MachineLearning"),

 ("What is the difference between parameters and hyperparameters?","No difference","Parameters are learned during training, hyperparameters are set before","Hyperparameters are learned, parameters are set","Both are the same","Parameters are learned during training, hyperparameters are set before","ML fundamentals","Lecture MachineLearning"),

 ("What is the purpose of momentum in gradient descent?","To slow down training","To accelerate convergence by accumulating gradients","To increase loss","To remove features","To accelerate convergence by accumulating gradients","Optimization","Lecture MachineLearning-p2"),

 ("What is Adam optimizer?","A type of neural network","An adaptive learning rate optimization algorithm","A loss function","A regularization technique","An adaptive learning rate optimization algorithm","Optimization","Lecture MachineLearning-p3-1"),

 ("What is the main idea behind ensemble methods like Random Forest?","Use one tree","Combine multiple models to reduce variance and improve accuracy","Use only neural networks","Avoid decision trees","Combine multiple models to reduce variance and improve accuracy","ML algorithms","Lecture MachineLearning-p2"),

 ("What is bagging in ensemble learning?","Training one model","Training multiple models on different random subsets of data","Using bags for storage","Removing data","Training multiple models on different random subsets of data","ML techniques","Lecture MachineLearning-p2"),

 ("What is boosting in ensemble learning?","Random combination","Sequentially training models where each focuses on previous errors","Training in parallel only","Using only strong learners","Sequentially training models where each focuses on previous errors","ML techniques","Lecture MachineLearning-p2"),

 ("What is the difference between bagging and boosting?","No difference","Bagging trains in parallel, boosting trains sequentially focusing on errors","Boosting trains in parallel, bagging sequentially","Both are identical","Bagging trains in parallel, boosting trains sequentially focusing on errors","ML techniques","Lecture MachineLearning-p2"),

 ("What is the purpose of dropout rate in neural networks?","To keep all neurons","To specify the fraction of neurons to randomly drop during training","To add more layers","To reduce epochs","To specify the fraction of neurons to randomly drop during training","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is weight initialization in neural networks?","Setting all weights to zero","Setting initial weights before training begins","Final weight values","Removing weights","Setting initial weights before training begins","Neural networks","Lecture MachineLearning-p3-1"),

 ("Why is proper weight initialization important?","It's not important","Poor initialization can lead to vanishing/exploding gradients","It only affects speed","It reduces accuracy","Poor initialization can lead to vanishing/exploding gradients","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of the softmax function?","To make training harder","To convert outputs to probability distribution for multi-class classification","To normalize inputs","To reduce dimensions","To convert outputs to probability distribution for multi-class classification","Neural networks","Lecture MachineLearning-p2"),

 ("What is precision in classification?","Total correct predictions","True Positives divided by all predicted positives","True Positives divided by all actual positives","True Negatives divided by total","True Positives divided by all predicted positives","ML evaluation","Lecture MachineLearning-p2"),

 ("What is recall (sensitivity) in classification?","False positives rate","True Positives divided by all actual positives","True Positives divided by predicted positives","Total accuracy","True Positives divided by all actual positives","ML evaluation","Lecture MachineLearning-p2"),

 ("What does high precision but low recall indicate?","Model is perfect","Model is conservative, misses many positives but rarely wrong when it predicts positive","Model is random","Model always predicts positive","Model is conservative, misses many positives but rarely wrong when it predicts positive","ML evaluation","Lecture MachineLearning-p2"),

 ("What does high recall but low precision indicate?","Model is perfect","Model catches most positives but has many false alarms","Model is conservative","Model predicts nothing","Model catches most positives but has many false alarms","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the F1 score?","Average of precision and recall","Harmonic mean of precision and recall","Product of precision and recall","Difference between precision and recall","Harmonic mean of precision and recall","ML evaluation","Lecture MachineLearning-p2"),

 ("When should you use F1 score?","Never","When you need to balance precision and recall, especially with imbalanced classes","Only for regression","Only for clustering","When you need to balance precision and recall, especially with imbalanced classes","ML evaluation","Lecture MachineLearning-p2"),

 ("What is a true negative (TN) in classification?","Correctly predicted positive","Correctly predicted negative","Incorrectly predicted positive","Incorrectly predicted negative","Correctly predicted negative","ML evaluation","Lecture MachineLearning-p2"),

 ("What is specificity in classification?","True positive rate","True Negative divided by all actual negatives","True Positive divided by actual positives","False positive rate","True Negative divided by all actual negatives","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the purpose of standardization in data preprocessing?","To remove outliers","To scale features to have mean 0 and standard deviation 1","To delete data","To add features","To scale features to have mean 0 and standard deviation 1","Data preprocessing","Lecture MachineLearning"),

 ("What is the difference between standardization and normalization?","No difference","Standardization uses mean and std, normalization scales to a range like 0-1","Normalization uses mean and std","Both use the same formula","Standardization uses mean and std, normalization scales to a range like 0-1","Data preprocessing","Lecture MachineLearning"),

 ("What is label encoding?","Encoding images","Converting categorical labels to numerical values","Removing labels","Adding more labels","Converting categorical labels to numerical values","Data preprocessing","Lecture MachineLearning"),

 ("What is the problem with label encoding for nominal categories?","No problem","It introduces ordinal relationships where none exist","It's too slow","It uses too much memory","It introduces ordinal relationships where none exist","Data preprocessing","Lecture MachineLearning"),

 ("What is gradient boosting?","Random combination","Ensemble method building models sequentially to correct previous errors using gradients","Training one model","Removing gradients","Ensemble method building models sequentially to correct previous errors using gradients","ML algorithms","Lecture MachineLearning-p2"),

 ("What is XGBoost?","A neural network","An optimized implementation of gradient boosting","A clustering algorithm","A data structure","An optimized implementation of gradient boosting","ML algorithms","Lecture MachineLearning-p2"),

 ("What is feature importance in tree-based models?","Random values","Measure of how useful each feature is for making predictions","Always equal for all features","Not measurable","Measure of how useful each feature is for making predictions","ML interpretation","Lecture MachineLearning-p2"),

 ("What is the purpose of pruning in decision trees?","To make trees larger","To reduce tree size and prevent overfitting","To add more branches","To remove all leaves","To reduce tree size and prevent overfitting","ML algorithms","Lecture MachineLearning-p2"),

 ("What is Gini impurity used for?","Classification accuracy","Measuring how often a randomly chosen element would be incorrectly labeled","Regression error","Data cleaning","Measuring how often a randomly chosen element would be incorrectly labeled","ML algorithms","Lecture MachineLearning-p2"),

 ("What is information gain in decision trees?","Loss function","Decrease in entropy after splitting on an attribute","Increase in complexity","Data augmentation","Decrease in entropy after splitting on an attribute","ML algorithms","Lecture MachineLearning-p2"),

 ("What is K-fold cross-validation?","Using K models","Splitting data into K parts, training K times with different validation sets","Using K features","Training for K epochs","Splitting data into K parts, training K times with different validation sets","ML evaluation","Lecture MachineLearning-p2"),

 ("What is leave-one-out cross-validation?","Remove one feature","K-fold CV where K equals the number of samples","Remove one class","Use only one sample","K-fold CV where K equals the number of samples","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the purpose of data splitting in machine learning?","To delete data","To create independent sets for training, validation, and testing","To increase data size","To compress data","To create independent sets for training, validation, and testing","ML fundamentals","Lecture MachineLearning"),

 ("What is the No Free Lunch theorem in machine learning?","All algorithms are free","No algorithm is universally best for all problems","All algorithms perform equally","Free algorithms are best","No algorithm is universally best for all problems","ML theory","Lecture MachineLearning"),

 
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

