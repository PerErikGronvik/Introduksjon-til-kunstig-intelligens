# Multiple choice quiz program - similar to flashcards but with multiple choice questions
import numpy as np

# qa_mc = ["question","a", "b", "c", "d", "answer","category","source"]
qa_mc = [

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

 ("What is a sigmoid function?","A function that creates a linear relationship between inputs and outputs","A mathematical function that produces an S-shaped curve used in ML","A function only used in statistics for hypothesis testing","A function that always outputs zero or one exclusively","A mathematical function that produces an S-shaped curve used in ML","GLM","flashcards"),

 ("What is the difference between a generative and discriminative model?","No difference between them, they are essentially the same","Discriminative learns joint probability, generative learns conditional","Generative learns joint probability, discriminative learns conditional","Both learn only conditional probability without joint distributions","Generative learns joint probability, discriminative learns conditional","ML","flashcards"),



 ("What is gradient descent?","A data preprocessing technique for cleaning datasets","An optimization algorithm to minimize cost functions","A type of neural network architecture for classification","A feature selection method for dimensionality reduction","An optimization algorithm to minimize cost functions","ML","flashcards"),

 ("What is a decision boundary?","The edge of a dataset defining its limits","A surface that separates different classes","A type of cost function for optimization","A data validation technique for accuracy","A surface that separates different classes","ML","flashcards"),

 ("What does 96 percent accuracy mean?","The model is 96 percent confident in its predictions","Model correctly predicted 96 percent of output labels","Training took 96 percent of expected computational time","96 percent of features were used during model training","Model correctly predicted 96 percent of output labels","ML","flashcards"),

 ("What is the logit function used for in logistic regression?","To increase model complexity during training","To transform probabilities to linear combinations","To validate input data for quality assurance","To reduce computational cost during inference","To transform probabilities to linear combinations","GLM","flashcards"),

 ("What is Naive Bayes?","A complex neural network architecture for deep learning tasks and applications","A probabilistic classifier using Bayes' theorem with independence assumptions","A regression algorithm for continuous value prediction in supervised learning","A clustering method for unsupervised learning tasks and pattern discovery","A probabilistic classifier using Bayes' theorem with independence assumptions","ML","flashcards"),

 ("What is Naive Bayes particularly good at?","Image recognition and computer vision tasks","Spam detection and text sentiment analysis","Weather prediction and climate modeling","Stock price prediction and forecasting","Spam detection and text sentiment analysis","ML","flashcards"),

 ("What is Deep Learning?","Basic machine learning with simple algorithms","Subset of ML using neural networks with many layers","Only for image processing and computer vision","A type of database for storing large datasets","Subset of ML using neural networks with many layers","ML","flashcards"),

 ("What is Support Vector Machine (SVM)?","Unsupervised clustering algorithm for grouping similar data points","Supervised algorithm finding optimal hyperplane for classification","Only for text processing and sentiment analysis in NLP applications","A data preprocessing tool for feature engineering and transformation","Supervised algorithm finding optimal hyperplane for classification","ML","flashcards"),

 ("When is SVM typically used?","Large datasets only with millions of samples","Small datasets requiring high accuracy","Only for regression tasks and predictions","Real-time applications requiring speed","Small datasets requiring high accuracy","ML","flashcards"),

 ("What is K-means?","Supervised learning algorithm for classification of labeled examples","Unsupervised clustering algorithm partitioning data into K clusters","Classification algorithm for labeled data using decision boundaries","Regression technique for continuous predictions with numeric outputs","Unsupervised clustering algorithm partitioning data into K clusters","ML","flashcards"),

 ("What is the purpose of feature scaling?","To reduce dataset size for faster computation and storage efficiency","To normalize the range of features for better model performance","To add more features to improve predictions through engineering","To remove outliers from the dataset entirely before training","To normalize the range of features for better model performance","ML","flashcards"),

 ("What is the curse of dimensionality?","Having too few features for effective modeling","Problems arising when analyzing high-dimensional data","Having too much data for processing capacity","Network connectivity issues during training","Problems arising when analyzing high-dimensional data","ML","flashcards"),

 ("What is Lasso regression?","Basic linear regression without regularization","Linear regression with L1 regularization","Non-linear regression for complex patterns","Clustering algorithm for unsupervised learning","Linear regression with L1 regularization","ML","flashcards"),

 ("Adding a new column based on data available is considered creating a","Feature","A Label","A Method","A Column","Feature","Data engineering","Lab"),

 ("What is Accuracy in ML","Percentage of correctly predicted instances out of the total instances","TP divided by sum of TP and FP for positive predictions overall","TP divided by sum of TP and FN for recall calculation purposes","TN divided by sum of TN and FP for specificity calculation","Percentage of correctly predicted instances out of the total instances","ML evaluation","Lab"),

 ("What is Precision in ML","TP + TN / Total","TP / (TP + FP)","TP / (TP + FN)","TN / (TN + FP)","TP / (TP + FP)","ML evaluation","Lab"),

 ("What is Recall in ML","TP + TN / Total","TP / (TP + FP)","TP / (TP + FN)","TN / (TN + FP)","TP / (TP + FN)","ML evaluation","Lab"),

 ("What is F1 Score in ML","2 * (Precision * Recall) / (Precision + Recall)","(Precision * Recall) / (Precision + Recall)","Precision divided by (Precision + Recall)","2 * (Precision + Recall) / (Precision * Recall)","2 * (Precision * Recall) / (Precision + Recall)","ML evaluation","Lab"),

 ("What is correct about Log Loss?","Used for regression problems","It is used for classification","Not used in ML at all today","A type of regularization","It is used for classification","ML evaluation","Lab"),

 ("What does Log Loss represent?","How close the predicted probabilities are to the actual class labels","The accuracy of the model on the test dataset overall performance","The precision of the model for positive class predictions overall","The recall of the model for positive class predictions evaluation","How close the predicted probabilities are to the actual class labels","ML evaluation","Lab"),

 ("What is the ideal value for Log Loss?","0","1","2","5","0","ML evaluation","Lab"),

 ("What is a ROC curve?","A plot of true positive rate vs false positive rate","A plot of precision vs recall for all thresholds","A plot of accuracy vs error rate for evaluation","A plot of loss vs epochs during training process","A plot of true positive rate vs false positive rate","ML evaluation","Lab"),

 ("What does the area under the ROC curve (AUC) represent?","The model's ability to distinguish between classes","The model's accuracy on the test dataset","The model's precision for positive predictions","The model's recall for positive class instances","The model's ability to distinguish between classes","ML evaluation","Lab"),

 ("Is there a trade-off between precision and recall","Yes","No","Maybe","Rarely","Yes","ML evaluation","Lab"),

 ("What is the trade-off between precision and recall?","Increasing one decreases the other","They are independent of each other","Increasing one increases the other","They are always equal in practice","Increasing one decreases the other","ML evaluation","Lab"),

 ("What is regarded as the first ai software","Watson system","Deep Blue IBM","Logic Theorist","AlphaGo DeepMind","Logic Theorist","AI history","DIKU 002"),

 ("What is the first chat bot and who invented it.","Siri by Apple Corporation","Alexa by Amazon Web Services","Cortana by Microsoft Research","ELIZA by Joseph Weizenbaum","ELIZA by Joseph Weizenbaum","AI history","DIKU 002"),

 ("When was the AI winter","1970s era period","1980s era period","1990s era period","Both 1970 and 1990","Both 1970 and 1990","AI history","DIKU 002"),

 ("What happened in the 1980 revival of AI","The video game industry started using AI as a test bed, Japan announces a 850 millions investment in AI, first autonomous vehicle using a neural network, focus switched to narrow ai","Sweden announced a $600 million investment in computer research, AI was used in medical imaging for the first time, and companies began developing AI-powered tractors for agriculture.","Germany launched a national robotics initiative, new AI programs were built to compose music, and researchers focused on creating fully general intelligence systems.","France funded a major AI art project, universities began using AI to automate grading, and the first AI-operated subway system was tested in London for public transportation.","The video game industry started using AI as a test bed, Japan announces a 850 millions investment in AI, first autonomous vehicle using a neural network, focus switched to narrow ai","AI history","DIKU 002"),

 ("What is a key ethical requirement for Artificial Intelligence developed and used in Norway?","It should improve efficiency and reduce labor","It should respect human rights and democracy","It should align with market innovation goals","It should adapt to European data frameworks","It should respect human rights and democracy","AI act","DIKU 002"),
 
 ("Which description best matches Artificial Intelligence?","Produces insights based on data, is commonly “one-off,” and usually takes the form of a report or presentation.","Automates tasks or predicts future events based on data, is commonly used “live,” and often takes the form of software.","Focuses on collecting, cleaning, and storing data for later analysis, and mainly supports database infrastructure.","Provides statistical summaries to guide strategic planning, often completed once per project and presented to management.","Automates tasks or predicts future events based on data, is commonly used “live,” and often takes the form of software.","AI","History of AI document"),
 
 ("Which description best matches Data Science?","Produces insights based on data, is commonly \"one-off,\" and usually takes the form of a report or presentation.","Builds neural networks that run continuously and make decisions without human input.","Automates tasks or predicts future events based on data, is commonly used \"live,\" and often takes the form of software.","Focuses on developing digital interfaces that gather user feedback for automated systems.","Produces insights based on data, is commonly \"one-off,\" and usually takes the form of a report or presentation.","AI","History of AI document"),

 ("what is a holdout set in machine learning?", "A portion of the dataset set aside for final model evaluation","A subset of data used for training the model","Data used for hyperparameter tuning","Data used for feature selection","A portion of the dataset set aside for final model evaluation","ML concepts","statistisk analyse mamo2100"),

 ("numpy is used for?", "Data manipulation and numerical computations","Building neural networks","Creating visualizations","Data storage","Data manipulation and numerical computations","Python libraries","shiberas"),

 ("pandas is used for?", "Data manipulation and analysis","Building machine learning models","Creating web applications","Data visualization","Data manipulation and analysis","Python libraries","shiberas"),

 ("the difference between numpy and pandas is?", "numpy is for numerical computations, pandas is for data manipulation and analysis","Both are used for numerical computations","Both are used for data visualization","Both are used for building machine learning models","numpy is for numerical computations, pandas is for data manipulation and analysis","Python libraries","shiberas"),

 ("scikit-learn is a machine learning library in python, and it is used for?", "Building and training machine learning models","Data manipulation and analysis","Creating visualizations","Statistical modeling","Building and training machine learning models","Python libraries","shiberas"),

 ("matplotlib is used for?", "Creating visualizations and plots","Building machine learning models","Data manipulation and analysis","Numerical computations","Creating visualizations and plots","Python libraries","shiberas"),

 ("Hvilken kodelinje er riktig for å utføre regresjon", "model = LinearRegression().fit(X, y)","model = LogisticRegression().fit(X, y)","model = KMeans().fit(X)","model = SVC().fit(X, y)","model = LinearRegression().fit(X, y)","GLM regresjon","shiberas"),

 ("What is the main purpose of a single-number evaluation metric?","To make code run faster during execution and reduce computation time","To help quickly compare different models and choose the best one","To reduce the size of the dataset for storage and memory efficiency","To eliminate the need for human judgment entirely in all decisions","To help quickly compare different models and choose the best one","ML strategy","Machine Learning Yearning"),

 ("What should you do if you have multiple metrics you care about?","Pick one as the optimizing metric and others as satisficing metrics","Ignore all but one metric completely during evaluation process","Average all metrics together into single combined value overall","Only use accuracy for all tasks regardless of context or scenario","Pick one as the optimizing metric and others as satisficing metrics","ML strategy","Machine Learning Yearning"),

 ("What is the recommended split ratio for dev/test sets in modern ML with large datasets?","50/50 dev/test split","70/30 dev/test split","98/1/1 train/dev/test","60/20/20 split ratio","98/1/1 train/dev/test","ML strategy","Machine Learning Yearning"),

 ("What is the key principle about dev and test set distribution?","They should come from different distributions to test generalization well","They should come from the same distribution as what you want to do well on","Dev set should be harder than test set for better evaluation results","Test set should be from training data for consistency in testing","They should come from the same distribution as what you want to do well on","ML fundamentals","Machine Learning Yearning"),

 ("What size should your dev and test sets be?","As large as possible to maximize available data for evaluation","Large enough to give high confidence in the overall performance of your system","Exactly 30% of total data according to standard practice rules","At least 10,000 examples each as a minimum requirement standard","Large enough to give high confidence in the overall performance of your system","ML fundamentals","Machine Learning Yearning"),

 ("When should you change dev/test sets or metrics?","Never, keep them fixed throughout the entire project lifecycle","When your metric is no longer measuring what is most important to you","Only after deploying to production environment successfully","Every week according to regular scheduled maintenance procedures","When your metric is no longer measuring what is most important to you","ML fundamentals","Machine Learning Yearning"),

 ("What is the first thing to setup before starting an ML project?","Build the neural network architecture","Set up dev/test sets and metrics","Collect more data from various sources","Deploy to production immediately","Set up dev/test sets and metrics","ML fundamentals","Machine Learning Yearning"),

 ("What does 'avoidable bias' refer to?","The difference between training error and dev error in evaluation","The difference between training error and human-level performance","Any bias in the training data from collection process","The difference between dev error and test error in validation","The difference between training error and human-level performance","ML concepts","Machine Learning Yearning"),

 ("What does 'variance' indicate in ML model performance?","The model is not fitting the training set well enough overall","The model is overfitting - doing well on training but not on dev/test","The data is biased from improper collection procedures","Human-level performance is low compared to baseline expectations","The model is overfitting - doing well on training but not on dev/test","ML concepts","Machine Learning Yearning"),

 ("If training error is 15 percent and dev error is 16 percent, and human-level error is 0 percent, what should you focus on?","Reduce variance","Reducing bias","Get more data","Change metric","Reducing bias","ML strategy","Machine Learning Yearning"),

 ("If training error is 1 percent and dev error is 11 percent, and human-level error is 0 percent, what should you focus on?","Reducing bias","Reducing variance","Error is too low","Model is perfect","Reducing variance","ML strategy","Machine Learning Yearning"),

 ("What is the most reliable way to reduce bias in your model?","Get more training data","Train a bigger model","Add regularization","Smaller learning rate","Train a bigger model","ML strategy","Machine Learning Yearning"),

 ("What is the most reliable way to reduce variance in your model?","Train a bigger model with more parameters","Get more training data or add regularization","Use a higher learning rate for faster","Remove features from the input data","Get more training data or add regularization","ML techniques","Machine Learning Yearning"),

 ("What is 'Bayes error' or 'Bayes optimal error'?","The error rate of the best possible function","The error rate of Bayes' theorem application","Always 0% in all possible scenarios","The average human error rate performance","The error rate of the best possible function","ML theory","Machine Learning Yearning"),

 ("Why is human-level performance often used as a proxy for Bayes error?","Because humans are always optimal at every single task in all possible domains","Because it's easy to calculate without complex computations or theoretical work","Because for tasks humans are good at, human-level performance is close to Bayes error","Because Bayes error doesn't exist in practice at all for any real-world tasks","Because for tasks humans are good at, human-level performance is close to Bayes error","ML theory","Machine Learning Yearning"),

 ("When should you stop trying to reduce bias?","Never stop trying to reduce it further ever","When training error reaches human-level performance","When dev error is 0% on all examples tested","After 100 epochs of training iterations completed","When training error reaches human-level performance","ML techniques","Machine Learning Yearning"),

 ("What is error analysis?","Running your model on examples it got wrong and analyzing the patterns","Deleting wrong predictions from the dataset entirely without review","Calculating error rates across all test examples systematically","Testing on production data with real user inputs continuously","Running your model on examples it got wrong and analyzing the patterns","ML techniques","Machine Learning Yearning"),

 ("During error analysis, what should you do with misclassified examples?","Delete them from the dataset immediately without review","Manually examine them and categorize the types of errors","Ignore them and continue training without changes","Retrain the model immediately without analysis first","Manually examine them and categorize the types of errors","ML techniques","Machine Learning Yearning"),

 ("If you find that 5 percent of dev set errors are due to a particular category, is it worth fixing?","Yes, always fix every category immediately regardless of impact","Maybe not - focus on categories that account for larger error percentages","Yes, fix all errors regardless of size or importance to metrics","No, 5 percent is too small to matter at all in any situation","Maybe not - focus on categories that account for larger error percentages","ML techniques","Machine Learning Yearning"),

 ("What should you do if your dev and test sets come from different distributions?","Use them anyway without changes to the dataset splits at all","Make sure they come from the same distribution as your target application","Always use random splits for both sets without consideration","Ignore the difference completely and proceed with training","Make sure they come from the same distribution as your target application","ML fundamentals","Machine Learning Yearning"),

 ("What is a 'training-dev set'?","Another name for the validation set used in cross-validation for hyperparameter tuning","Data from the same distribution as training, but not used for training - helps detect variance","Data for developers only to use during development process and debugging activities","The same as the test set completely without any differences in distribution or purpose","Data from the same distribution as training, but not used for training - helps detect variance","ML concepts","Machine Learning Yearning"),

 ("If training error is low, training-dev error is low, but dev error is high, what's the problem?","Bias problem with the model","Variance problem with the model","Data mismatch problem between sets","The model is perfect already","Data mismatch problem between sets","ML strategy","Machine Learning Yearning"),

 ("What should you do when you have a data mismatch problem?","Get more training data immediately without further analysis first to increase dataset size","Try to understand the difference between training and dev sets and add similar data to training","Change the algorithm completely to a different approach entirely without investigation","Use a smaller model instead of the current architecture design to reduce complexity","Try to understand the difference between training and dev sets and add similar data to training","ML techniques","Machine Learning Yearning"),

 ("What is transfer learning?","Transferring data between multiple databases for storage","Using knowledge from one task to help with another task","Moving models between different servers for deployment","Sharing code between different projects in development","Using knowledge from one task to help with another task","ML techniques","Machine Learning Yearning"),

 ("When does transfer learning make sense?","Always in every situation regardless of task similarity or data availability for best results","When task A and B have the same input, you have more data for A than B, and low-level features help both","Only for image tasks specifically in computer vision applications with convolutional networks","Never use it at all because it always hurts performance and wastes computational resources","When task A and B have the same input, you have more data for A than B, and low-level features help both","ML techniques","Machine Learning Yearning"),

 ("What is multi-task learning?","Training multiple separate models independently for each task","Training one model to perform multiple tasks simultaneously","Using multiple GPUs for training to speed up computation","Training on multiple datasets sequentially one after another","Training one model to perform multiple tasks simultaneously","ML techniques","Machine Learning Yearning"),

 ("When does multi-task learning make sense?","Always in every scenario regardless of task characteristics or complexity and when you have sufficient data","When tasks share lower-level features, similar amount of data for each task, and you can train a big enough network","Only for NLP tasks specifically in natural language processing applications where semantic meaning matters","Never use it in practice because it hurts performance and increases complexity without providing benefits","When tasks share lower-level features, similar amount of data for each task, and you can train a big enough network","ML techniques","Machine Learning Yearning"),

 ("What is end-to-end deep learning?","Training only the final layer of neural network architecture","Replacing a multi-step pipeline with a single neural network","Training from scratch every time without transfer learning","Using multiple models in sequence for complex predictions","Replacing a multi-step pipeline with a single neural network","ML techniques","Machine Learning Yearning"),

 ("What is the main advantage of end-to-end learning?","It's always faster than alternatives in all situations encountered","It lets the model learn the optimal representation without hand-designed components","It requires less data overall compared to other approaches in practice","It's easier to debug issues when problems occur during development","It lets the model learn the optimal representation without hand-designed components","ML techniques","Machine Learning Yearning"),

 ("What is the main disadvantage of end-to-end learning?","It's too slow for production deployment","It requires a very large amount of data","It always overfits the training set badly","It can't work with images at all ever","It requires a very large amount of data","ML techniques","Machine Learning Yearning"),

 ("What should guide your decision to use end-to-end learning?","Always use it regardless of circumstances in all situations","Whether you have enough data to learn the complexity of the mapping","Use it only for vision tasks specifically in computer vision","Never use it at all in practice for any application scenario","Whether you have enough data to learn the complexity of the mapping","ML techniques","Machine Learning Yearning"),

 ("Hvilken kodelinje er riktig for å utføre klassifisering", "model = LogisticRegression().fit(X, y)","model = LinearRegression().fit(X, y)","model = KMeans().fit(X)","model = SVC().fit(X, y)","model = LogisticRegression().fit(X, y)","GLM regresjon","shiberas"),

 ("Hvilken kodelinje er riktig for å utføre klynging", "model = KMeans().fit(X)","model = LogisticRegression().fit(X, y)","model = LinearRegression().fit(X, y)","model = SVC().fit(X, y)","model = KMeans().fit(X)","GLM regresjon","shiberas"),

 ("Hvilken kodelinje er riktig for å finne den mest forekommende verdien i en kolonne i en pandas dataframe", "Grc_df['Outlet_Size'].value_counts().idxmax()","Grc_df['Outlet_Size'].max()","Grc_df['Outlet_Size'].mean()","Grc_df['Outlet_Size'].min()","Grc_df['Outlet_Size'].value_counts().idxmax()","Pandas dataframe","shiberas"),

 ("Hvilken kodelinje er riktig for å dele en kolonne i en pandas dataframe i 10 like store binner", "Grc_Concat_df['Item_Weight_Binned'] = pd.cut(Grc_Concat_df['Item_Weight'], bins=10)","Grc_Concat_df['Item_Weight_Binned'] = pd.qcut(Grc_Concat_df['Item_Weight'], bins=10)","Grc_Concat_df['Item_Weight_Binned'] = pd.split(Grc_Concat_df['Item_Weight'], bins=10)","Grc_Concat_df['Item_Weight_Binned'] = pd.bucket(Grc_Concat_df['Item_Weight'], bins=10)","Grc_Concat_df['Item_Weight_Binned'] = pd.cut(Grc_Concat_df['Item_Weight'], bins=10)","Pandas dataframe","shiberas"),

 ("Hvilken kodelinje er riktig for å fylle inn manglende verdier i en kolonne med medianen av den kolonnen", "df['Age'] = df['Age'].fillna(df['Age'].median())","df['Age'] = df['Age'].fillna(df['Age'].mean())","df['Age'] = df['Age'].fillna(df['Age'].mode())","df['Age'] = df['Age'].fillna(df['Age'].min())","df['Age'] = df['Age'].fillna(df['Age'].median())","Pandas dataframe","shiberas"),

 ("According to the AI Index Report 2025, which sector saw the highest AI private investment in 2024?","Healthcare applications","Data management and processing","Transportation systems","Retail and e-commerce overall","Data management and processing","AI industry trends","AI Index Report 2025"),

 ("What was the approximate total global AI private investment in 2024 according to the AI Index Report?","$50 billion","$97.2 billion","$150 billion","$25 billion","$97.2 billion","AI industry trends","AI Index Report 2025"),

 ("Which country led in AI private investment in 2024? according to the 2025 AI Index Report","China overall","United Kingdom","United States","Germany total","United States","AI industry trends","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what percentage of companies reported adopting at least one AI capability?","25 percent","42 percent","55 percent","72 percent","55 percent","AI adoption","AI Index Report 2025"),

 ("What is the main reason companies cited for not adopting AI according to the AI Index Report 2025?","Too expensive overall","Lack of skilled personnel","No clear business case","Regulatory concerns","Lack of skilled personnel","AI adoption","AI Index Report 2025"),

 ("According to the AI Index Report 2025, which AI application area saw the most significant growth in 2024?","Robotics applications","Computer vision systems","Natural language processing","Autonomous vehicles overall","Natural language processing","AI applications","AI Index Report 2025"),

 ("What trend did the AI Index Report 2025 identify regarding AI model training costs?","Decreasing significantly","Increasing exponentially","Remaining stable overall","Becoming unpredictable","Increasing exponentially","AI development","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what is the primary ethical concern about AI in 2024?","Cost of deployment infrastructure","Bias and fairness in algorithms","Speed of processing data","Energy consumption levels","Bias and fairness in algorithms","AI ethics","AI Index Report 2025"),

 ("Which region showed the fastest growth in AI research publications according to the AI Index Report 2025?","North America","Europe region","Asia region","South America","Asia region","AI research","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what percentage of AI PhD graduates in the US go into industry rather than academia?","35 percent of graduates","50 percent of graduates","65 percent of graduates","80 percent of graduates","65 percent of graduates","AI workforce","AI Index Report 2025"),

 ("What does the AI Index Report 2025 say about AI's impact on job displacement?","Minimal impact observed so far in the workforce overall","Significant displacement in manufacturing and routine cognitive tasks","Only affects low-skill jobs primarily in service sectors","No measurable impact yet detected in any sector nationwide","Significant displacement in manufacturing and routine cognitive tasks","AI impact","AI Index Report 2025"),

 ("According to the AI Index Report 2025, which application of AI in transportation saw the most investment in 2024?","Traffic management systems","Autonomous vehicles technology","Route optimization algorithms","Predictive maintenance tools","Autonomous vehicles technology","AI applications","AI Index Report 2025"),

 ("What trend did the AI Index Report 2025 identify in AI regulatory frameworks globally?","Decreasing regulation overall worldwide trend","Increasing fragmentation and country-specific approaches","Complete harmonization worldwide trend overall","No significant changes observed anywhere globally","Increasing fragmentation and country-specific approaches","AI policy","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what is the estimated growth rate of the global AI market from 2024 to 2030?","10 percent annually on average","20 percent annually on average","37 percent annually on average","50 percent annually on average","37 percent annually on average","AI industry trends","AI Index Report 2025"),

 ("Which AI technique showed the most improvement in benchmark performance in 2024 according to the report?","Reinforcement learning","Large language models","Computer vision systems","Speech recognition tech","Large language models","AI development","AI Index Report 2025"),

 ("According to the AI Index Report, what percentage of AI systems deployed in production experienced some form of failure or incident?","15 percent of systems","28 percent of systems","45 percent of systems","60 percent of systems","28 percent of systems","AI reliability","AI Index Report 2025"),

 ("What does the AI Index Report 2025 identify as the biggest barrier to AI adoption in developing countries?","Lack of interest overall in AI technology","Infrastructure and connectivity limitations","Cultural resistance to technology adoption","Too many regulations in place domestically","Infrastructure and connectivity limitations","AI adoption","AI Index Report 2025"),

 ("According to the 2025 report, which industry sector has the highest AI adoption rate?","Healthcare industry sector","Finance and insurance industry","Retail industry sector","Manufacturing industry sector","Finance and insurance industry","AI adoption","AI Index Report 2025"),

 ("What trend did the AI Index Report identify regarding open-source AI models in 2024?","Declining in popularity overall trend","Significant increase in development and adoption","Remaining stable without change overall","Being replaced by proprietary models","Significant increase in development and adoption","AI development","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what is the primary driver of AI innovation?","Government funding programs","Private sector investment","Academic research overall","International collaboration","Private sector investment","AI development","AI Index Report 2025"),

 ("What does the report say about AI's energy consumption in 2024?","Decreasing due to efficiency gains","Growing concern due to training large models","Not significant enough to measure","Completely offset by renewable energy","Growing concern due to training large models","AI sustainability","AI Index Report 2025"),

 ("According to the AI Index Report 2025, which country has the most comprehensive AI strategy?","USA","China","Singapore","EU","China","AI policy","AI Index Report 2025"),

 ("What percentage of Fortune 500 companies have a dedicated AI strategy according to the AI Index Report 2025?","40 percent of companies overall","60 percent of companies overall","75 percent of companies overall","90 percent of companies overall","75 percent of companies overall","AI adoption","AI Index Report 2025"),

 ("According to the AI Index Report 2025, what is the average time to deploy an AI model to production in 2024?","1-3 months on average overall","4-6 months on average overall","7-12 months on average overall","Over 1 year on average overall","4-6 months on average overall","AI development","AI Index Report 2025"),

 ("What does the AI Index Report 2025 identify as the most promising emerging AI application?","AI for drug discovery","AI climate mitigation","AI education tech","AI cybersecurity","AI for drug discovery","AI applications","AI Index Report 2025"),

 ("What are the three main types of machine learning?","Supervised, Unsupervised, Reinforcement","Classification, Regression, Clustering","Neural, Decision trees, SVM methods","Deep, Shallow, Transfer learning","Supervised, Unsupervised, Reinforcement","ML fundamentals","studocu"),

 ("In supervised learning, what is required for training?","Both labeled input and output pairs","Only input data needed overall","Only output labels are needed","No data needed for supervised","Both labeled input and output pairs","ML fundamentals","studocu"),

 ("What is the main goal of unsupervised learning?","To predict future values accurately from patterns","To find patterns and structure in unlabeled data","To maximize rewards over time period iteratively","To classify data into known categories accurately","To find patterns and structure in unlabeled data","ML fundamentals","studocu"),

 ("What is the key characteristic of reinforcement learning?","Learning from labeled examples provided by experts","Finding clusters in data without supervision guidance","Learning through trial and error with rewards and penalties","Reducing dimensionality of high-dimensional feature spaces","Learning through trial and error with rewards and penalties","ML fundamentals","studocu"),



 ("What is the purpose of a validation set?","To train the model initially first before testing","To tune hyperparameters and prevent overfitting","To replace the test set entirely in the workflow","To label data manually by hand for training sets","To tune hyperparameters and prevent overfitting","ML concepts","studocu"),





 ("What is a confusion matrix used for?","To confuse the model during training process intentionally with adversarial examples","To evaluate classification model performance by showing true/false positives and negatives","To visualize training loss over time during optimization for monitoring convergence","To select features for modeling during preprocessing based on importance scores","To evaluate classification model performance by showing true/false positives and negatives","ML evaluation","studocu"),

 ("What does True Positive (TP) mean in a confusion matrix?","Model correctly predicted negative class","Model incorrectly predicted positive class","Model correctly predicted positive class","Model incorrectly predicted negative class","Model correctly predicted positive class","ML evaluation","studocu"),

 ("What does False Positive (FP) mean?","Correctly predicted positive class","Incorrectly predicted positive (Type I error)","Correctly predicted negative class","Incorrectly predicted negative class","Incorrectly predicted positive (Type I error)","ML evaluation","studocu"),

 ("What does False Negative (FN) mean?","Correctly predicted negative class in classification","Incorrectly predicted negative (Type II error)","Correctly predicted positive class in classification","Incorrectly predicted positive class in classification","Incorrectly predicted negative (Type II error)","ML evaluation","studocu"),

 ("What is gradient descent?","A clustering algorithm for data grouping tasks in unsupervised learning","An optimization algorithm that iteratively adjusts parameters to minimize loss","A classification method for labels in supervised tasks and predictions","A data preprocessing technique for cleaning datasets before training","An optimization algorithm that iteratively adjusts parameters to minimize loss","ML algorithms","studocu"),

 ("What is a learning rate in gradient descent?","The speed of data loading into memory for processing","The step size for parameter updates during optimization","The accuracy of the model on validation data overall","The number of epochs to run during training iterations","The step size for parameter updates during optimization","ML algorithms","studocu"),

 ("What happens if the learning rate is too high?","The algorithm may overshoot the minimum and fail to converge","Training is too slow overall for convergence to occur","The model becomes too accurate overall without issues","Nothing significant happens during training iterations","The algorithm may overshoot the minimum and fail to converge","ML algorithms","studocu"),

 ("What happens if the learning rate is too low?","Training is very slow and may get stuck in local minima","Perfect convergence is achieved every time guaranteed","Model overfits immediately to the data without warning","No training occurs at all during the process at all","Training is very slow and may get stuck in local minima","ML algorithms","studocu"),

 ("What is batch gradient descent?","Uses the entire dataset to compute gradient","Uses one sample at a time for updates only","Uses random samples only for gradient updates","Uses only test data for computations overall","Uses the entire dataset to compute gradient","ML algorithms","studocu"),

 ("What is stochastic gradient descent (SGD)?","Uses one training example at a time to update parameters","Uses entire dataset at once for all gradient computations","Uses validation set only for parameter update calculations","Uses multiple datasets together for better generalization","Uses one training example at a time to update parameters","ML algorithms","studocu"),

 ("What is mini-batch gradient descent?","Uses entire dataset at once for gradient","Uses small random batches of training data","Uses one sample only for each update","Uses test data only for computations","Uses small random batches of training data","ML algorithms","studocu"),

 ("What is the purpose of an activation function in neural networks?","To introduce non-linearity so the network can learn complex patterns","To slow down training processes for better convergence and stability","To reduce overfitting problems by adding regularization constraints","To normalize inputs for better numerical stability during training","To introduce non-linearity so the network can learn complex patterns","Neural networks","studocu"),

 ("What is the most common activation function for hidden layers?","Sigmoid function for non-linearity","ReLU (Rectified Linear Unit)","Linear function for simplicity","Softmax function for outputs","ReLU (Rectified Linear Unit)","Neural networks","studocu"),

 ("What activation function is typically used for binary classification output?","Sigmoid function","ReLU for hidden","Tanh for normal","Linear function","Sigmoid function","Neural networks","studocu"),

 ("What activation function is used for multi-class classification output?","Softmax function","Sigmoid for bin","ReLU for hidden","Tanh for normal","Softmax function","Neural networks","studocu"),

 ("What is backpropagation?","Algorithm for computing gradients and updating weights by propagating errors backward","Forward pass through network for making predictions on input data during inference","Data preprocessing step for normalizing features before training the neural network","Model evaluation metric for measuring performance on test dataset during validation","Algorithm for computing gradients and updating weights by propagating errors backward","Neural networks","studocu"),

 ("What is an epoch in neural network training?","One complete pass through the entire training dataset","One forward pass only through network for prediction","One weight update only during training optimization","One batch processed during training iteration cycle","One complete pass through the entire training dataset","Neural networks","studocu"),

 ("What is dropout in neural networks?","Regularization technique that randomly drops neurons during training to prevent overfitting","Removing bad data points from the dataset before training to improve data quality overall","Stopping training early when validation loss stops improving to save computational resources","Removing features from model to reduce complexity and improve generalization performance","Regularization technique that randomly drops neurons during training to prevent overfitting","Neural networks","studocu"),

 ("What is the vanishing gradient problem?","Gradients become very small in deep networks, making training difficult","Gradients explode during training process causing numerical instability","Gradients disappear completely during training requiring restarts","Gradients become negative values only preventing proper convergence","Gradients become very small in deep networks, making training difficult","Neural networks","studocu"),

 ("What is the main difference between AI and Machine Learning?","They are the same thing essentially","AI is the broader concept, ML is a subset focused on learning from data","ML is broader than AI overall","AI only works with images","AI is the broader concept, ML is a subset focused on learning from data","AI concepts","eksamen 2022"),

 ("What is the purpose of data normalization in machine learning?","To remove outliers from data","To scale features to similar ranges for better model performance","To add more data","To label the data manually","To scale features to similar ranges for better model performance","Data preprocessing","eksamen 2022"),

 ("Which metric would be most appropriate for imbalanced classification problems?","Accuracy alone","F1-score or precision-recall","Mean squared error","R-squared coefficient","F1-score or precision-recall","ML evaluation","eksamen 2022"),

 ("What is the curse of dimensionality?","Having too much data available overall","Performance degradation as the number of features increases","Having too few samples for training","Model training is too fast overall","Performance degradation as the number of features increases","ML concepts","eksamen 2022"),

 ("What is the purpose of PCA (Principal Component Analysis)?","To increase dimensions in data space","To reduce dimensionality while preserving variance","To classify data into categories overall","To cluster data into groups overall","To reduce dimensionality while preserving variance","ML techniques","eksamen 2022"),

 ("What does it mean when a model has high training accuracy but low test accuracy?","The model is underfitting the data overall","The model is overfitting the data","The model is perfect overall for task","The data is bad quality overall","The model is overfitting the data","ML concepts","eksamen 2022"),

 ("What are the three main components of AI according to the introduction lecture?","Hardware, Software, Data","Reasoning, Learning, Perception","Input, Processing, Output","Training, Testing, Deployment","Reasoning, Learning, Perception","AI fundamentals","Lecture Introduction-DAVE3625"),

 ("What is the main limitation of rule-based AI systems?","They require expensive hardware and infrastructure for deployment overall","They cannot handle uncertainty and require explicit programming for all scenarios","They are too slow for real-time processing applications and live systems","They use too much memory for practical applications in real environments","They cannot handle uncertainty and require explicit programming for all scenarios","AI limitations","Lecture Limitations-with-AI"),

 ("What is the main difference between strong AI and weak AI?","Strong AI has more accurate predictions than weak AI overall","Strong AI has general intelligence like humans, weak AI is task-specific","Strong AI processes information faster than weak AI in practice","Strong AI uses larger datasets than weak AI for training process","Strong AI has general intelligence like humans, weak AI is task-specific","AI concepts","Lecture Introduction-DAVE3625"),

 ("What is the primary goal of supervised learning?","To find hidden patterns in unlabeled data from datasets","To learn a mapping from inputs to outputs using labeled examples","To group similar items into clusters using algorithms","To maximize rewards through trial and error with feedback","To learn a mapping from inputs to outputs using labeled examples","ML fundamentals","Lecture MachineLearning"),

 ("In the context of machine learning, what is a feature?","The output variable that we're trying to predict from data","An individual measurable property or characteristic of the data","The algorithm used to train the model during learning process","The training process that optimizes the model for performance","An individual measurable property or characteristic of the data","ML fundamentals","Lecture MachineLearning"),

 ("What is the purpose of splitting data into training and test sets?","To save storage space by reducing data size","To evaluate model performance on unseen data","To speed up training by using smaller datasets","To reduce overfitting during training phase","To evaluate model performance on unseen data","ML fundamentals","Lecture MachineLearning"),

 ("What does the bias-variance tradeoff refer to?","Speed vs accuracy tradeoff in model performance","The balance between underfitting and overfitting","Training time vs testing time overall for models","Memory usage vs performance overall in algorithms","The balance between underfitting and overfitting","ML theory","Lecture MachineLearning-p2"),

 ("What is the main advantage of decision trees?","They always achieve the most accurate predictions","They are interpretable and easy to understand","They require the least computational resources","They work without any training data required","They are interpretable and easy to understand","ML algorithms","Lecture MachineLearning-p2"),

 ("What is ensemble learning?","Using multiple datasets from different sources","Combining multiple models to improve performance","Training one model multiple times on same data","Using multiple computers to speed up training","Combining multiple models to improve performance","ML techniques","Lecture MachineLearning-p2"),

 ("What is the purpose of the kernel trick in SVM?","To reduce training time for large datasets","To map data to higher dimensions for linear separation","To reduce the number of features in data","To normalize the data before classification","To map data to higher dimensions for linear separation","SVM","Lecture MachineLearning-p2"),

 ("What is a confusion matrix used to evaluate?","Regression models and continuous predictions","Classification model performance and accuracy","Clustering quality and cluster cohesion overall","Data quality and missing values in datasets","Classification model performance and accuracy","ML evaluation","Lecture MachineLearning-p2"),

 ("What is cross-validation primarily used for?","To increase dataset size through replication","To get a more reliable estimate of model performance","To speed up training by using less data","To visualize data patterns and relationships","To get a more reliable estimate of model performance","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the main purpose of regularization in machine learning?","To increase model complexity and flexibility","To prevent overfitting by penalizing complex models","To speed up training process significantly","To reduce data size for faster processing","To prevent overfitting by penalizing complex models","ML techniques","Lecture MachineLearning-p2"),

 ("What is the difference between L1 and L2 regularization?","There is no difference between them at all in practice","L1 can zero out coefficients (feature selection), L2 shrinks them","L2 can zero out coefficients, L1 shrinks them instead","Both zero out coefficients equally across all features","L1 can zero out coefficients (feature selection), L2 shrinks them","ML techniques","Lecture MachineLearning-p2"),

 ("What is the K in K-Nearest Neighbors (KNN)?","The number of features in the dataset","The number of nearest neighbors to consider","The number of classes in classification","The number of iterations to run overall","The number of nearest neighbors to consider","ML algorithms","Lecture MachineLearning-p3-1"),

 ("What is a hyperparameter in machine learning?","A parameter that is learned during training process","A parameter set before training that controls the learning process","The final output of the trained model after training","A type of activation function for neurons in network","A parameter set before training that controls the learning process","ML fundamentals","Lecture MachineLearning-p3-1"),

 ("What is grid search used for?","To visualize data in graphical format clearly","To systematically search for optimal hyperparameters","To clean data and remove outliers from dataset","To reduce dimensions in feature space overall","To systematically search for optimal hyperparameters","ML optimization","Lecture MachineLearning-p3-1"),

 ("What is the elbow method used for in K-means clustering?","To find the optimal K value","To measure classification score","To normalize data before step","To split data into train/test","To find the optimal K value","Clustering","Lecture MachineLearning-p4-unsupervised"),

 ("What is the main goal of dimensionality reduction?","To increase features for better accuracy in all predictions","To reduce the number of features while preserving important information","To improve accuracy directly without any data loss at all","To speed up data collection and storage processes overall","To reduce the number of features while preserving important information","ML techniques","Lecture MachineLearning-p4-unsupervised"),

 ("What is PCA (Principal Component Analysis) primarily used for?","Classification of data into categories overall","Dimensionality reduction by finding principal components","Clustering similar data points together overall","Regression analysis of continuous variables overall","Dimensionality reduction by finding principal components","ML techniques","Lecture MachineLearning-p4-unsupervised"),

 ("In unsupervised learning, what is the main difference from supervised learning?","No computer is used in the process","No labeled data is used for training","No training is needed at all overall","No testing is needed afterwards overall","No labeled data is used for training","ML fundamentals","Lecture MachineLearning-p4-unsupervised"),

 ("What is hierarchical clustering?","A single-step clustering algorithm only","A method that creates a tree of clusters","A supervised learning classification technique","A type of neural network architecture","A method that creates a tree of clusters","Clustering","Lecture MachineLearning-p4-unsupervised"),

 ("What is the Silhouette score used for?","To measure classification prediction accuracy","To evaluate clustering quality and cohesion","To measure regression prediction error overall","To select features for model training","To evaluate clustering quality and cohesion","Clustering","Lecture MachineLearning-p4-unsupervised"),

 ("What is anomaly detection used for?","To find typical patterns in data for further analysis","To identify unusual data points that don't fit normal patterns","To classify data into known categories using algorithms","To reduce dimensions in feature space for efficiency","To identify unusual data points that don't fit normal patterns","ML applications","Lecture MachineLearning-p4-unsupervised"),

 ("What does it mean when we say AI has a 'black box' problem?","AI always produces incorrect predictions overall","It's difficult to understand how AI makes decisions","AI algorithms are too simple to be useful overall","AI requires expensive hardware to operate properly","It's difficult to understand how AI makes decisions","AI limitations","Lecture Limitations-with-AI"),

 ("What is algorithmic bias in AI?","Systematic errors from overly accurate AI predictions overall","Systematic errors in AI systems due to biased training data or design","Systematic errors from AI processing data too slowly overall","Systematic errors from excessive AI memory usage during training","Systematic errors in AI systems due to biased training data or design","AI ethics","Lecture Limitations-with-AI"),

 ("What is meant by AI explainability or interpretability?","The ability to make AI process data faster overall for systems","The ability to understand and explain how an AI system makes decisions","The ability to reduce AI operational costs overall for organizations","The ability to improve AI prediction accuracy overall in models","The ability to understand and explain how an AI system makes decisions","AI concepts","Lecture Limitations-with-AI"),

 ("What is the main concern with AI systems making critical decisions in healthcare or justice?","The cost of implementation overall for organizations","Lack of transparency and potential for bias affecting human lives","The speed of decision making process in real time","The storage requirements for data overall in systems","Lack of transparency and potential for bias affecting human lives","AI ethics","Lecture Limitations-with-AI"),

 ("What is data poisoning in the context of AI security?","Using too much data in training process overall","Deliberately manipulating training data to compromise the model","Deleting data from the database system entirely","Encrypting data for security purposes in storage","Deliberately manipulating training data to compromise the model","AI security","Lecture Limitations-with-AI"),

 ("What is adversarial attack in AI?","Training with more data samples overall","Crafting inputs designed to fool the AI model","Using better hardware for training model","Improving the algorithm's efficiency overall","Crafting inputs designed to fool the AI model","AI security","Lecture Limitations-with-AI"),

 ("What is the main energy concern with large AI models?","They use too little energy overall for their performance","Training large models requires massive computational resources and energy","They only work on batteries overall which limits their usage","They can't be powered properly overall in most environments","Training large models requires massive computational resources and energy","AI sustainability","Lecture Limitations-with-AI"),

 ("What is transfer learning in machine learning?","Transferring data between different computers in network","Using knowledge from one task to improve learning on a related task","Moving models between different servers overall in cloud","Translating between different languages overall in systems","Using knowledge from one task to improve learning on a related task","ML techniques","Lecture MachineLearning-p3-1"),

 ("What is the difference between classification and regression?","There is no difference between them at all in practice or theory","Classification predicts discrete categories, regression predicts continuous values","Regression predicts categories, classification predicts continuous values","Both predict the same types of outputs for all problems and scenarios","Classification predicts discrete categories, regression predicts continuous values","ML fundamentals","Lecture MachineLearning"),

 ("What is a neural network layer?","A physical component of the computer hardware","A collection of neurons that process inputs together","A type of data format for storage purposes","A training algorithm for optimization purposes","A collection of neurons that process inputs together","Neural networks","Lecture MachineLearning-p3-1"),



 ("What is the vanishing gradient problem in deep learning?","Gradients becoming too large during training process","Gradients becoming too small in early layers, preventing learning","Gradients disappearing completely from network layers","Loss function increasing during training continuously","Gradients becoming too small in early layers, preventing learning","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is meant by model deployment in machine learning?","Training the model on data initially first for predictions","Putting the trained model into production for real-world use","Collecting data for the model training process and analysis","Testing the model's accuracy on data overall for validation","Putting the trained model into production for real-world use","ML lifecycle","Lecture MachineLearning"),

 ("What is A/B testing in the context of ML deployment?","Testing two different datasets overall in parallel comparison","Comparing two model versions with real users to see which performs better","Testing on A grade vs B grade overall results for evaluation","Testing accuracy vs bias metrics overall in model performance","Comparing two model versions with real users to see which performs better","ML deployment","Lecture MachineLearning"),

 ("What is model monitoring in production?","Ignoring the model after deployment step completely always","Continuously tracking model performance to detect degradation or issues","Only checking once per month overall schedule for maintenance","Manual testing every few months only occasionally when needed","Continuously tracking model performance to detect degradation or issues","ML deployment","Lecture MachineLearning"),

 ("What is concept drift in machine learning?","Model getting better over time overall continuously in performance","When the statistical properties of the target variable change over time","Model staying the same forever overall consistently without changes","Training getting faster over time overall automatically with updates","When the statistical properties of the target variable change over time","ML deployment","Lecture MachineLearning"),



 ("What is the purpose of data augmentation?","To delete unnecessary data from dataset completely for cleaning","To artificially increase training data by creating modified versions","To compress data for storage purposes efficiently in databases","To visualize data patterns graphically for further analysis","To artificially increase training data by creating modified versions","Data preprocessing","Lecture MachineLearning"),

 ("What is imbalanced data in classification?","Equal class distribution across all classes evenly","When one class has significantly more samples than others","All classes are missing from the dataset completely","Data is corrupted and cannot be used for training","When one class has significantly more samples than others","Data issues","Lecture MachineLearning-p2"),

 ("What technique can help with imbalanced datasets?","Ignore the problem completely always without action","Oversampling minority class or undersampling majority class","Delete all data from the dataset completely always","Use only one class for predictions in the model","Oversampling minority class or undersampling majority class","Data preprocessing","Lecture MachineLearning-p2"),

 ("What is SMOTE in machine learning?","A type of neural network architecture design for deep learning","Synthetic Minority Oversampling Technique for handling imbalanced data","A clustering algorithm for grouping data points into categories","A regularization method for preventing overfitting issues in models","Synthetic Minority Oversampling Technique for handling imbalanced data","Data preprocessing","Lecture MachineLearning-p2"),

 ("What is the ROC curve's x-axis and y-axis?","Precision and Recall for the model results","False Positive Rate and True Positive Rate","Accuracy and Loss for the model performance","Bias and Variance for the model tradeoff","False Positive Rate and True Positive Rate","ML evaluation","Lecture MachineLearning-p2"),

 ("What does an AUC (Area Under Curve) of 0.5 indicate?","Perfect model performance overall","Random guessing performance","Worst possible performance","Good performance on average","Random guessing performance","ML evaluation","Lecture MachineLearning-p2"),

 ("What does an AUC close to 1.0 indicate?","Excellent model performance","Poor model overall results","Random model performance","Biased model performance","Excellent model performance","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the purpose of max pooling in CNNs?","To increase image size and resolution for better detail","To reduce spatial dimensions while retaining important features","To add more layers to the network architecture design","To train faster with reduced computational costs overall","To reduce spatial dimensions while retaining important features","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is a convolutional layer in CNNs?","A fully connected layer connecting all neurons together","A layer that applies filters to detect features in images","An output layer for final predictions in network","A normalization layer for data scaling in pipeline","A layer that applies filters to detect features in images","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the main advantage of CNNs for image processing?","They are faster than other algorithms in training overall","They can automatically learn spatial hierarchies of features","They use less memory during model execution overall","They require less data for effective training overall","They can automatically learn spatial hierarchies of features","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is a recurrent neural network (RNN) primarily used for?","Image classification tasks primarily","Sequential data like text or time series","Clustering algorithms for grouping data","Dimensionality reduction techniques","Sequential data like text or time series","Neural networks","Lecture MachineLearning-p3-1"),

 ("What problem do LSTMs solve compared to basic RNNs?","They are faster overall in training and inference","They better handle long-term dependencies in sequences","They use less memory during computation processes","They are simpler overall in architecture design","They better handle long-term dependencies in sequences","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of attention mechanism in neural networks?","To make training faster and more efficient overall","To allow the model to focus on relevant parts of the input","To reduce parameters in the network architecture","To normalize data before processing in pipeline","To allow the model to focus on relevant parts of the input","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is a loss function in machine learning?","A function that measures the difference between predictions and actual values","A function that always returns zero value for output consistently overall","A function that adds features to dataset for training and preprocessing","A function that removes outliers from data automatically during cleaning","A function that measures the difference between predictions and actual values","ML fundamentals","Lecture MachineLearning"),

 ("What is mean squared error (MSE) typically used for?","Regression problems","Classification","Clustering tasks","Dimensionality","Regression problems","ML evaluation","Lecture MachineLearning"),

 ("What is cross-entropy loss typically used for?","Classification problems","Regression tasks overall","Clustering tasks mainly","Data preprocessing steps","Classification problems","ML evaluation","Lecture MachineLearning-p2"),

 ("What is early stopping in neural network training?","Starting training early in the process beforehand","Stopping training when validation performance stops improving","Stopping after one epoch of training immediately","Never stopping training at any point in process","Stopping training when validation performance stops improving","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of a learning rate scheduler?","To maintain constant learning rate throughout training","To adjust learning rate during training for better convergence","To increase learning rate only during training process","To remove learning rate entirely from the model overall","To adjust learning rate during training for better convergence","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is batch size in neural network training?","Total dataset size available for training process","Number of samples processed before updating model parameters","Number of epochs to run during training procedure","Number of layers in network architecture structure","Number of samples processed before updating model parameters","Neural networks","Lecture MachineLearning-p3-1"),

 ("What happens if batch size is too large?","Training is always better overall with larger batches","May not fit in memory and may lead to poor generalization","Training is faster only without other benefits overall","Model is more accurate overall with larger batches always","May not fit in memory and may lead to poor generalization","Neural networks","Lecture MachineLearning-p3-1"),

 ("What happens if batch size is too small?","Perfect training results are achieved","Training becomes noisy and slow","Training is optimal for convergence","No training occurs at all in practice","Training becomes noisy and slow","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of a validation set distinct from test set?","No specific purpose in the workflow at all here","To tune hyperparameters without touching the test set","To replace training set entirely in process overall","To reduce data size only for efficiency purposes","To tune hyperparameters without touching the test set","ML fundamentals","Lecture MachineLearning"),

 ("What is stratified sampling in train/test split?","Random sampling only without structure","Maintaining class proportions in splits","Taking only one class for training","Taking all data together in one set","Maintaining class proportions in splits","Data preprocessing","Lecture MachineLearning"),

 ("What is the difference between parameters and hyperparameters?","Parameters are learned during training, hyperparameters are set before","No difference at all between the two concepts in machine learning","Hyperparameters learned during training, parameters set manually","Both are the same thing in machine learning practice overall","Parameters are learned during training, hyperparameters are set before","ML fundamentals","Lecture MachineLearning"),

 ("What is the purpose of momentum in gradient descent?","To slow down training process intentionally overall","To accelerate convergence by accumulating gradients","To increase loss function during training process","To remove features entirely from the model overall","To accelerate convergence by accumulating gradients","Optimization","Lecture MachineLearning-p2"),

 ("What is Adam optimizer?","A type of neural network architecture design overall","An adaptive learning rate optimization algorithm","A loss function type used in training process","A regularization technique method for models overall","An adaptive learning rate optimization algorithm","Optimization","Lecture MachineLearning-p3-1"),

 ("What is the main idea behind ensemble methods like Random Forest?","Use one tree model for all predictions in system","Combine multiple models to reduce variance and improve accuracy","Use only neural networks for classification tasks","Avoid decision trees entirely in the approach overall","Combine multiple models to reduce variance and improve accuracy","ML algorithms","Lecture MachineLearning-p2"),

 ("What is bagging in ensemble learning?","Training one model only for predictions overall","Training multiple models on different random subsets of data","Using bags for storage only in memory systems","Removing data points entirely from dataset overall","Training multiple models on different random subsets of data","ML techniques","Lecture MachineLearning-p2"),

 ("What is boosting in ensemble learning?","Sequentially training models where each focuses on previous errors","Random combination of models without structure or organization","Training in parallel only for efficiency and speed purposes","Using only strong learners in the ensemble without weak ones","Sequentially training models where each focuses on previous errors","ML techniques","Lecture MachineLearning-p2"),

 ("What is the difference between bagging and boosting?","Bagging trains in parallel, boosting trains sequentially focusing on errors","No difference at all between the two approaches in ensemble methods","Boosting trains in parallel while bagging trains sequentially overall","Both are identical methods with same implementation in practice","Bagging trains in parallel, boosting trains sequentially focusing on errors","ML techniques","Lecture MachineLearning-p2"),

 ("What is the purpose of dropout rate in neural networks?","To keep all neurons active during training process","To specify the fraction of neurons to randomly drop during training","To add more layers overall to the network structure","To reduce epochs needed for convergence in training","To specify the fraction of neurons to randomly drop during training","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is weight initialization in neural networks?","Setting all weights to zero at start initially","Setting initial weights before training begins","Final weight values only after training completes","Removing weights entirely from the network overall","Setting initial weights before training begins","Neural networks","Lecture MachineLearning-p3-1"),

 ("Why is proper weight initialization important?","It's not important at all for training process","Poor initialization can lead to vanishing/exploding gradients","It only affects speed of training convergence overall","It reduces accuracy overall in the final model results","Poor initialization can lead to vanishing/exploding gradients","Neural networks","Lecture MachineLearning-p3-1"),

 ("What is the purpose of the softmax function?","To make training harder overall for better results overall","To convert outputs to probability distribution for multi-class classification","To normalize inputs only before processing data in pipeline","To reduce dimensions available in the output layer structure","To convert outputs to probability distribution for multi-class classification","Neural networks","Lecture MachineLearning-p2"),



 ("What does high precision but low recall indicate?","Model is perfect overall in all predictions everywhere possible with no errors at all","Model is conservative, misses many positives but rarely wrong when it predicts positive","Model is random predictions without pattern or structure at all in any systematic way","Model always predicts positive class for all inputs everywhere without discrimination","Model is conservative, misses many positives but rarely wrong when it predicts positive","ML evaluation","Lecture MachineLearning-p2"),

 ("What does high recall but low precision indicate?","Model is perfect overall in all predictions everywhere","Model catches most positives but has many false alarms","Model is conservative in predictions it makes overall","Model predicts nothing at all for any inputs overall","Model catches most positives but has many false alarms","ML evaluation","Lecture MachineLearning-p2"),



 ("What is a true negative (TN) in classification?","Correctly predicted positive class","Correctly predicted negative class","Incorrectly predicted positive class","Incorrectly predicted negative class","Correctly predicted negative class","ML evaluation","Lecture MachineLearning-p2"),

 ("What is specificity in classification?","True positive rate overall across all classes","True Negative divided by all actual negatives","True Positive divided by all actual positives","False positive rate overall across all classes","True Negative divided by all actual negatives","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the purpose of standardization in data preprocessing?","To remove outliers completely from the dataset","To scale features to have mean 0 and standard deviation 1","To delete data entirely from the database system","To add features only to expand the dataset size","To scale features to have mean 0 and standard deviation 1","Data preprocessing","Lecture MachineLearning"),

 ("What is the difference between standardization and normalization?","No difference at all between the two methods for scaling data","Standardization uses mean and std, normalization scales to a range like 0-1","Normalization uses mean and std values for scaling in preprocessing","Both use the same formula for data transformation in all cases","Standardization uses mean and std, normalization scales to a range like 0-1","Data preprocessing","Lecture MachineLearning"),

 ("What is label encoding?","Encoding images into data format for storage","Converting categorical labels to numerical values","Removing labels completely from the dataset","Adding more labels unnecessarily to features","Converting categorical labels to numerical values","Data preprocessing","Lecture MachineLearning"),

 ("What is the problem with label encoding for nominal categories?","No problem at all with the method for categories","It introduces ordinal relationships where none exist","It's too slow overall for processing large datasets","It uses too much memory for storage in systems","It introduces ordinal relationships where none exist","Data preprocessing","Lecture MachineLearning"),

 ("What is gradient boosting?","Random combination of models without structure or purpose for performance","Ensemble method building models sequentially to correct previous errors using gradients","Training one model only for the task without combining multiple models","Removing gradients entirely from process to simplify computational requirements","Ensemble method building models sequentially to correct previous errors using gradients","ML algorithms","Lecture MachineLearning-p2"),

 ("What is XGBoost?","A neural network architecture design overall","An optimized implementation of gradient boosting","A clustering algorithm type for grouping data","A data structure format for storage purposes","An optimized implementation of gradient boosting","ML algorithms","Lecture MachineLearning-p2"),

 ("What is feature importance in tree-based models?","Random values only without meaning here for analysis","Measure of how useful each feature is for making predictions","Always equal for all features in model by design","Not measurable at all in practice for tree models","Measure of how useful each feature is for making predictions","ML algorithms","Lecture MachineLearning-p2"),

 ("What is the purpose of pruning in decision trees?","To make trees larger overall in structure","To reduce tree size and prevent overfitting","To add more branches overall to the tree","To remove all leaves completely from tree","To reduce tree size and prevent overfitting","ML algorithms","Lecture MachineLearning-p2"),

 ("What is Gini impurity used for?","Classification accuracy only for evaluation of model performance","Measuring how often a randomly chosen element would be incorrectly labeled","Regression error only for predictions of continuous values overall","Data cleaning tasks for preprocessing and preparation of datasets","Measuring how often a randomly chosen element would be incorrectly labeled","ML algorithms","Lecture MachineLearning-p2"),

 ("What is information gain in decision trees?","Loss function only for optimization of models","Decrease in entropy after splitting on an attribute","Increase in complexity of the model architecture","Data augmentation method for training on datasets","Decrease in entropy after splitting on an attribute","ML algorithms","Lecture MachineLearning-p2"),

 ("What is K-fold cross-validation?","Using K models overall for predictions and ensemble results","Splitting data into K parts, training K times with different validation sets","Using K features only for training the model on reduced dataset","Training for K epochs only in total for computational efficiency","Splitting data into K parts, training K times with different validation sets","ML evaluation","Lecture MachineLearning-p2"),

 ("What is leave-one-out cross-validation?","Remove one feature only from dataset overall","K-fold CV where K equals the number of samples","Remove one class only from dataset completely","Use only one sample for training the model","K-fold CV where K equals the number of samples","ML evaluation","Lecture MachineLearning-p2"),

 ("What is the purpose of data splitting in machine learning?","To delete data completely from the system for cleanup","To create independent sets for training, validation, and testing","To increase data size overall for better model performance","To compress data efficiently for storage and memory usage","To create independent sets for training, validation, and testing","ML fundamentals","Lecture MachineLearning"),

 ("What is the No Free Lunch theorem in machine learning?","All algorithms are free to use always","No algorithm is universally best for all problems","All algorithms perform equally well overall","Free algorithms are best for all tasks","No algorithm is universally best for all problems","ML theory","Lecture MachineLearning"),

 ("What is the formula for a simple linear regression model?","weight = b1 + b0 value","weight = b1 × height + b0","weight = height / b1 constant","weight = b0 - b1 value","weight = b1 × height + b0","Regression","DIKU 004 - Supervised Machine Learning"),

 ("In linear regression, what does b1 represent?","The intercept value","The slope of the line","The error term value","The prediction score","The slope of the line","Regression","DIKU 004 - Supervised Machine Learning"),

 ("In linear regression, what does b0 represent?","The slope of line","The intercept value","The correlation coef","The standard dev val","The intercept value","Regression","DIKU 004 - Supervised Machine Learning"),

 ("What does R² (R-squared) measure in regression?","The slope steepness of the line","The quality of fit (closer to 1 is better)","The number of data points in set","The training speed of the model","The quality of fit (closer to 1 is better)","Regression","DIKU 004 - Supervised Machine Learning"),

 ("What is the best-fit line in regression?","The line that passes through all points on graph","The line that minimizes error between prediction and data","The line with the steepest slope overall on the plot","The line with zero intercept in model for simplicity","The line that minimizes error between prediction and data","Regression","DIKU 004 - Supervised Machine Learning"),

 ("What is regression used for in supervised learning?","Only for categorizing data into classes for sorting","Prediction, interpolation, and inference of continuous values","Only for clustering data into groups using algorithms","Only for dimensionality reduction tasks in preprocessing","Prediction, interpolation, and inference of continuous values","Regression","DIKU 004 - Supervised Machine Learning"),

 ("What is classification in supervised learning?","Dividing data into categories based on known labels","Predicting continuous numerical values for regression","Finding patterns in unlabeled data sets for clustering","Reducing the number of features overall for efficiency","Dividing data into categories based on known labels","Classification","DIKU 004 - Supervised Machine Learning"),

 ("What does classification require for training?","Labeled data (training set) and testing data","Only unlabeled data for the task at hand","Only test data for evaluation and metrics","No data at all for the process to work","Labeled data (training set) and testing data","Classification","DIKU 004 - Supervised Machine Learning"),

 ("What tool is used to evaluate classification model performance?","Confusion matrix","Regression line","Scatter plots","Histograms only","Confusion matrix","Classification","DIKU 004 - Supervised Machine Learning"),

 ("Classification boundaries can be which of the following?","Linear or complex (curved, multidimensional)","Only linear boundaries in space for simple","Only circular boundaries in plane for data","Only straight vertical lines always in plots","Linear or complex (curved, multidimensional)","Classification","DIKU 004 - Supervised Machine Learning"),

 ("What is core business data?","Data most directly tied to company's value-generating activities","Any random data collected overall from all possible sources","Only financial statements for company accounting purposes","Only employee records for company human resources department","Data most directly tied to company's value-generating activities","Business data","DIKU 004 - Supervised Machine Learning"),

 ("What characterizes core business data?","High dollar density with measurable financial impact per record","Low dollar density overall for records without much value","No connection to profit for business operations whatsoever","Only historical data from the past with no current relevance","High dollar density with measurable financial impact per record","Business data","DIKU 004 - Supervised Machine Learning"),

 ("What percentage of enterprise data is typically structured?","Around 20% of data","Around 80% total","Around 50% total","Around 5% total","Around 20% of data","Data types","DIKU 004 - Supervised Machine Learning"),

 ("What percentage of enterprise data is typically unstructured?","Around 80% of data","Around 20% total","Around 50% total","Around 10% total","Around 80% of data","Data types","DIKU 004 - Supervised Machine Learning"),

 ("Which type of data is easier to manage?","Structured data (tabular)","Unstructured data overall","Both equally difficult","Neither can be managed","Structured data (tabular)","Data types","DIKU 004 - Supervised Machine Learning"),

 ("What type of data holds richer information but requires AI to extract value?","Unstructured data (images, audio, video, text)","Structured data only in tabular form for analysis","Numerical data only in spreadsheets for computation","Spreadsheet data in tabular format for easy access","Unstructured data (images, audio, video, text)","Data types","DIKU 004 - Supervised Machine Learning"),

 ("According to the 5 V's of Big Data, what does 'Volume' refer to?","Massive amounts of data (terabytes to petabytes)","Speed of data generation in systems continuously","Data accuracy and quality in storage for reliability","Data usefulness overall for analysis and decisions","Massive amounts of data (terabytes to petabytes)","Big Data","DIKU 004 - Supervised Machine Learning"),

 ("According to the 5 V's of Big Data, what does 'Velocity' refer to?","Data is generated and processed rapidly","Amount of data overall in storage systems","Data accuracy and quality measures overall","Data variety types in systems and databases","Data is generated and processed rapidly","Big Data","DIKU 004 - Supervised Machine Learning"),

 ("According to the 5 V's of Big Data, what does 'Variety' refer to?","Structured, semi-structured, and unstructured data types","Only structured data in databases for storage and queries","Only numerical data in spreadsheets for calculations overall","Only text data in documents for processing and analysis","Structured, semi-structured, and unstructured data types","Big Data","DIKU 004 - Supervised Machine Learning"),

 ("According to the 5 V's of Big Data, what does 'Veracity' refer to?","Data accuracy and reliability","Data volume overall in storage","Data speed overall in processing","Data storage types in systems","Data accuracy and reliability","Big Data","DIKU 004 - Supervised Machine Learning"),

 ("According to the 5 V's of Big Data, what does 'Value' refer to?","Data size overall in storage","Data usefulness for decision-making","Data speed overall in systems","Data format types for processing","Data usefulness for decision-making","Big Data","DIKU 004 - Supervised Machine Learning"),

 ("When did digital storage become inexpensive, marking the beginning of the digital age?","Around 1990 era","Around 2008 era","Around 2015 era","Around 2000 era","Around 2008 era","AI history","DIKU 004 - Supervised Machine Learning"),

 ("What costs businesses over $3.1 trillion per year?","Poor data quality","Hardware costs","Training costs","Storage costs","Poor data quality","Data quality","DIKU 004 - Supervised Machine Learning"),

 ("What is symbolic AI also known as?","GOFAI (Good Old-Fashioned AI)","Deep Learning methods overall","Neural networks architectures","Genetic algorithms approaches","GOFAI (Good Old-Fashioned AI)","AI types","DIKU 004 - Supervised Machine Learning"),

 ("What does symbolic AI use to represent knowledge?","Symbols (nouns) and relations (verbs/adjectives)","Only numbers available for computations and math","Only images available for visual processing tasks","Only text available for natural language operations","Symbols (nouns) and relations (verbs/adjectives)","AI types","DIKU 004 - Supervised Machine Learning"),

 ("What logic operations does symbolic AI use?","AND, OR, NOT","Multiply ops","Addition ops","Division ops","AND, OR, NOT","AI types","DIKU 004 - Supervised Machine Learning"),

 ("What is fuzzy logic used for?","Handling uncertainty with degrees of truth (values between 0 and 1)","Binary true/false only without any intermediate values for decisions","Only integer values without any decimal or fractional representation","Only text processing for natural language understanding applications","Handling uncertainty with degrees of truth (values between 0 and 1)","AI types","DIKU 004 - Supervised Machine Learning"),

 ("What is an example application of fuzzy logic?","Home appliances and subway control systems","Only image recognition for computer vision","Only text analysis for language processing","Only speech recognition for audio systems","Home appliances and subway control systems","AI types","DIKU 004 - Supervised Machine Learning"),

 ("According to George Box, what is true about models?","All models are perfect in predictions","All models are wrong, but some are useful","All models are useless for real tasks","Models are always accurate in practice","All models are wrong, but some are useful","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("What does learning mean in the context of machine learning?","Memorizing all data completely","Behavioral change from experience","Deleting old data only overall","Increasing storage capacity","Behavioral change from experience","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("How do machines learn?","By copying humans directly","By building models from data","By guessing randomly always","By following fixed rules only","By building models from data","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("What are the main types of models?","Only mathematical models used in analysis","Descriptive, predictive, mechanistic, and normative","Only statistical models for predictions","Only graphical models for visualization","Descriptive, predictive, mechanistic, and normative","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("What is a descriptive model?","Predicts future events from data","Represents current state","Optimizes strategies overall for outcomes","Shows causal processes in detail","Represents current state","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("What is a predictive model?","Shows current state only at time","Shows trends over time","Shows optimal strategies for outcomes","Shows logical relations in structure","Shows trends over time","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("What is a mechanistic model?","Shows current state overall snapshot","Shows causal processes","Shows optimal strategies for decisions","Shows trends only over time","Shows causal processes","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("What is a normative model?","Shows current state overall snapshot","Shows optimal strategies","Shows causal processes in detail","Shows trends only over time","Shows optimal strategies","ML theory","DIKU 004 - Supervised Machine Learning"),

 ("In supervised learning, what does 'supervised' refer to?","The algorithm supervises itself automatically","Learning from labeled examples with known outputs","No human involvement in the process at all","Random learning process without guidance","Learning from labeled examples with known outputs","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("What are the two main types of supervised learning problems?","Clustering and association","Regression and classification","Only neural networks overall","Only decision trees methods","Regression and classification","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("What type of output does regression predict?","Categories only in classification","Continuous numerical values","Binary only yes or no","Text only in NLP tasks","Continuous numerical values","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("What type of output does classification predict?","Continuous values for regression","Categorical outcomes","Only numbers in numeric form","Only images in computer vision","Categorical outcomes","ML fundamentals","DIKU 004 - Supervised Machine Learning"),

 ("How is supervised learning performance measured?","Only by speed of computation","By accuracy or error metrics","Only by cost of resources","Only by time to complete","By accuracy or error metrics","ML evaluation","DIKU 004 - Supervised Machine Learning"),

 ("What is machine learning according to DAVE3625?","Manual programming of all rules and explicit instructions for every possible scenario encountered","Application of AI that allows systems to automatically learn and improve from experience without explicit programming","Only statistical analysis of historical data patterns without any learning capabilities","Only data collection and storage without any processing or analysis of information gathered","Application of AI that allows systems to automatically learn and improve from experience without explicit programming","ML fundamentals","DAVE3625-MachineLearning1"),

 ("In the ML algorithm building process, what is the first step?","Test the model first for accuracy","Collect data initially","Train the model first on data","Deploy the model first to production","Collect data initially","ML fundamentals","DAVE3625-MachineLearning1"),

 ("How does a reinforcement learning agent learn?","Only from labeled examples in training set","By interacting with environment via trial and error","By clustering data together into groups","By reducing dimensions only for efficiency","By interacting with environment via trial and error","Reinforcement learning","DAVE3625-MachineLearning1"),

 ("What feedback system does reinforcement learning use?","No feedback at all during learning","Reward if correct, penalty if wrong","Only penalties given for errors","Only rewards given for success","Reward if correct, penalty if wrong","Reinforcement learning","DAVE3625-MachineLearning1"),

 ("What is the purpose of recommender systems?","To classify images only in computer vision applications overall","Suggest relevant items to users and predict products likely to interest them","Only for search engines to rank results and improve query matching","Only for social media platforms to suggest friends and connections","Suggest relevant items to users and predict products likely to interest them","Recommender systems","DAVE3625-MachineLearning1"),

 ("When should you use machine learning?","Always for every problem regardless of complexity or requirements","When rules are not explicitly known, but patterns can be inferred from data","Only for image processing tasks in computer vision applications","Only for text processing tasks in natural language understanding","When rules are not explicitly known, but patterns can be inferred from data","ML fundamentals","DAVE3625-MachineLearning1"),

 ("When was the Dartmouth Conference that founded AI?","1950 era in the early fifties","1956 era","1960 era in the early sixties","1970 era in the early seventies","1956 era","AI history","DIKU 002 - History of AI-p2"),

 ("Who were among the founding fathers of AI at Dartmouth?","Only John McCarthy from MIT who coined the term","John McCarthy, Marvin Minsky, Claude Shannon, and others","Only Alan Turing from Cambridge University in UK","Only Marvin Minsky from MIT who built SNARC","John McCarthy, Marvin Minsky, Claude Shannon, and others","AI history","DIKU 002 - History of AI-p2"),

 ("What was SNARC?","The first computer ever built for calculations in history","First neural network machine developed by Marvin Minsky in 1951","A programming language for early AI systems development","A database system for storing AI knowledge and information","First neural network machine developed by Marvin Minsky in 1951","AI history","DIKU 002 - History of AI-p2"),

 ("What was Logic Theorist designed to operate on?","Only numerical data types","Symbols rather than numbers","Only text data primarily","Only image data overall","Symbols rather than numbers","AI history","DIKU 002 - History of AI-p2"),

 ("Who developed the Perceptron?","Marvin Minsky","Frank Rosenblatt","John McCarthy","Claude Shannon","Frank Rosenblatt","AI history","DIKU 002 - History of AI-p2"),

 ("What was the Perceptron?","A programming language for artificial intelligence applications","An electronic device following biological principles, capable of learning","A database system for storing neural network information","A mechanical calculator using biological computation methods","An electronic device following biological principles, capable of learning","AI history","DIKU 002 - History of AI-p2"),

 ("What is the structure of Rosenblatt's Perceptron model?","inputs → outputs directly without any processing","inputs → weights → activation function → output","only activation function used in isolation","weights only without other processing components","inputs → weights → activation function → output","Neural networks","DIKU 002 - History of AI-p2"),

 ("Who developed early versions of deep learning models in the 1960s?","Frank Rosenblatt","Alexey G. Ivakhnenko","John McCarthy","Marvin Minsky","Alexey G. Ivakhnenko","AI history","DIKU 002 - History of AI-p2"),

 ("Which lambda function correctly creates a binary classification from grades (pass if >=10)?","df['passed'] = df['final_grade'].apply(lambda x: 1 if x >= 10 else 0)","df['passed'] = df['final_grade'].apply(lambda x: 1 if x > 10 else 0)","df['passed'] = df['final_grade'].apply(lambda x: True if x < 10 else False)","df['passed'] = df['final_grade'].apply(lambda x: 0 if x >= 10 else 1)","df['passed'] = df['final_grade'].apply(lambda x: 1 if x >= 10 else 0)","Lambda functions","Lab6"),

 ("Which code correctly applies StandardScaler to normalize features in scikit-learn?","scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_train)","scaler = StandardScaler(); X_scaled = scaler.transform(X_train)","scaler = StandardScaler(); X_scaled = scaler.fit(X_train)","scaler = StandardScaler(); X_scaled = scaler.normalize(X_train)","scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_train)","Feature scaling","Lab5"),

 ("What does np.where(df['quality'] >= 7, 1, 0) do?","Returns 1 where quality >= 7 and 0 otherwise","Returns 0 where quality >= 7 and 1 otherwise","Returns True where quality >= 7 and False otherwise","Returns quality value if >= 7, otherwise 0","Returns 1 where quality >= 7 and 0 otherwise","NumPy operations","Lab5"),

 ("Which lambda function filters rows where studytime > 2 in a pandas dataframe?","df_filtered = df[df['studytime'].apply(lambda x: x > 2)]","df_filtered = df[df['studytime'].map(lambda x: x > 2)]","df_filtered = df[df.apply(lambda x: x['studytime'] > 2, axis=1)]","df_filtered = df.filter(lambda x: x['studytime'] > 2)","df_filtered = df[df['studytime'].apply(lambda x: x > 2)]","Lambda functions","Lab6"),

 ("What does train_test_split(X, y, test_size=0.3, random_state=42) do?","Splits data 70% training / 30% testing with seed 42 for reproducibility","Splits data 30% training / 70% testing with seed 42 for reproducibility","Splits data 70% training / 30% testing with random splits each time","Splits data equally 50/50 with seed 42 for reproducibility","Splits data 70% training / 30% testing with seed 42 for reproducibility","Train-test split","Lab5"),

 ("Which code checks for missing values in each column of a pandas dataframe?","missing_values = df.isnull().sum()","missing_values = df.isna().count()","missing_values = df.null_count()","missing_values = df.missing().sum()","missing_values = df.isnull().sum()","Data cleaning","Lab5"),

 ("What does GridSearchCV(knn, param_grid, cv=5) do?","Tests parameter combinations with 5-fold cross-validation","Tests 5 different models with various grid parameters","Validates model 5 times on the same test set repeatedly","Splits data into 5 equal parts for training the model","Tests parameter combinations with 5-fold cross-validation","Hyperparameter tuning","Lab5"),

 ("Which lambda expression correctly categorizes ages into groups?","df['age_group'] = df['age'].apply(lambda x: 'young' if x < 20 else 'old')","df['age_group'] = df['age'].map(lambda x: 'young' if x < 20 else 'old')","df['age_group'] = df.apply(lambda x: 'young' if x['age'] < 20 else 'old')","df['age_group'] = lambda x: df['age'] < 20 ? 'young' : 'old'","df['age_group'] = df['age'].apply(lambda x: 'young' if x < 20 else 'old')","Lambda functions","Lab6"),

 ("What does df.select_dtypes(include=[np.number]).columns.tolist() return?","List of numerical column names from the dataframe","List of all column names including non-numerical types","List of numerical values from the first row of data","List of column types for all columns in dataframe","List of numerical column names from the dataframe","Pandas operations","Lab6"),

 ("Which code snippet correctly implements K-Nearest Neighbors with optimal K?","knn = KNeighborsClassifier(n_neighbors=5); knn.fit(X_train, y_train)","knn = KNN(neighbors=5); knn.train(X_train, y_train)","knn = KNeighborsClassifier(k=5); knn.fit(X_train, y_train)","knn = KNeighborsRegressor(n_neighbors=5); knn.fit(X_train, y_train)","knn = KNeighborsClassifier(n_neighbors=5); knn.fit(X_train, y_train)","KNN implementation","Lab5"),

 ("What does SVC(kernel='rbf') create compared to SVC(kernel='linear')?","Non-linear decision boundary vs linear hyperplane","Linear hyperplane vs non-linear decision boundary","Both create identical linear boundaries","Both create identical non-linear boundaries","Non-linear decision boundary vs linear hyperplane","SVM kernels","Lab5"),

 ("Which lambda function creates a new feature combining two columns?","df['total'] = df.apply(lambda row: row['A'] + row['B'], axis=1)","df['total'] = df.apply(lambda row: row['A'] + row['B'], axis=0)","df['total'] = df.map(lambda row: row['A'] + row['B'] together)","df['total'] = lambda row: df['A'] + df['B'] for all rows overall","df['total'] = df.apply(lambda row: row['A'] + row['B'], axis=1)","Lambda functions","Lab6"),

 ("What does df.rename(columns={'G1': 'period_1_grades'}, inplace=True) do?","Renames column 'G1' to 'period_1_grades' and modifies original dataframe","Creates new dataframe with renamed column 'G1' to 'period_1_grades'","Renames row 'G1' to 'period_1_grades' and modifies original dataframe","Renames all columns to 'period_1_grades' in original dataframe","Renames column 'G1' to 'period_1_grades' and modifies original dataframe","Pandas operations","Lab6"),

 ("Which code correctly creates a Random Forest classifier with 100 trees?","rf = RandomForestClassifier(n_estimators=100); rf.fit(X_train, y_train)","rf = RandomForest(trees=100); rf.fit(X_train, y_train)","rf = RandomForestClassifier(n_trees=100); rf.train(X_train, y_train)","rf = ForestClassifier(estimators=100); rf.fit(X_train, y_train)","rf = RandomForestClassifier(n_estimators=100); rf.fit(X_train, y_train)","Random Forest","Lab6"),

 ("What does DecisionTreeClassifier(max_depth=5) limit?","Maximum depth of tree to 5 levels","Maximum number of features to only 5","Maximum number of samples to only 5","Maximum number of branches to only 5","Maximum depth of tree to 5 levels","Decision Trees","Lab6"),

 ("Which lambda function correctly filters dataframe for absences <= 5?","filtered = df[df['absences'].apply(lambda x: x <= 5)]","filtered = df[df.apply(lambda x: x['absences'] <= 5)]","filtered = df.filter(lambda x: x['absences'] <= 5)","filtered = df[lambda x: df['absences'] <= 5]","filtered = df[df['absences'].apply(lambda x: x <= 5)]","Lambda functions","Lab6"),

 ("What does GaussianNB() assume about feature distributions?","Features follow Gaussian (normal) distribution within each class","Features follow uniform distribution across all classes","Features follow Poisson distribution within each class","Features follow exponential distribution across all classes","Features follow Gaussian (normal) distribution within each class","Naive Bayes","Lab6"),

 ("Which code snippet correctly calculates confusion matrix in scikit-learn?","cm = confusion_matrix(y_test, y_pred)","cm = confusion_matrix(y_pred, y_test)","cm = accuracy_score(y_test, y_pred)","cm = classification_report(y_test, y_pred)","cm = confusion_matrix(y_test, y_pred)","Model evaluation","Lab5"),

 ("What does df['quality'].value_counts() return?","Frequency count of each unique value in 'quality' column","Sum of all values in 'quality' column overall","Number of non-null values in 'quality' column","Statistical summary of 'quality' column values","Frequency count of each unique value in 'quality' column","Pandas operations","Lab5"),

 ("Which lambda expression creates age categories (child/teen/adult)?","df['category'] = df['age'].apply(lambda x: 'child' if x<13 else ('teen' if x<20 else 'adult'))","df['category'] = df['age'].map(lambda x: 'child' if x<13 else ('teen' if x<20 else 'adult'))","df['category'] = lambda x: 'child' if df['age']<13 else ('teen' if df['age']<20 else 'adult')","df['category'] = df.apply(lambda x: 'child' if x<13 else ('teen' if x<20 else 'adult'))","df['category'] = df['age'].apply(lambda x: 'child' if x<13 else ('teen' if x<20 else 'adult'))","Lambda functions","Lab6"),

 ("What does accuracy_score(y_test, y_pred) calculate?","Proportion of correct predictions out of total predictions","Sum of true positives and true negatives only","Difference between predicted and actual values","Average of precision and recall for the model","Proportion of correct predictions out of total predictions","Model evaluation","Lab5"),

 ("Which code creates a countplot to visualize class distribution?","sns.countplot(x='quality_binary', data=df)","sns.barplot(x='quality_binary', data=df)","sns.scatterplot(x='quality_binary', data=df)","sns.lineplot(x='quality_binary', data=df)","sns.countplot(x='quality_binary', data=df)","Data visualization","Lab5"),

 ("What does df.drop(['quality', 'quality_binary'], axis=1) do?","Removes columns 'quality' and 'quality_binary' from dataframe","Removes rows 'quality' and 'quality_binary' from dataframe","Removes all columns except 'quality' and 'quality_binary'","Removes all rows except 'quality' and 'quality_binary'","Removes columns 'quality' and 'quality_binary' from dataframe","Pandas operations","Lab5"),

 ("Which lambda function correctly converts Celsius to Fahrenheit?","df['fahrenheit'] = df['celsius'].apply(lambda x: (x * 9/5) + 32)","df['fahrenheit'] = df['celsius'].map(lambda x: (x * 9/5) + 32)","df['fahrenheit'] = df.apply(lambda x: (x['celsius'] * 9/5) + 32)","df['fahrenheit'] = lambda x: (df['celsius'] * 9/5) + 32","df['fahrenheit'] = df['celsius'].apply(lambda x: (x * 9/5) + 32)","Lambda functions","Lab6"),

 ("What does SVC(kernel='linear').decision_function(X_test) return?","Signed distance from samples to hyperplane","Probability predictions for each class result","Binary predictions zero or one results","Accuracy score of the model overall","Signed distance from samples to hyperplane","SVM operations","Lab5"),

 ("Which code correctly splits features and target variable?","X = df.drop(columns=['passed']); y = df['passed']","X = df.remove(['passed']); y = df['passed']","X = df.drop('passed'); y = df.select('passed')","X = df.exclude(['passed']); y = df.get('passed')","X = df.drop(columns=['passed']); y = df['passed']","Data preparation","Lab6"),

 ("What does roc_curve(y_test, y_pred_proba) calculate?","False positive rate and true positive rate at various thresholds","Only accuracy at different thresholds across the dataset","Only precision at different thresholds across predictions","Only recall at different thresholds across all predictions","False positive rate and true positive rate at various thresholds","Model evaluation","Lab5"),

 ("Which lambda filters students with failures > 0 OR absences > 10?","filtered = df[df.apply(lambda x: x['failures'] > 0 or x['absences'] > 10, axis=1)]","filtered = df[df['failures'].apply(lambda x: x > 0 or df['absences'] > 10)]","filtered = df.filter(lambda x: x['failures'] > 0 or x['absences'] > 10)","filtered = lambda x: df[df['failures'] > 0 or df['absences'] > 10]","filtered = df[df.apply(lambda x: x['failures'] > 0 or x['absences'] > 10, axis=1)]","Lambda functions","Lab6"),

 ("What does pd.read_csv('data/wine.csv', sep=';') do?","Reads CSV file using semicolon as delimiter","Reads CSV file using comma as delimiter","Reads CSV file using space as delimiter","Reads CSV file using tab as delimiter","Reads CSV file using semicolon as delimiter","Data loading","Lab5"),

 ("Which code correctly implements Gaussian Naive Bayes?","nb = GaussianNB(); nb.fit(X_train, y_train)","nb = NaiveBayes(); nb.fit(X_train, y_train)","nb = GaussianNB(); nb.train(X_train, y_train)","nb = BayesClassifier(); nb.fit(X_train, y_train)","nb = GaussianNB(); nb.fit(X_train, y_train)","Naive Bayes","Lab6"),

 ("What does df.describe() provide for numerical columns?","Statistical summary including mean, std, min, max, and quartiles","Only mean and median values for all statistical measures","Only count of non-null values in all the columns overall","Only maximum and minimum values in all columns of dataframe","Statistical summary including mean, std, min, max, and quartiles","Data exploration","Lab5"),

 ("Which lambda creates a grade trend feature (difference between periods)?","df['trend'] = df.apply(lambda x: x['period_2_grades'] - x['period_1_grades'], axis=1)","df['trend'] = df.apply(lambda x: x['period_2_grades'] - x['period_1_grades'], axis=0)","df['trend'] = df.map(lambda x: x['period_2_grades'] - x['period_1_grades'])","df['trend'] = lambda x: df['period_2_grades'] - df['period_1_grades']","df['trend'] = df.apply(lambda x: x['period_2_grades'] - x['period_1_grades'], axis=1)","Lambda functions","Lab6"),

 ("What parameter in KNeighborsClassifier determines the number of neighbors?","n_neighbors","k_value","num_neigh","neigh_count","n_neighbors","KNN parameters","Lab5"),

 ("Which code plots ROC curves for multiple models?","plt.plot(fpr, tpr, label='Model')","plt.scatter(fpr, tpr, label='Model')","plt.bar(fpr, tpr, label='Model')","plt.hist(fpr, tpr, label='Model')","plt.plot(fpr, tpr, label='Model')","Data visualization","Lab5"),

 ("What does df.info() display about a dataframe?","Column names, data types, non-null counts, and memory usage","Only column names in the dataframe and their order","Only data types for each column in the dataframe","Only non-null counts for all columns in dataframe","Column names, data types, non-null counts, and memory usage","Data exploration","Lab6"),

 ("Which lambda function correctly bins continuous age into categories?","df['age_bin'] = df['age'].apply(lambda x: '0-20' if x<=20 else ('21-40' if x<=40 else '41+'))","df['age_bin'] = df['age'].map(lambda x: '0-20' if x<=20 else ('21-40' if x<=40 else '41+'))","df['age_bin'] = lambda x: '0-20' if df['age']<=20 else ('21-40' if df['age']<=40 else '41+')","df['age_bin'] = df.apply(lambda x: '0-20' if x<=20 else ('21-40' if x<=40 else '41+'))","df['age_bin'] = df['age'].apply(lambda x: '0-20' if x<=20 else ('21-40' if x<=40 else '41+'))","Lambda functions","Lab6"),

 ("What does RandomForestClassifier(n_estimators=100, max_depth=10) create?","Random forest with 100 trees, each limited to depth 10","Random forest with 10 trees, each limited to depth 100","Decision tree with 100 nodes and depth 10","Neural network with 100 neurons and 10 layers","Random forest with 100 trees, each limited to depth 10","Random Forest","Lab6"),

 ("Which code correctly computes feature importance from Random Forest?","importances = rf.feature_importances_","importances = rf.get_feature_importance()","importances = rf.compute_importances()","importances = rf.importance_scores()","importances = rf.feature_importances_","Random Forest","Lab6"),

 ("What does classification_report(y_test, y_pred) provide?","Precision, recall, F1-score, and support for each class","Only accuracy score for the model across all data","Only confusion matrix for the results from model","Only ROC-AUC score for each class in the dataset","Precision, recall, F1-score, and support for each class","Model evaluation","Lab6"),

 ("Which lambda applies a discount based on quantity purchased?","df['price'] = df.apply(lambda x: x['base_price'] * 0.9 if x['qty'] > 10 else x['base_price'], axis=1)","df['price'] = df['base_price'].apply(lambda x: x * 0.9 if df['qty'] > 10 else x)","df['price'] = lambda x: df['base_price'] * 0.9 if df['qty'] > 10 else df['base_price']","df['price'] = df.map(lambda x: x['base_price'] * 0.9 if x['qty'] > 10 else x['base_price'])","df['price'] = df.apply(lambda x: x['base_price'] * 0.9 if x['qty'] > 10 else x['base_price'], axis=1)","Lambda functions","Lab6"),

 ("Why is StandardScaler important for KNN and SVM algorithms?","Features with larger scales dominate distance calculations without scaling","Scaling increases model accuracy by 100% in all cases for better results","Scaling is required by pandas operations for distance-based algorithms","Scaling reduces computation time significantly for all machine learning","Features with larger scales dominate distance calculations without scaling","Feature scaling","Lab5"),

 ("What does df.head(n) return?","First n rows of the dataframe","Last n rows of the dataframe","n random rows from dataframe","Summary statistics of n columns","First n rows of the dataframe","Pandas operations","Lab5"),

 ("Which lambda categorizes students by study time and absences?","df['risk'] = df.apply(lambda x: 'high' if x['studytime']<2 and x['absences']>10 else 'low', axis=1)","df['risk'] = df['studytime'].apply(lambda x: 'high' if x<2 and df['absences']>10 else 'low')","df['risk'] = lambda x: 'high' if df['studytime']<2 and df['absences']>10 else 'low'","df['risk'] = df.map(lambda x: 'high' if x['studytime']<2 and x['absences']>10 else 'low')","df['risk'] = df.apply(lambda x: 'high' if x['studytime']<2 and x['absences']>10 else 'low', axis=1)","Lambda functions","Lab6"),

 ("What is the purpose of random_state=42 in train_test_split?","Ensures reproducible splits across different runs","Sets training size to exactly 42 percent","Limits random samples to exactly 42 items","Creates 42 different splits for testing","Ensures reproducible splits across different runs","Data splitting","Lab5"),

 ("Which code correctly creates binary target from continuous grades?","df['pass'] = np.where(df['grade'] >= 10, 1, 0)","df['pass'] = df['grade'].map(lambda x: 1 >= 10)","df['pass'] = np.if_else(df['grade'] >= 10, 1, 0)","df['pass'] = df['grade'].where(df['grade'] >= 10)","df['pass'] = np.where(df['grade'] >= 10, 1, 0)","Feature engineering","Lab5"),

 ("What does param_grid = {'n_neighbors': np.arange(1, 31)} create?","Dictionary with array of integers from 1 to 30 for grid search","Dictionary with array of integers from 0 to 31 for grid search","List with array of integers from 1 to 30 for testing","Dictionary with single value 31 for grid search","Dictionary with array of integers from 1 to 30 for grid search","Hyperparameter tuning","Lab5"),

 ("Which lambda calculates BMI from weight and height?","df['bmi'] = df.apply(lambda x: x['weight'] / (x['height'] ** 2), axis=1)","df['bmi'] = df.apply(lambda x: x['weight'] / (x['height'] ** 2), axis=0)","df['bmi'] = lambda x: df['weight'] / (df['height'] ** 2)","df['bmi'] = df.map(lambda x: x['weight'] / (x['height'] ** 2))","df['bmi'] = df.apply(lambda x: x['weight'] / (x['height'] ** 2), axis=1)","Lambda functions","Lab6"),

 ("What does auc(fpr, tpr) calculate?","Area under the ROC curve","Area under precision curve","Total number of predictions","Average classification score","Area under the ROC curve","Model evaluation","Lab5"),

 ("Which code correctly handles missing values by filling with mean?","df['age'].fillna(df['age'].mean(), inplace=True)","df['age'].replace(np.nan, df['age'].mean())","df['age'].fill(df['age'].mean(), inplace=True)","df['age'].substitute(np.nan, df['age'].mean())","df['age'].fillna(df['age'].mean(), inplace=True)","Data cleaning","Lab6"),

 ("What does df.apply(lambda x: x.max() - x.min(), axis=0) calculate?","Range (max - min) for each column","Range (max - min) for each row overall","Sum of max and min for each column","Mean of max and min for each row","Range (max - min) for each column","Lambda functions","Lab6"),

 ("Which visualization shows the relationship between two continuous variables?","plt.scatter(x, y)","plt.bar(x, y)","plt.hist(x, bins)","plt.pie(x, labels)","plt.scatter(x, y)","Data visualization","Lab5"),

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

def display_question(q, shuffled_answers, answer_key_map):
    """Display a multiple choice question with options"""
    print(f"\n" + "-"*50)
    question_id = q[0]  # Use question text as ID
    attempts = question_attempts.get(question_id, 0)
    unique_answered = len(answered_questions)
    total_questions = len(valid_questions)
    print(f"\n{q[0]}")
    print(f"\na) {shuffled_answers[0]}")
    print(f"s) {shuffled_answers[1]}")
    print(f"d) {shuffled_answers[2]}")
    print(f"f) {shuffled_answers[3]}")
    
def display_answer(q, user_answer_text, correct_answer, shuffled_answers):
    """Display the correct answer and explanation"""
    if user_answer_text:
        # Find the letter for user answer and correct answer
        answer_map = {shuffled_answers[0]: 'a', shuffled_answers[1]: 's', shuffled_answers[2]: 'd', shuffled_answers[3]: 'f'}
        correct_letter = answer_map.get(correct_answer, '?')
        user_letter = answer_map.get(user_answer_text, '?')
        
        if user_answer_text.lower() == correct_answer.lower():
            print(f"Correct answer: {correct_letter} '{correct_answer}' | Your answer: {user_letter} | ✓ CORRECT!")
        else:
            print(f"Correct answer: {correct_letter} '{correct_answer}' | Your answer: {user_letter} | ✗ INCORRECT!")
    else:
        # Just showing answer without user input
        answer_map = {shuffled_answers[0]: 'a', shuffled_answers[1]: 's', shuffled_answers[2]: 'd', shuffled_answers[3]: 'f'}
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
    correct_answer = question[5]
    
    # Shuffle the answer choices
    answer_choices = [question[1], question[2], question[3], question[4]]
    np.random.shuffle(answer_choices)
    
    # Create mapping for answer keys
    answer_key_map = {'a': answer_choices[0], 's': answer_choices[1], 'd': answer_choices[2], 'f': answer_choices[3]}
    
    display_question(question, answer_choices, answer_key_map)
    choice = get_user_choice()
    
    if choice == 'q':
        break
    elif choice in ['a', 's', 'd', 'f']:
        # Track attempt
        question_attempts[question_id] = question_attempts.get(question_id, 0) + 1
        
        # Map choice to actual answer text
        user_answer_text = answer_key_map[choice]
        
        display_answer(question, user_answer_text, correct_answer, answer_choices)
        
        # Update score and tracking
        total_answered += 1
        answered_questions.add(question_id)  # Track unique questions answered
        
        if user_answer_text.lower() == correct_answer.lower():
            score += 1
            # Remove question from remaining questions (mastered!)
            remaining_questions.remove(question)
        else:
            # Track wrong answer
            if question_id not in wrong_answers:
                wrong_answers[question_id] = {'wrong_count': 0, 'correct_answer': correct_answer}
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

