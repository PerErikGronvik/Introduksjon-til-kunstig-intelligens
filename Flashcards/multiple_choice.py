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

 ("Suppose you are given a data set of student complaints from OsloMet’s customer service center. The data set is labelled. You are now given a task to understand how angry or happy the students are in those complaints. What kind of algorithms would you use?","Regression","Clustering","Classification","Association","Classification","NLP / sentiment analysis","eksamen 2021"),

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

 ("The major reason behind the increased use of Artificial Intelligence today is due to","Availability of increased data","Cloud computing","Powerful processors","Increased connectivity between devices","All of the above","AI trends","eksamen 2021"),

 ("What is the preferred way to work with an A.I. algorithm?","Identify the problem -> prepare data -> choose algorithms -> train the algorithm -> run the algorithm","Identify the problem -> choose algorithms -> run the algorithm -> prepare data -> train the algorithm -> export data to algorithms","Identify the problem -> choose algorithms -> train the algorithm -> run the algorithm -> prepare data -> export data to algorithms","All of the above","Identify the problem -> prepare data -> choose algorithms -> train the algorithm -> run the algorithm","AI workflow","eksamen 2021"),

 ("What is a sigmoid function?","A function that creates a linear relationship","A mathematical function that produces an S-shaped curve used in ML","A function only used in statistics","A function that always outputs zero or one","A mathematical function that produces an S-shaped curve used in ML","GLM","flashcards"),

 ("What is the difference between a generative and discriminative model?","No difference, they are the same","Discriminative learns joint probability, generative learns conditional","Generative learns joint probability, discriminative learns conditional","Both learn only conditional probability","Generative learns joint probability, discriminative learns conditional","ML","flashcards"),

 ("What is the purpose of regularization in machine learning?","To make models more complex","To prevent overfitting by adding penalty terms","To increase training speed","To reduce the number of features","To prevent overfitting by adding penalty terms","ML","flashcards"),

 ("What is the difference between L1 and L2 regularization?","No difference","L1 uses absolute values, L2 uses squared values","L2 uses absolute values, L1 uses squared values","Both use the same penalty method","L1 uses absolute values, L2 uses squared values","ML","flashcards"),

 ("What is gradient descent?","A data preprocessing technique","An optimization algorithm to minimize cost functions","A type of neural network","A feature selection method","An optimization algorithm to minimize cost functions","ML","flashcards"),

 ("What is a decision boundary?","The edge of a dataset","A surface that separates different classes","A type of cost function","A data validation technique","A surface that separates different classes","ML","flashcards"),

 ("What does 96 percent accuracy mean?","The model is 96% confident","Model correctly predicted 96% of output labels","Training took 96% of expected time","96% of features were used","Model correctly predicted 96% of output labels","ML","flashcards"),

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

 ("What is F1 Score in ML","2 * (Precision * Recall) / (Precision + Recall)","TP / (TP + FP)","TP / (TP + FN)","TN / (TN + FP)","2 * (Precision * Recall) / (Precision + Recall)","ML evaluation","Lab"),

 ("What is correct about Log Loss?","It is used for regression","It is used for classification","It is not used in ML","It is a type of regularization","It is used for classification","ML evaluation","Lab"),

 ("What does Log Loss represent?","How close the predicted probabilities are to the actual class labels","The accuracy of the model","The precision of the model","The recall of the model","How close the predicted probabilities are to the actual class labels","ML evaluation","Lab"),

 ("What is the ideal value for Log Loss?","0","1","100","-1","0","ML evaluation","Lab"),

 ("What is a ROC curve?","A plot of true positive rate vs false positive rate","A plot of precision vs recall","A plot of accuracy vs error rate","A plot of loss vs epochs","A plot of true positive rate vs false positive rate","ML evaluation","Lab"),

 ("What does the area under the ROC curve (AUC) represent?","The model's ability to distinguish between classes","The model's accuracy","The model's precision","The model's recall","The model's ability to distinguish between classes","ML evaluation","Lab"),

 ("Is there a trade-off between precision and recall","Yes","No","Sometimes","Never","Yes","ML evaluation","Lab"),

 ("What is the trade-off between precision and recall?","Increasing one decreases the other","They are independent","Increasing one increases the other","They are always equal","Increasing one decreases the other","ML evaluation","Lab"),

 ("What is the main goal of feature engineering?","To create new features that improve model performance","To reduce the number of features","To visualize the data","To select the best model","To create new features that improve model performance","Data engineering","Lab"),
 
 ("What is true about cross-validation?","It provides a more reliable estimate of model performance","It is not important","It is only used for small datasets","It always gives higher accuracy","It provides a more reliable estimate of model performance","ML evaluation","Lab"),

 ("What is L1 regularization?","A technique to reduce overfitting by adding a penalty for larger coefficients","A technique to increase model complexity","A method for feature selection","A type of neural network","A technique to reduce overfitting by adding a penalty for larger coefficients","ML","Lab")





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

