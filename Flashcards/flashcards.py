# imports for creating comandline based flashcards. No timekeeping or scoring yet. The questions and answers are typed into the code. maybe pandas dataframe of some kind of arraystructure.

import numpy as np


# Questions and answers, with categories. 
qa = [
    ("What is a sigmoid function?", "A sigmoid function is a mathematical function that produces an S-shaped curve. It is often used in machine learning and statistics to model probabilities. It separates the input space into two regions and classify them as 1 or 0.", "GLM"),
    ("What is the difference between a generative and discriminative model?", "A generative model learns the joint probability distribution of the input features and the output labels, while a discriminative model learns the conditional probability distribution of the output labels given the input features.", "ML"),
    ("What is the purpose of regularization in machine learning?", "Regularization is a technique used to prevent overfitting in machine learning models. It adds a penalty term to the loss function that discourages the model from fitting the training data too closely.", "ML"),
    ("What is the difference between L1 and L2 regularization?", "L1 regularization adds a penalty term proportional to the absolute value of the model coefficients, while L2 regularization adds a penalty term proportional to the square of the model coefficients. L1 regularization can lead to sparse models, while L2 regularization tends to produce models with small but non-zero coefficients.", "ML"),
    ("What is a cost function?", "A cost function is a mathematical function that quantifies the difference between the predicted output of a model and the actual output. It is used to optimize the model during training by minimizing the error.", "ML"),
    ("What is gradient descent?", "Gradient descent is an optimization algorithm used to minimize the cost function of a machine learning model. It works by iteratively adjusting the model parameters in the direction of the steepest descent of the cost function.", "ML"),
    ("What is a decision boundary?", "A decision boundary is a surface that separates different classes in a classification problem. It is determined by the model's parameters and can be linear or non-linear.", "ML"),
    ("What is the difference between supervised and unsupervised learning?", "Supervised learning involves training a model on labeled data, while unsupervised learning involves training a model on unlabeled data. In supervised learning, the goal is to predict the output labels for new input data, while in unsupervised learning, the goal is to discover patterns or structure in the input data.", "ML"),
    ("What is cross-validation?", "Cross-validation is a technique used to evaluate the performance of a machine learning model. It involves splitting the data into training and validation sets multiple times and averaging the results to get a more accurate estimate of the model's performance.", "ML"),
    ("What does a 96 percent accuracy mean?", "It means that the model correctly predicted the output labels for 96 percent of the input data. However, it is important to consider other metrics such as precision, recall, and F1 score to get a more complete picture of the model's performance.", "ML"),
    ("what is the stages of a ml model, according to the course","train, evaluate, minimize the cost function, mesure the accuracy, measure probability score","ML"),
    ("What is a good classification usecase", "how many in the bus at a given time", "ML"),
    ("What is a bad classification usecase", "predicting the weather", "ML"),
    ("What is the purpose of the logit function in logistic regression?", "The logit function is used to transform the predicted probabilities of a binary classification problem into a linear combination of the input features. It maps the probabilities from the range [0, 1] to the range (-∞, +∞), allowing for easier optimization of the model parameters.", "GLM"),
    ("What is the difference between logistic regression and linear regression?", "Logistic regression is used for binary classification problems, while linear regression is used for regression problems. Logistic regression predicts the probability of an input belonging to a certain class, while linear regression predicts a continuous output value.", "GLM"),
    ("what is naive bayes", "a simple probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features", "ML"),





]




# Shuffle questions
np.random.shuffle(qa)

# Loop through cards. flip with enter, flip back with enter. n for next card, p for previous card, q to quit.
i = 0
show_answer = False
while True:
    if not show_answer:
        print(f"\nQuestion: {qa[i][0]}")
    else:
        print(f"\nAnswer: {qa[i][1]}")
    cmd = input("Press Enter to flip, 'n' for next, 'p' for previous, 'q' to quit: ").strip().lower()
    if cmd == '':
        show_answer = not show_answer
    elif cmd == 'n':
        i = (i + 1) % len(qa)
        show_answer = False
    elif cmd == 'p':
        i = (i - 1) % len(qa)
        show_answer = False
    elif cmd == 'q':
        break
    else:
        print("Invalid command. Please try again.")



