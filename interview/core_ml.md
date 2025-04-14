### 🔍 **Deep Learning / ML Fundamentals**

1. **Can you explain the bias-variance tradeoff in your own words, and how it affects model performance?**
2. **What’s your understanding of overfitting and underfitting? How do you detect and handle them in practice?**
3. **Describe the difference between L1 and L2 regularization. When would you use one over the other?**
4. **What’s your go-to strategy for tuning a deep learning model with millions of parameters?**
5. **What’s the role of the learning rate in training deep neural networks? What happens if it's too high or too low?**

---

### 🧠 **Deep Learning Architecture**

6. **Walk me through the architecture of a Transformer. Why is it so powerful compared to RNNs or LSTMs?**
7. **Suppose you want to train a CNN for medical image classification with limited data. What techniques would you use?**
8. **What are attention mechanisms, and where have you used them in a project or research work?**
9. **What’s the role of batch normalization, and how does it affect training?**
10. **Have you worked with diffusion models, GANs, or other generative models? If so, can you explain their pros and cons?**

---

### 🛠️ **Practical ML & Deployment**

11. **How do you handle data drift and concept drift in deployed ML systems?**
12. **What steps would you take to productionize a deep learning model at scale?**
13. **How do you choose between PyTorch and TensorFlow for a project?**
14. **Have you optimized any models for inference (e.g., quantization, pruning, distillation)? Explain your process.**
15. **Tell me about a real-world machine learning pipeline you've built. What were the main challenges?**

---

### 🔬 **Research & Experimentation**

16. **Can you talk about a paper or idea that inspired you recently? How would you extend or critique it?**
17. **How do you evaluate whether a new technique is worth pursuing further in research?**
18. **Have you published any papers or contributed to open source projects? What was your role?**
19. **Describe a situation where you had to challenge an established technique and propose a better one.**
20. **How do you balance reading papers with doing hands-on experiments?**

---

### 🧩 **Problem Solving & System Design**

21. **Design a recommendation system for a video platform. Walk through the ML architecture, features, and feedback loop.**
22. **You have a noisy, incomplete dataset with lots of missing labels. How would you build a robust model from it?**
23. **How would you build a model that adapts in real time to user behavior, e.g., online learning?**
24. **Design a scalable image classification service that can handle 10 million queries per day.**
25. **How would you approach explainability in deep learning models in regulated industries (e.g., finance or healthcare)?**

---

### ⚡ Bonus Curveballs (for fun and creativity)

26. **If you had unlimited compute and data, what ML problem would you solve and why?**
27. **What’s a misconception about AI you often see, even among technical folks?**
28. **Imagine GPT-like models didn’t exist. What would be your approach to build a powerful NLP system today?**
29. **Which part of the ML lifecycle do you think is most overlooked but crucial?**
30. **How would you explain backpropagation to a high schooler?**

---


### 🔹 **1. Explain the bias-variance tradeoff.**

**Follow-ups:**
- How does model complexity relate to bias and variance?
- Can you give an example of a model with high bias? High variance?
- How would you identify this tradeoff in practice (e.g., in training/validation curves)?
- What techniques can help balance this tradeoff?

---

### 🔹 **2. Describe how backpropagation works.**

**Follow-ups:**
- What role does the chain rule play in backpropagation?
- How is it different in recurrent neural networks?
- Can you walk me through backpropagation in a multi-layer perceptron with ReLU activations?
- What are some common issues during backpropagation in deep networks?

---

### 🔹 **3. What are attention mechanisms and why are they useful?**

**Follow-ups:**
- How do attention weights get computed in Transformers?
- Can you explain scaled dot-product attention?
- Why is self-attention important for sequence modeling?
- What are the computational tradeoffs of using attention compared to CNNs or RNNs?

---

### 🔹 **4. You have limited labeled data for a classification task. What do you do?**

**Follow-ups:**
- Would you consider transfer learning or data augmentation? How?
- How about semi-supervised or self-supervised approaches?
- What are the risks with synthetic data or pseudo-labeling?
- Would you ever use active learning? How would you implement it?

---

### 🔹 **5. What is gradient clipping and when would you use it?**

**Follow-ups:**
- Can you explain the problem of exploding gradients?
- What's the difference between clipping by value vs. by norm?
- How would you identify if your model needs gradient clipping?
- In which types of networks is this especially important?

---

### 🔹 **6. Walk me through the architecture of a Transformer.**

**Follow-ups:**
- What are the key components in a Transformer encoder block?
- How does positional encoding work and why is it needed?
- Why is LayerNorm used, and why is it placed before or after attention layers?
- How does scaling affect training stability in attention layers?

---

### 🔹 **7. How do you evaluate a multi-class classification model?**

**Follow-ups:**
- What metrics would you use for imbalanced classes?
- When is accuracy misleading?
- Can you explain the difference between micro, macro, and weighted F1 scores?
- How would you handle evaluation in a hierarchical class structure?

---

### 🔹 **8. You’ve trained a model, but it's overfitting. What next?**

**Follow-ups:**
- What indicators show overfitting in the training process?
- What regularization techniques would you try first?
- Would early stopping help? How do you implement it?
- Would you change the model size, or focus on the data?

---

### 🔹 **9. Suppose your deployed model starts degrading in performance over time. What do you do?**

**Follow-ups:**
- How would you detect data drift or concept drift?
- What monitoring tools or metrics would you use?
- Would you retrain from scratch or fine-tune on new data?
- What are your thoughts on continual learning in production systems?

---

### 🔹 **10. Can you explain model distillation? Why and when would you use it?**

**Follow-ups:**
- How does the student model learn from the teacher?
- What kind of loss functions are used in distillation?
- What’s the benefit in real-world applications (e.g., mobile deployment)?
- Can you distill from an ensemble? How?

---

## 🔹 Core Machine Learning

### **1. What are the assumptions behind linear regression?**
**Follow-ups:**
- What happens if the assumptions are violated?
- How do you test for multicollinearity?
- How does regularization (like Ridge or Lasso) help here?

### **2. What is the difference between generative and discriminative models?**
**Follow-ups:**
- Where would you prefer using each?
- Can you give examples from NLP or vision?
- How would you convert a generative model into a discriminative one?

---

## 🔹 Neural Networks & Training

### **3. Why might a deep neural network fail to converge?**
**Follow-ups:**
- What would you check first? Learning rate? Initialization?
- What’s the impact of poor weight initialization?
- How do optimizers like Adam help in such situations?

### **4. How do skip connections help in deep networks (e.g., ResNets)?**
**Follow-ups:**
- How do they influence the gradient flow?
- What’s the difference between dense connections (e.g., DenseNet) and skip connections?
- Why is identity mapping important in ResNets?

---

## 🔹 Advanced Architectures

### **5. Explain the concept of multi-head attention.**
**Follow-ups:**
- Why not just one attention head?
- How do you aggregate outputs from different heads?
- How would reducing the number of heads affect performance?

### **6. What are graph neural networks and where would you apply them?**
**Follow-ups:**
- How do message-passing and aggregation work?
- Can you describe one real-world application (e.g., molecules, social networks)?
- What challenges do GNNs face when scaling?

---

## 🔹 Generative Models

### **7. How do GANs work? What are the common issues?**
**Follow-ups:**
- What is mode collapse? How can you mitigate it?
- Can you explain how WGAN improves stability?
- How would you evaluate a GAN model?

### **8. Compare VAEs and GANs.**
**Follow-ups:**
- Which would you prefer for structured latent representations?
- How does the ELBO (Evidence Lower Bound) work in VAEs?
- Can you combine VAEs and GANs?

---

## 🔹 NLP

### **9. What is the role of tokenization in NLP?**
**Follow-ups:**
- Compare Byte Pair Encoding (BPE) with WordPiece.
- What challenges does tokenization present for multilingual models?
- Would you ever use character-level models?

### **10. How do you fine-tune a large language model on a domain-specific task?**
**Follow-ups:**
- What are the risks of catastrophic forgetting?
- Would you use adapters or LoRA? Why?
- How would you approach prompt tuning vs. full fine-tuning?

---

## 🔹 Reinforcement Learning

### **11. Explain the exploration-exploitation trade-off.**
**Follow-ups:**
- How do algorithms like UCB or ε-greedy handle this?
- What if the environment is non-stationary?
- Have you used policy gradients or Q-learning before?

### **12. What are the main components of the RL problem formulation?**
**Follow-ups:**
- Define state, action, reward, policy, and value function.
- How do model-based and model-free approaches differ?
- How do you stabilize training in deep RL?

---

## 🔹 Research & Theory

### **13. How do you come up with new ideas for ML research?**
**Follow-ups:**
- Walk me through how you’d extend a recent paper.
- What does a solid experimental design look like?
- How do you validate if a new idea is *actually* novel?

### **14. What's a recent ML paper that impressed you and why?**
**Follow-ups:**
- What are its key contributions?
- How would you improve or challenge its results?
- What would be the next logical experiment?

---

## 🔹 Systems & Scaling

### **15. How do you handle training large models on distributed systems?**
**Follow-ups:**
- What’s the difference between data parallelism and model parallelism?
- What is gradient accumulation?
- How does mixed-precision training help?

### **16. How do you monitor and maintain an ML system in production?**
**Follow-ups:**
- What are your favorite tools or metrics?
- How would you deal with drift in user behavior?
- Would you ever deploy an online learning model?

---

## 🔹 Ethical & Practical Thinking

### **17. What ethical challenges do you consider when deploying ML systems?**
**Follow-ups:**
- How do you test for model bias?
- What’s your approach to ensuring fairness in sensitive applications?
- How do you handle explainability in complex models?

### **18. What tradeoffs do you consider when choosing a model?**
**Follow-ups:**
- Accuracy vs interpretability?
- Latency vs complexity?
- Explain how you'd handle this in a medical or real-time system.

---

## 🔹 Core Machine Learning Questions (With Follow-Ups)

---

### **1. What is the difference between supervised, unsupervised, and semi-supervised learning?**

**Follow-ups:**
- Can you give real-world examples of each?
- How does the performance and data requirement differ across them?
- Where would self-supervised learning fit?

---

### **2. How do decision trees work? What are their pros and cons?**

**Follow-ups:**
- How does the tree decide which feature to split on?
- What causes overfitting in decision trees?
- How does pruning help?

---

### **3. What is the difference between bagging and boosting?**

**Follow-ups:**
- Can you explain how Random Forest uses bagging?
- What are common boosting algorithms? How do they differ (e.g., XGBoost vs. AdaBoost)?
- Why does boosting often outperform bagging?

---

### **4. What is cross-validation and why is it important?**

**Follow-ups:**
- What’s the difference between k-fold, stratified k-fold, and leave-one-out?
- When might cross-validation fail or mislead you?
- How would you apply cross-validation in time-series data?

---

### **5. What are precision, recall, F1-score, and accuracy?**

**Follow-ups:**
- Why is accuracy not enough for imbalanced datasets?
- Can you draw the confusion matrix and explain how each metric is derived?
- When would you prioritize precision over recall?

---

### **6. What is regularization and why is it important in ML models?**

**Follow-ups:**
- Can you explain L1 vs. L2 regularization intuitively and mathematically?
- Why does L1 encourage sparsity?
- Can you use both L1 and L2 together?

---

### **7. What are the main assumptions of logistic regression?**

**Follow-ups:**
- What happens if features are highly correlated?
- How do you deal with multicollinearity?
- How do you interpret model coefficients?

---

### **8. What’s the difference between generative and discriminative models?**

**Follow-ups:**
- Why is Naive Bayes considered generative?
- Why is logistic regression discriminative?
- Can you use a generative model for classification?

---

### **9. What is the curse of dimensionality?**

**Follow-ups:**
- How does it affect nearest neighbor algorithms?
- What techniques help mitigate it?
- Can dimensionality reduction ever hurt performance?

---

### **10. How do SVMs work?**

**Follow-ups:**
- What is the margin in SVM?
- How do kernels help in non-linear classification?
- Can you explain the difference between hard-margin and soft-margin SVMs?

---

### **11. Explain the ROC and AUC metrics.**

**Follow-ups:**
- What does an AUC of 0.5 or 1.0 mean?
- How does ROC compare to precision-recall curves?
- When would PR curves be more informative than ROC?

---

### **12. What is feature selection vs. feature extraction?**

**Follow-ups:**
- Can you give examples of each?
- How does PCA fit into this?
- What’s the impact on interpretability and performance?

---

### **13. How do you handle missing data in a dataset?**

**Follow-ups:**
- When would you drop vs. impute missing values?
- What imputation strategies do you prefer?
- How does missing not at random (MNAR) affect modeling?

---

### **14. What is the difference between parametric and non-parametric models?**

**Follow-ups:**
- Is KNN parametric or non-parametric?
- What tradeoffs are involved?
- When would you choose one over the other?

---

### **15. What is multicollinearity and why is it a problem in linear models?**

**Follow-ups:**
- How do you detect multicollinearity?
- What are Variance Inflation Factors (VIF)?
- How can regularization help?

---

### **16. How does K-means clustering work?**

**Follow-ups:**
- What are its assumptions and limitations?
- How do you choose the value of K?
- Can K-means be used on non-spherical clusters?

---

### **17. What’s the difference between KNN and K-means?**

**Follow-ups:**
- One is supervised, the other is unsupervised — explain.
- What are their runtime complexities?
- Can you use both in a single pipeline?

---

### **18. How does gradient descent work?**

**Follow-ups:**
- What’s the difference between batch, mini-batch, and stochastic gradient descent?
- How does the learning rate affect convergence?
- What happens if the learning rate is too high?

---

### **19. What’s the difference between classification and regression tasks?**

**Follow-ups:**
- How does the loss function differ?
- Can you convert a regression problem into a classification one?
- What about multi-label or ordinal classification — where do they fit?

---

### **20. How do you detect and handle outliers in a dataset?**

**Follow-ups:**
- What statistical methods do you prefer (e.g., IQR, z-score)?
- When would you keep outliers instead of removing them?
- How do outliers affect distance-based models?

---

### **21. What is the difference between convex and non-convex optimization?**

**Follow-ups:**
- Why is convexity important in ML?
- Give an example of a convex loss function.
- How do optimizers deal with non-convexity in deep learning?

---

### **22. What is a loss function? How is it different from a cost function?**

**Follow-ups:**
- Why is MSE used for regression but not classification?
- How does using cross-entropy affect gradient behavior?
- What happens if your loss function is poorly chosen?

---

### **23. What is the no free lunch theorem in ML?**

**Follow-ups:**
- What implications does it have for choosing algorithms?
- How do we typically “cheat” the no free lunch result in practice?
- Is deep learning an exception to this rule?

---

### **24. Explain the difference between early stopping and dropout.**

**Follow-ups:**
- How do they help prevent overfitting?
- Would you use both in the same model?
- When might dropout hurt performance?

---

### **25. What is the bias in model evaluation?**

**Follow-ups:**
- What are data leakage and label leakage?
- Can train/test split introduce bias?
- How do you ensure your validation metric is representative?

---

### **26. What’s the difference between online and batch learning?**

**Follow-ups:**
- Which use cases require online learning?
- How do memory and compute constraints impact this decision?
- What’s the difference between online learning and reinforcement learning?

---

### **27. How do you perform feature scaling and why is it important?**

**Follow-ups:**
- When do you use standardization vs normalization?
- Which models are sensitive to feature scale?
- How would you scale features during inference?

---

### **28. What are some common data preprocessing techniques?**

**Follow-ups:**
- How do you handle categorical variables?
- What’s one-hot encoding vs label encoding?
- How do you handle date/time features?

---

### **29. What is the difference between PCA and t-SNE?**

**Follow-ups:**
- Which would you use for visualization vs compression?
- How do they each preserve data structure?
- What are the pitfalls of using t-SNE?

---

### **30. What is a confusion matrix and what can it tell you?**

**Follow-ups:**
- Can you derive precision, recall, and F1 from it?
- What does it reveal that a single number like accuracy doesn’t?
- How does it change for multi-class classification?

---

### **31. What are ensemble learning methods?**

**Follow-ups:**
- How does voting work in ensemble classifiers?
- When does ensembling fail to help?
- What’s the tradeoff between ensemble complexity and interpretability?

---

### **32. What’s the difference between underfitting and overfitting?**

**Follow-ups:**
- Can you detect them using training/validation curves?
- What causes underfitting in decision trees? Overfitting in neural nets?
- How do you tune to avoid both?

---

### **33. How do kernel methods work in SVMs?**

**Follow-ups:**
- Can you explain the RBF kernel?
- What does the kernel trick allow us to do?
- How does it relate to the curse of dimensionality?

---

### **34. What is model interpretability and why does it matter?**

**Follow-ups:**
- Which models are inherently interpretable?
- How do SHAP or LIME help explain black-box models?
- Would you trade accuracy for explainability?

---

### **35. How do you perform hyperparameter tuning?**

**Follow-ups:**
- What’s the difference between grid search and random search?
- Have you used Bayesian optimization or Hyperopt?
- How does cross-validation fit into this process?

---

### **36. What is stratified sampling? When should you use it?**

**Follow-ups:**
- How does it help with class imbalance?
- Can you apply stratification in cross-validation?
- What if your dataset has multiple categorical groupings?

---

### **37. What are the differences between classification threshold tuning and class weighting?**

**Follow-ups:**
- How does changing the threshold affect precision/recall?
- When would you use class weights vs sampling strategies?
- What tradeoffs exist when trying to “balance” a dataset?

---

### **38. What is the relationship between likelihood and loss functions?**

**Follow-ups:**
- How does maximizing likelihood relate to minimizing loss?
- Why is log-likelihood commonly used?
- Can you give an example in logistic regression?

---

### **39. How do you detect and prevent data leakage?**

**Follow-ups:**
- Can you give an example from a Kaggle competition or real project?
- How does leakage affect model evaluation?
- How do you design your pipeline to prevent it?

---

### **40. What are probabilistic models in ML?**

**Follow-ups:**
- How are probabilistic graphical models (like Bayesian networks) different from deterministic models?
- When would you use a probabilistic prediction instead of a point estimate?
- What’s the role of uncertainty in prediction?

---

### **41. What’s the difference between hard and soft classification?**

**Follow-ups:**
- When would you prefer probabilistic (soft) outputs?
- How do soft predictions affect ROC/AUC calculation?
- How would you threshold soft predictions?

---

### **42. What is label imbalance, and how do you address it?**

**Follow-ups:**
- What’s the difference between oversampling and undersampling?
- How does SMOTE work?
- Can you use class weights with gradient boosting models?

---

### **43. How does Naive Bayes work, and what are its assumptions?**

**Follow-ups:**
- What happens when the independence assumption is violated?
- Why does Naive Bayes perform surprisingly well in NLP tasks?
- How do you handle zero probabilities in Naive Bayes?

---

### **44. What’s the intuition behind the k-nearest neighbors algorithm?**

**Follow-ups:**
- How do you choose the value of *k*?
- Why is KNN sensitive to feature scaling?
- How does high dimensionality affect KNN?

---

### **45. How do you choose an appropriate model for a given ML problem?**

**Follow-ups:**
- What factors do you consider: data size, dimensionality, interpretability?
- Would you always start with a simple model? Why or why not?
- Can you walk me through how you iterate through model selection?

---

### **46. What are the tradeoffs between linear and non-linear models?**

**Follow-ups:**
- Can you explain this using logistic regression vs decision trees?
- How do feature interactions influence your choice?
- How does regularization play into this decision?

---

### **47. What is multivariate vs univariate analysis in ML?**

**Follow-ups:**
- Why is univariate analysis not enough during EDA?
- How do correlations or mutual information help in multivariate analysis?
- What tools/libraries do you use for this?

---

### **48. What is the role of entropy in decision trees?**

**Follow-ups:**
- How is information gain calculated?
- What other criteria can be used (e.g., Gini index)?
- How does entropy affect split selection?

---

### **49. Explain the concept of "variance" in model performance.**

**Follow-ups:**
- How does variance differ from overfitting?
- How do ensemble methods reduce variance?
- How do cross-validation results help detect high variance?

---

### **50. What’s the difference between model parameters and hyperparameters?**

**Follow-ups:**
- Can you give examples for linear regression, decision trees, and neural networks?
- How do you optimize hyperparameters?
- Are hyperparameters ever learned during training?

---

### **51. How do you evaluate clustering algorithms without labels?**

**Follow-ups:**
- What are metrics like silhouette score and Davies-Bouldin index?
- When is visual inspection valid? When is it misleading?
- What assumptions do these metrics rely on?

---

### **52. What is dimensionality reduction and why is it important?**

**Follow-ups:**
- What are the differences between PCA, t-SNE, and UMAP?
- How do you know how many dimensions to keep?
- How does dimensionality reduction help in model performance or training speed?

---

### **53. Explain how gradient boosting works.**

**Follow-ups:**
- How does it differ from bagging?
- What’s the intuition behind learning from residuals?
- Can boosting overfit? How do you prevent it?

---

### **54. What is the role of the learning rate in training?**

**Follow-ups:**
- What happens if the learning rate is too high? Too low?
- How do learning rate schedulers help?
- What is a warm-up phase?

---

### **55. What’s the difference between entropy and cross-entropy?**

**Follow-ups:**
- Why do we use cross-entropy in classification?
- Can you write down the formula for binary cross-entropy?
- What does minimizing cross-entropy actually mean in practice?

---

### **56. What is the role of the sigmoid function?**

**Follow-ups:**
- Why does sigmoid saturate?
- How does ReLU solve the vanishing gradient problem?
- What’s the difference between sigmoid and softmax?

---

### **57. How do you detect underperforming features in a model?**

**Follow-ups:**
- What’s feature importance? How do you measure it?
- How would you use SHAP or permutation importance?
- Can removing features ever *improve* accuracy?

---

### **58. What is stochasticity in machine learning models?**

**Follow-ups:**
- What sources of randomness exist in training?
- How do you ensure reproducibility in experiments?
- What role does random seed setting play?

---

### **59. What is bootstrapping in the context of ML?**

**Follow-ups:**
- How does it relate to bagging?
- How would you use bootstrapping for confidence intervals?
- Can bootstrapping be used in cross-validation?

---

### **60. What is model calibration and why is it important?**

**Follow-ups:**
- How do you measure calibration?
- What’s the difference between sharpness and calibration?
- Can you name techniques to calibrate models (e.g., Platt scaling, isotonic regression)?

---
