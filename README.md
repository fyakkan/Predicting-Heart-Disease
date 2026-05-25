Heart Disease (s6e2) — First-Order Optimizer Comparison Study
Furkan Yakkan — Department of Electrical and Computer Engineering, Abdullah Gul University (AGU), Kayseri, Türkiye.

ECE 567 — Foundations of Optimization for Machine Learning. Instructor: Dr. Khaled Hejja.

Companion to my ECE 567 Final Project at Abdullah Gul University. Hand-rolled comparison of SGD, Momentum, NAG, AdaGrad, RMSprop, Adam, AdamW on the heart-disease task — convex (logistic regression) and non-convex (shallow MLP) settings, plus a Lasso feature-selection path.

This public Kaggle notebook is the companion to my ECE 567 Final Project, a survey-style paper comparing seven hand-rolled first-order optimizers — SGD, Polyak Momentum, Nesterov (NAG), AdaGrad, RMSprop, Adam, AdamW — on the Playground Series S6E2 heart-disease classification task. The notebook reproduces every key experiment end-to-end: exploratory data analysis, preprocessing, the convex (linear) and non-convex (shallow MLP) optimizer comparisons, a Lasso feature-selection path, and the final Kaggle submission produced by the winning model.

Evaluation metric: ROC-AUC (probability submissions).
Offline best held-out validation AUC: 0.95278 (MLP + Adam).
Kaggle late-submission private AUC: 0.95288 (MLP + Adam).

Outline
Load data + EDA
Preprocessing (standardize continuous, one-hot nominal)
Linear baselines (sklearn L-BFGS + hand-rolled vanilla GD)
Seven-optimizer comparison on logistic regression (convex)
Lasso path — feature death order vs EDA correlations
Seven-optimizer comparison on shallow MLP (non-convex)
Final retrain on full 630k and submission
Acknowledgment
This work was carried out as part of the ECE 567 graduate course at Abdullah Gul University (AGU), Kayseri, Türkiye, under the supervision of Dr. Khaled Hejja. All code was written from scratch in NumPy; sklearn was used only as a reference solver for benchmarking purposes.
