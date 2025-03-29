\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}

\title{ELE489 HW1 -- KNN on Wine Dataset}
\author{}
\date{}

\begin{document}

\maketitle

\section*{📁 Project Structure}

\begin{verbatim}
├── knn.py                   # Custom k-NN implementation
├── analysis.ipynb          # Jupyter notebook with explanations and plots
├── wine.data               # UCI Wine dataset
├── README.md               # Project description & instructions
\end{verbatim}

\section*{🔍 Project Description}

\textbf{Objective:} Implement and evaluate the k-NN algorithm using different values of K and distance metrics. Compare a manual implementation with scikit-learn's version using the Wine dataset.

\section*{📊 Dataset Information}

\begin{itemize}
    \item Source: \href{https://archive.ics.uci.edu/dataset/109/wine}{UCI ML Repository - Wine Dataset}
    \item 178 samples, 13 numerical features
    \item 3 classes (wine cultivars)
\end{itemize}

\section*{⚙️ How to Run}

\subsection*{1. Install dependencies}

\begin{verbatim}
pip install pandas numpy matplotlib seaborn scikit-learn
\end{verbatim}

\subsection*{2. Run the code}

Open \texttt{analysis.ipynb} in Jupyter Notebook or Colab, or run the script if converted to a Python file.

\section*{📈 Outputs}

\begin{itemize}
    \item Feature visualizations (KDE, boxplots, pairplots)
    \item Accuracy vs. K (custom \& sklearn)
    \item Confusion matrices and classification reports
    \item Match rate between custom and sklearn predictions
\end{itemize}

\section*{✅ Homework Checklist}

\begin{itemize}
    \item [x] Data loading and visualization
    \item [x] Preprocessing and normalization
    \item [x] Train/test split (80/20)
    \item [x] Custom KNN implementation
    \item [x] Euclidean \& Manhattan distances
    \item [x] Accuracy, confusion matrix, report
    \item [x] Sklearn comparison
    \item [x] README.md and GitHub structure
\end{itemize}

\section*{👨‍🏫 Instructor Info}

\textbf{Course:} ELE489 – Fundamentals of Machine Learning \\
\textbf{Instructor:} Prof. Seniha Esen Yüksel \\
\textbf{University:} Hacettepe University

\section*{🔗 References}

\begin{itemize}
    \item \href{https://archive.ics.uci.edu/dataset/109/wine}{UCI Wine Dataset}
    \item \href{https://www.kaggle.com/code/prashant111/knn-classifier-tutorial}{Kaggle KNN Tutorial}
    \item \href{https://www.w3schools.com/python/python_ml_confusion_matrix.asp}{Confusion Matrix Guide}
\end{itemize}

\end{document}
