# DL-RAG Project

A collection of projects focused on C++ programming, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Deep Learning.

## 🚀 Projects Overview

### 1. Document-Based LLM and RAG Assistant

An interactive AI application that allows users to upload documents and ask questions about their content.

**Features**
- Upload documents and ask questions about their content.
- LLM Mode for language-model-based responses.
- Research Paper Information Mode for exploring paper content.
- Hybrid Mode combining document retrieval and LLM responses.
- Context-based answer generation using retrieved information.
- Interactive interface built with Streamlit.

**Technologies:** Python, Streamlit, LLMs, RAG, Embeddings, Vector Search.

### 2. CNN-Based Deep Learning Model

A Convolutional Neural Network (CNN) project focused on image-based classification and prediction.

**Features**
- Image preprocessing.
- CNN model implementation and training.
- Image classification and prediction.
- Model evaluation.

**Technologies:** Python, CNN, Deep Learning.

### 3. C++ Programming Project

A programming project containing C++ learning materials, examples, and tutorial resources.

**Topics**
- C++ fundamentals.
- Programming concepts and examples.
- C++ learning resources.

**Technologies:** C++.

## 📂 Repository Structure

```text
DL-RAG-project/
│
├── .agents/
├── .vscode/
│
├── assessment/
│   ├── adaptive_learning.pdf
│   ├── mevion.pdf
│   ├── mevita.pdf
│   ├── mevius.pdf
│   ├── VLA.pdf
│   ├── project_app.py
│   └── RAG_project.ipynb
│
├── app.py
├── cnn.py
├── cnn_1.py
├── cpp.txt
├── cpp_tutorial_LLM.py
├── paper.pdf
├── RAGTutorial.ipynb
├── RAGTutorial_copy.ipynb
├── streamlit_tutorial.py
│
├── requirements.txt
├── README.md
└── .gitignore
```


## 🛠️ Technologies Used

- Python
- C++
- Streamlit
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Convolutional Neural Networks (CNNs)
- Machine Learning and Deep Learning
- Jupyter Notebook

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/nehant-17/DL-RAG-project.git
cd DL-RAG-project
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application

If `app.py` is your main Streamlit application:

```bash
streamlit run app.py
```

If the RAG application is located in `assessment/project_app.py`, run:

```bash
streamlit run assessment/project_app.py
```

Use the entry point that actually launches your application.

## 🔐 Security and Best Practices

- Never upload API keys, passwords, or credentials.
- Store secrets in a local `.env` file.
- Exclude virtual environments and Python cache files from version control.
- Avoid uploading confidential documents or private datasets.
- Keep `requirements.txt` updated with the required dependencies.

## 🎯 Project Objectives

- Explore C++ programming and software development.
- Develop an interactive document-based AI question-answering system.
- Experiment with LLMs, document retrieval, and hybrid answering strategies.
- Implement and evaluate a CNN-based deep learning model.

## 🚀 Future Improvements

- Improve document retrieval and answer relevance.
- Add support for additional document formats.
- Evaluate the RAG application using retrieval and answer-quality metrics.
- Improve CNN performance using appropriate datasets and evaluation metrics.
- Enhance application usability and reliability.

## 👨‍💻 Author

Developed as a practical exploration of programming, machine learning, and AI applications.

## 📄 License

A license can be added if the project is intended for public use or distribution.