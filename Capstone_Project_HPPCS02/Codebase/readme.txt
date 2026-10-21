CV Sorting System using LLMs (HPPCS02)
=======================================

Project Overview:
-----------------
This project implements an intelligent CV sorting and ranking system using Large Language Models (LLMs). The system automatically evaluates candidate resumes against job requirements and generates a ranked list based on multiple criteria including skills matching, experience level, and semantic relevance.

Directory Structure:
-------------------
Capstone_Project-HPPCS02/
├── Report/
│   └── Capstone_Project_Report_HPPCS02.pdf
└── Codebase/
    ├── main.py
    ├── execution.txt
    ├── job_description.txt
    ├── cv1.pdf
    ├── cv2.pdf
    ├── cv3.pdf
    ├── cv4.pdf
    ├── cv5.pdf
    ├── cv6.pdf
    ├── cv7.pdf
    ├── cv8.pdf
    ├── cv9.pdf
    ├── cv10.pdf
    ├── requirements.txt
    ├── ranked_candidates.txt
    └── execution_log.txt

Prerequisites:
--------------
- Python 3.7 or higher
- Required Python packages (install via requirements.txt)

Installation:
-------------

GOOGLE COLAB INSTALLATION:
1. Upload all files from Codebase directory to Google Colab
2. Install dependencies:
   !pip install sentence-transformers
   !pip install PyPDF2
   !pip install python-docx
   !pip install scikit-learn
   !pip install nltk

LOCAL SYSTEM INSTALLATION:
1. Extract all files to C:\Capstone_Project-HPPCS02
2. Navigate to Codebase directory:
   cd C:\Capstone_Project-HPPCS02\Codebase
3. Install required dependencies:
   pip install -r requirements.txt

Execution:
----------

GOOGLE COLAB EXECUTION:
1. Upload all Codebase files to Colab environment
2. Run the main script:
   !python main.py
3. Download generated output files:
   - ranked_candidates.txt
   - execution_log.txt

LOCAL SYSTEM EXECUTION:
1. Open Command Prompt/Terminal
2. Navigate to Codebase directory:
   cd C:\Capstone_Project-HPPCS02\Codebase
3. Run the main script:
   python main.py

Input Files Required:
--------------------
- job_description.txt: Contains the job requirements and description
- cv1.pdf to cv10.pdf: Candidate resumes in PDF format

Output Files Generated:
-----------------------
- ranked_candidates.txt: Complete ranked list with scores and analysis
- execution_log.txt: Detailed processing timeline and system logs

LLM Models Used:
----------------
1. Rule-based LLM Engine: For exact skills and experience matching
2. Sentence Transformers (all-MiniLM-L6-v2): For semantic analysis and contextual understanding

Key Features:
-------------
- Dual LLM evaluation approach
- Multi-dimensional scoring (skills, experience, semantic relevance)
- Transparent scoring methodology
- Individual candidate feedback and improvement suggestions
- Fast processing (10 candidates in ~2 seconds)

Scoring Methodology:
--------------------
- Skills Score: Percentage of required skills matched
- Experience Score: Comparison against required experience level
- Semantic Score: Contextual relevance using cosine similarity
- Overall Score: Weighted combination of all scores

Support:
--------
For any issues regarding execution, please check:
1. All input files are present in the Codebase directory
2. Python and required dependencies are properly installed
3. Sufficient system memory is available

Note: This system processes resumes locally and does not require internet connection or API keys for local execution.