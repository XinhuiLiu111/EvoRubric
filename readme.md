# EvoRubric: Evolutionary Algorithm-Based Automated Essay Scoring System

## Project Overview

EvoRubric is a research project focused on automated essay scoring systems based on evolutionary algorithms. This project optimizes scoring criteria through evolutionary algorithms to improve the accuracy and consistency of AI models in essay scoring tasks. The project compares the performance of two large language models, GLM and Doubao, under different scoring criteria.

## Key Features

- 🧬 **Evolutionary Algorithm Optimization**: Automatically optimize scoring criteria using evolutionary algorithms
- 🤖 **Multi-Model Comparison**: Support performance comparison between GLM and Doubao models
- 📊 **Comprehensive Evaluation**: Provide multiple evaluation metrics including Kappa coefficient and APO
- 📈 **Visual Analysis**: Generate scoring distribution charts and performance comparison graphs
- 📝 **Detailed Logging**: Complete experimental logs and scoring process records

## Project Structure
EvoRubric/
├── 📁 Code/                           # Core code directory
│   ├── 🐍 Algo.py                     # Core evolutionary algorithm implementation
│   ├── 📁 CoT/                        # Chain of Thought module
│   │   ├── 🐍 cot_logger.py           # CoT logger
│   │   ├── 🐍 cot_model_client.py     # CoT model client
│   │   └── 🐍 cot_scorer.py           # CoT scorer
│   ├── 🐍 Cross-model_Kappa_doubao_evolved.py  # Doubao model cross-model evaluation
│   ├── 🐍 Cross-model_Kappa_glm_evolved.py     # GLM model cross-model evaluation
│   ├── 🐍 config.py                   # Configuration file
│   ├── 🐍 getData7.py                 # Data processing module
│   ├── 🐍 main.py                     # Main program entry
│   └── 🐍 utils.py                    # Utility functions module
│
├── 📁 Data/                           # Data directory
│   ├── 📁 ASAP/                       # ASAP dataset
│   │   ├── 📁 prompts/                # Essay prompts
│   │   ├── 📊 score range.xlsx        # Score ranges
│   │   ├── 📁 scoring_rubric/         # Scoring rubrics
│   │   └── 📊 training_set_rel3.xlsx  # Training dataset
│   │
│   ├── 📁 Cross-model/                # Cross-model experiment results
│   │   ├── 📁 doubao(glm-rubric)/     # Doubao model using GLM scoring criteria
│   │   │   ├── 📁 essay_set_1/        # Essay set 1 experiment results
│   │   │   │   ├── 📁 conversations/  # Scoring conversation records
│   │   │   │   ├── 📄 detailed_scores.csv     # Detailed scoring data
│   │   │   │   ├── 📄 metrics.csv             # Evaluation metrics
│   │   │   │   └── 📄 used_scoring_criteria.txt # Used scoring criteria
│   │   │   ├── 📁 essay_set_2/        # Essay set 2 experiment results
│   │   │   ├── 📁 essay_set_3/        # Essay set 3 experiment results
│   │   │   ├── 📁 essay_set_4/        # Essay set 4 experiment results
│   │   │   ├── 📁 essay_set_5/        # Essay set 5 experiment results
│   │   │   ├── 📁 essay_set_6/        # Essay set 6 experiment results
│   │   │   ├── 📁 essay_set_7/        # Essay set 7 experiment results
│   │   │   └── 📁 essay_set_8/        # Essay set 8 experiment results
│   │   │
│   │   └── 📁 glm(doubao-rubric)/     # GLM model using Doubao scoring criteria
│   │       ├── 📁 essay_set_1/        # Essay set 1 experiment results
│   │       ├── 📁 essay_set_2/        # Essay set 2 experiment results
│   │       ├── 📁 essay_set_3/        # Essay set 3 experiment results
│   │       ├── 📁 essay_set_4/        # Essay set 4 experiment results
│   │       ├── 📁 essay_set_5/        # Essay set 5 experiment results
│   │       ├── 📁 essay_set_6/        # Essay set 6 experiment results
│   │       ├── 📁 essay_set_7/        # Essay set 7 experiment results
│   │       └── 📁 essay_set_8/        # Essay set 8 experiment results
│   │
│   ├── 📁 Task_Characteristics_Results/ # Task characteristics analysis results
│   │   ├── 📁 Essay length/           # Essay length analysis
│   │   ├── 📁 Essay type/             # Essay type analysis
│   │   └── 📁 Grade level/            # Grade level analysis
│   │
│   ├── 📊 Cross-model scoring.xlsx    # Cross-model scoring summary
│   ├── 📊 Essay length-kappa.xlsx     # Essay length vs Kappa relationship
│   ├── 📊 Essay set describtion.xlsx  # Essay set descriptions
│   ├── 📊 Essay type-kappa.xlsx       # Essay type vs Kappa relationship
│   ├── 📊 Grade level-kappa.xlsx      # Grade level vs Kappa relationship
│   ├── 📊 Parent-Kappa.xlsx           # Parent Kappa evolution data
│   ├── 📊 Rubric.xlsx                 # Scoring rubric templates
│   ├── 📊 Score(test set).xlsx        # Test set scoring data
│   └── 📊 Test_set-kappa.xlsx         # Test set Kappa coefficients
│
├── 📁 logs/                           # Experiment logs
│   ├── 📁 CoT/                        # Chain of Thought logs
│   │   ├── 📄 doubao_cot_scoring.log  # Doubao model CoT scoring logs
│   │   └── 📄 glm_cot_scoring.log     # GLM model CoT scoring logs
│   ├── 📄 essay_set1-doubao.log       # Essay set 1 Doubao model logs
│   ├── 📄 essay_set1-glm.log          # Essay set 1 GLM model logs
│   ├── 📄 essay_set2-doubao.log       # Essay set 2 Doubao model logs
│   ├── 📄 essay_set2-glm.log          # Essay set 2 GLM model logs
│   ├── 📄 essay_set3-doubao.log       # Essay set 3 Doubao model logs
│   ├── 📄 essay_set3-glm.log          # Essay set 3 GLM model logs
│   ├── 📄 essay_set4-doubao.log       # Essay set 4 Doubao model logs
│   ├── 📄 essay_set4-glm.log          # Essay set 4 GLM model logs
│   ├── 📄 essay_set5-doubao.log       # Essay set 5 Doubao model logs
│   ├── 📄 essay_set5-glm.log          # Essay set 5 GLM model logs
│   ├── 📄 essay_set6-doubao.log       # Essay set 6 Doubao model logs
│   ├── 📄 essay_set6-glm.log          # Essay set 6 GLM model logs
│   ├── 📄 essay_set7-doubao.log       # Essay set 7 Doubao model logs
│   ├── 📄 essay_set7-glm.log          # Essay set 7 GLM model logs
│   ├── 📄 essay_set8-doubao.log       # Essay set 8 Doubao model logs
│   └── 📄 essay_set8-glm.log          # Essay set 8 GLM model logs
│
├── 📄 Data Dictionary.md              # Data dictionary
└── 📄 README.md                       # Project documentation



## Core Module Description

### 🧬 Evolutionary Algorithm Module (Algo.py)
- Implements genetic algorithms to optimize scoring rubrics
- Supports evolutionary operations including mutation, crossover, and selection
- Provides multiple fitness evaluation strategies

### 🤖 Model Evaluation Module
- **Cross-model_Kappa_glm_evolved.py**: Cross-rubric evaluation for GLM models
- **Cross-model_Kappa_doubao_evolved.py**: Cross-rubric evaluation for Doubao models

### 🧠 Chain of Thought (CoT) Module
- **cot_scorer.py**: Chain of thought-based essay scorer
- **cot_model_client.py**: CoT model client for handling AI model interactions
- **cot_logger.py**: Specialized logger for CoT scoring processes and results

### 📊 Data Processing Module (getData7.py)
- Data preprocessing and cleaning
- Training, validation, and test set partitioning
- Support for multiple data format reading

### 🔧 Utility Functions Module (utils.py)
- Thread-safe logging utilities
- Response caching mechanisms
- JSON parsing and processing tools
- General model interaction functions

## Experimental Design

### Model Comparison
- **Baseline Model**: Models using original scoring rubrics
- **EvoRubric Model**: Models using evolutionary algorithm-optimized scoring rubrics
- **APO Model**: Models using alternative parameter optimization methods

### Evaluation Metrics
- **Cohen's Kappa**: Inter-rater reliability coefficient
- **Mean Score Difference**: Difference between model scores and human scores
- **Score Distribution**: Visualization analysis of score distributions

### Task Characteristic Analysis
- **Essay Length**: Analysis of essay length impact on scoring accuracy
- **Essay Type**: Performance across different essay genres
- **Grade Level**: Scoring difficulty across different grade levels

## Dataset Description

The project uses the ASAP (Automated Student Assessment Prize) dataset, containing 8 different essay sets covering various topics, genres, and grade levels.

### Data File Description
- **Excel Files**: Contain experimental result summaries and analysis data
- **CSV Files**: Detailed scoring data and metrics
- **TXT Files**: Scoring rubrics and conversation records
- **Log Files**: Complete experimental execution records

## Quick Start

### Requirements
```bash
Python >= 3.8
pandas
numpy
scikit-learn
matplotlib
openpyxl
```

### Running Experiments
```bash
# Run main experiment
python Code/main.py

# Run GLM cross-model evaluation
python Code/Cross-model_Kappa_glm_evolved.py

# Run Doubao cross-model evaluation
python Code/Cross-model_Kappa_doubao_evolved.py
```

**Parameter Configuration**: To modify parameter configurations such as `essay_set`, `model`, etc., you can modify the corresponding parameters in the `get_args` function via command line configuration in each script file.

## Results Analysis

Experimental results are saved in various Excel files under the `Data/` directory, including:
- Cross-model scoring comparisons
- Model performance under different task characteristics
- Kappa coefficient changes during evolution
- Detailed scoring data and statistical analysis
...