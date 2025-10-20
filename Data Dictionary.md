# EvoRubric Data Dictionary

## Overview
This data dictionary describes the dataset structure used for essay scoring research in the EvoRubric project. The dataset contains multiple Excel files and related CSV and text files for analyzing the performance of different AI models in essay scoring tasks.

## File Structure

### Main Excel Files

#### 1. Cross-model scoring.xlsx
**File Description**: Cross-model scoring results summary
**Data Elements**:
- `essay_set` - Essay set number
- `Baseline(glm)` - GLM baseline model Kappa coefficient
- `glm-evolved(doubao rubric)` - GLM model Kappa coefficient evolved using Doubao scoring rubric
- `Baseline(doubao)` - Doubao baseline model Kappa coefficient
- `doubao-evolved(glm rubric)` - Doubao model Kappa coefficient evolved using GLM scoring rubric

#### 2. Essay length-kappa.xlsx
**File Description**: Analysis of relationship between essay length and Kappa coefficient
**Data Elements**:
- `Average_length_of_essays` - Average essay length
- `Number_of_essays` - Number of essays
- `Baseline(glm)` - GLM baseline model Kappa coefficient
- `EvoRubric(glm)` - GLM evolved model Kappa coefficient
- `Baseline(doubao)` - Doubao baseline model Kappa coefficient
- `EvoRubric(doubao)` - Doubao evolved model Kappa coefficient

#### 3. Essay set describtion.xlsx
**File Description**: Essay set description information
**Data Elements**:
- `essay_set` - Essay set number
- `Type_of_essay` - Essay type
- `Grade_level` - Grade level
- `Average_length_of_essays` - Average essay length

#### 4. Essay type-kappa.xlsx
**File Description**: Relationship between essay type and Kappa coefficient
**Data Elements**:
- `Type_of_essay` - Essay type
- `Number_of_essays` - Number of essays
- `Baseline(glm)` - GLM baseline model Kappa coefficient
- `EvoRubric(glm)` - GLM evolved model Kappa coefficient
- `Baseline(doubao)` - Doubao baseline model Kappa coefficient
- `EvoRubric(doubao)` - Doubao evolved model Kappa coefficient

#### 5. Grade level-kappa.xlsx
**File Description**: Relationship between grade level and Kappa coefficient
**Data Elements**:
- `Grade_level` - Grade level
- `Number_of_essays` - Number of essays
- `Baseline(glm)` - GLM baseline model Kappa coefficient
- `EvoRubric(glm)` - GLM evolved model Kappa coefficient
- `Baseline(doubao)` - Doubao baseline model Kappa coefficient
- `EvoRubric(doubao)` - Doubao evolved model Kappa coefficient

#### 6. Parent-Kappa.xlsx
**File Description**: Parent Kappa coefficient evolution process data
**Data Elements**:
- `Model` - Model name
- `essay_set` - Essay set number
- `Gen` - Evolution generation
- `Kappa_1` - 1st Kappa coefficient
- `Kappa_2` - 2nd Kappa coefficient
- `Kappa_3` - 3rd Kappa coefficient
- `Kappa_4` - 4th Kappa coefficient
- `Kappa_5` - 5th Kappa coefficient

#### 7. Rubric.xlsx
**File Description**: Scoring rubric template data
**Data Elements**:
- `essay_set` - Essay set number
- `官方评分标准` - Official scoring rubric
- `最佳评分标准(glm)` - Best scoring rubric for GLM model
- `最佳评分标准(doubao)` - Best scoring rubric for Doubao model

#### 8. Score(test set).xlsx
**File Description**: Test set scoring data
**Data Elements**:
- `essay_set` - Essay set number
- `essay` - Essay content
- `human_score` - Human score
- `base_score(glm)` - GLM baseline model score
- `base_score(doubao)` - Doubao baseline model score
- `evolved_score(glm)` - GLM evolved model score
- `evolved_score(doubao)` - Doubao evolved model score
- `APO(glm)` - GLM model APO score
- `APO(doubao)` - Doubao model APO score
- `CoT_score(glm)` - GLM model Chain of Thought score
- `CoT_score(doubao)` - Doubao model Chain of Thought score

#### 9. Test_set-kappa.xlsx
**File Description**: Test set Kappa coefficient data
**Data Elements**:
- `essay_set` - Essay set number
- `Baseline(glm)` - GLM baseline model Kappa coefficient
- `EvoRubric(glm)` - GLM evolved model Kappa coefficient
- `APO(glm)` - GLM model APO method Kappa coefficient
- `Baseline(doubao)` - Doubao baseline model Kappa coefficient
- `EvoRubric(doubao)` - Doubao evolved model Kappa coefficient
- `APO(doubao)` - Doubao model APO method Kappa coefficient
- `CoT(glm)` - GLM model Chain of Thought method Kappa coefficient
- `CoT(doubao)` - Doubao model Chain of Thought method Kappa coefficient



### Data Quality Notes

#### Completeness
- All Excel files contain complete column headers
- CSV file data is complete with no missing values
- Text files are encoded in UTF-8

#### Consistency
- essay_set field maintains consistent integer format across all files
- Scoring fields uniformly use numeric data types
- Kappa coefficients use floating-point format, ranging from -1 to 1


### Terminology

- **Baseline**: Baseline model, original model without evolutionary algorithm optimization
- **EvoRubric**: Model optimized through evolutionary algorithms
- **APO**: Alternative Parameter Optimization method
- **GLM**: GLM large language model
- **Doubao**: Doubao model
- **Kappa Coefficient**: Cohen's Kappa, a statistical measure for inter-rater reliability
