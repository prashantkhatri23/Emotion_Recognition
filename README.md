# Multi-Modal Multi-Focus Emotion Recognition

This repository contains the code for processing and analyzing the IEMOCAP emotion dataset (Interactive Emotional Dyadic Motion Capture Database). The dataset contains audio files of emotionally expressive speech and corresponding metadata. The code in this repository processes the metadata, extracts features from the audio files, and prepares the dataset for further analysis.

## Overview of our approach
1. Acoustic frame-level features and lexical word embeddings are used as input for separate BLSTMs.
2. A context-based attention module is applied to pool the outputs of the BLSTMs and generate an utterance-level temporal aggregate.
3. The temporal aggregates from the two models are fused using an attention-based modality fusion module.
4. The fused output is passed through a linear softmax layer to get the classification probabilities.

## Dataset

The IEMOCAP dataset is not included in this repository. You need to obtain the dataset from the [official website](https://sail.usc.edu/iemocap/). After obtaining the dataset, unzip it and place it in the root directory of the project.

## Downloading the Large Data File

The data file `data_processed.pkl` (670MB) is too large to be uploaded to GitHub. This file is created after running the `DataProcessing.ipynb` notebook (on the IEMOCAP dataset), which preprocesses the data and extracts features. You can either run the notebook to create the pickle file or download it from Google Drive using the following link:

[Download data_processed.pkl from Google Drive](<https://drive.google.com/file/d/18lCvVF1T9yE884QmQtFwlw7E8id9DlBY/view?usp=share_link>)

## Usage

Once you've either created or downloaded the `data_processed.pkl` file and placed it in your project directory, you can load the data using the following code:

```python
import pandas as pd

# Load the DataFrame from the pickle file
data = pd.read_pickle("data_processed.pkl")
```

## Notebooks

### 1. DataProcessing.ipynb

Description:
- Explore dataset structure.
- Process metadata and transcripts.
- Refine emotion categories.
- Extract LLD features with OpenSMILE.
- Remove samples with non-text characters.

Results:
- The final dataset will be stored in the data DataFrame and is ready for further processing and analysis.

### 2. UnimodalClassifier.ipynb

Description:
- [Add the description of the notebook here]

Results:
- [Add the results of the notebook here]

### 3. MMClassifier.ipynb

Description:
- [Add the description of the notebook here]

Results:
- [Add the results of the notebook here]

### 4. Analysis.ipynb

Description:
- [Add the description of the notebook here]

Results:
- [Add the results of the notebook here]

### 5. Demo.ipynb

Description:
- [Add the description of the notebook here]

Results:
- [Add the results of the notebook here]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

