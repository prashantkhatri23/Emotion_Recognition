# Multi-Modal Multi-Focus Emotion Recognition

## Overview of our approach
1. Acoustic frame-level features and lexical word embeddings are used as input for separate BLSTMs.
2. An context based attention module is applied to pool the outputs of the BLSTMs and generate an utterance-level temporal aggregate.
3. The temporal aggregates from the two models are fused using an attention-based modality fusion module.
4. The fused output is passed through a linear softmax layer to get the classification probabilities.



