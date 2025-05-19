# eXplainable AI (XAI)-enabled DeepECtransformer


To enhance the interpretability of DeepECTransformer’s multilabel enzyme function predictions, we incorporated an explainable artificial intelligence (XAI) module based on the Local Interpretable Model-agnostic Explanations (LIME) framework83. The approach was specifically adapted to address the multilabel nature of EC number predictions and to provide both local and global insights into the protein sequence segments or residues driving model decisions. Traditional LIME is designed for binary or multiclass outputs, but enzyme function prediction often involves multilabel assignments. To address this, we implemented a multilabel adaptation of LIME, which constructs independent explanation pipelines for each possible label. For each protein sequence input, the XAI module isolates the probability output for a single EC label using the model’s sigmoid activation, enabling LIME to generate label-specific explanations. To optimize computational efficiency, explanations are generated primarily for the label with the highest predicted probability for each input sequence, focusing interpretability efforts on the most relevant predictions.

To balance explanation quality with computational feasibility, local LIME explanations are computed for a representative subsample of up to 500 protein sequences per analysis batch. For each sequence, the module generates a local feature importance map, highlighting which residues in the protein sequence most influenced the model’s prediction for the selected EC label. For proteins outside this subsample, residue-level importance is estimated based on the aggregate statistics from the explained subset. This approach enables the extraction of both local (residue-level) and global (dataset-level) feature importance profiles, which are subsequently exported for downstream analysis. To further dissect model behavior, feature importance scores derived from LIME explanations were stratified by prediction type (correct predictions, paralog errors, non-paralog errors, and repetitions). Each protein sequence is mapped to a prediction category based on expert curation of the model outputs. Residue-level importance values are normalized within each error type, and summary plots are generated to visualize the distribution and magnitude of important sequence segments across different error categories.


## Procedure

**Note**: 
This source code was developed in Linux, and has been tested in Ubuntu 16.04 with Python 3.6.

1. Clone the repository

        git clone git@github.com:Dias-Lab/XAI_DeepProZyme.git

2. Create and activate virtual environment (takes 5~10 min)

        conda env create -f environment.yml
        conda activate deepectransformer

3. To use gpus properly, install the pytorch and cuda for your gpus. This code was tested on pytorch=1.7.0 using cuda version 10.2



## Example


- Run DeepECtransformer (takes < 1 min)

        python run_deepectransformer.py -i ./example/mdh_ecoli.fa -o ./example/results -g cpu -b 128 -cpu 2
        python run_deepectransformer.py -i ./example/mdh_ecoli.fa -o ./example/results -g cuda:3 -b 128 -cpu 2

## XAI results


After running the XAI-enabled DeepECtransformer, XAI results will be exported into the **xai_results** directory.

The XAI module is fully integrated into the DeepECTF workflow, allowing users to generate explainability outputs alongside standard predictions without additional user intervention. All explanation results, including local and global feature importance scores, are made available in standard tabular formats for further interpretation or visualization. This explainable AI approach provides actionable insights into the decision-making process of DeepECTF, supporting both the validation of correct predictions and the systematic investigation of model failures, facilitating the development of more robust and trustworthy protein function prediction systems.
