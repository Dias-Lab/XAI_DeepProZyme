import logging
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

import pandas as pd
import lime
import lime.lime_tabular


# import torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F


def argument_parser(version=None):
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--seq_file', required=False, 
                        default='./Dataset/uniprot_dataset.fa', help='Sequence data')
    parser.add_argument('-o', '--output_dir', required=True, help='Output directory')
    parser.add_argument('-e', '--epoch', required=False, type=int,
                        default=30, help='Total epoch number')
    parser.add_argument('-b', '--batch_size', required=False, type=int,
                        default=32, help='Batch size')
    parser.add_argument('-r', '--learning_rate', required=False, type=float,
                        default=1e-3, help='Learning rate')
    parser.add_argument('-gamma', '--gamma', required=False, type=float,
                        default=1.0, help='Focal loss gamma')
    parser.add_argument('-p', '--patience', required=False, type=int,
                        default=5, help='Patience limit for early stopping')
    parser.add_argument('-g', '--gpu', required=False, 
                        default='cuda:0', help='Specify gpu')
    parser.add_argument('-cpu', '--cpu_num', required=False, type=int,
                        default=4, help='Number of cpus to use')  
    parser.add_argument('-ckpt', '--checkpoint', required=False, 
                        default='checkpoint.pt', help='Checkpoint file')
    parser.add_argument('-l', '--log_dir', required=False, 
                        default='CNN_training.log', help='Log file directory')
    parser.add_argument('-third', '--third_level', required=False, type=boolean_string,
                        default=False, help='Predict upto third EC level')      
    return parser



# plot the accuracy and loss value of each model.
###################
def draw(avg_train_losses, avg_valid_losses, output_dir, file_name='CNN_loss_fig.png'):
    fig = plt.figure(figsize=(9,6))

    avg_train_losses = np.array(avg_train_losses)
    avg_train_losses = avg_train_losses[avg_train_losses.nonzero()]
    avg_valid_losses = np.array(avg_valid_losses)
    avg_valid_losses = avg_valid_losses[avg_valid_losses.nonzero()]
    min_position = avg_valid_losses.argmin()+1

    plt.plot(range(1, len(avg_train_losses)+1), avg_train_losses, label='Training loss')
    plt.plot(range(1, len(avg_valid_losses)+1), avg_valid_losses, label='Validation loss')
    plt.axvline(min_position, linestyle='--', color='r', label='Early stopping checkpoint')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(left=0)
    plt.ylim(bottom=0, )

    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/{file_name}', dpi=600)
    plt.show()
    return


def save_losses(avg_train_losses, avg_valid_losses, output_dir, file_name='losses.txt'):
    with open(f'{output_dir}/{file_name}', 'w') as fp:
        fp.write('Epoch\tAverage_train_loss\tAverage_valid_loss\n')
        cnt = 0
        for train_loss, valid_loss in zip(avg_train_losses, avg_valid_losses):
            cnt += 1
            fp.write(f'{cnt}\t{train_loss:0.12f}\t{valid_loss:0.12f}\n')
    return


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha==None:
            self.alpha=1
        else:
            self.alpha = torch.Tensor(alpha).view(-1, 1)
        
    def forward(self, pred, label):
        BCE_loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()


class DeepECConfig():
    def __init__(self,
                 model = None,
                 optimizer = None,
                 criterion = None,
                 scheduler = None,
                 n_epochs = 50,
                 device = 'cpu',
                 patience = 5,
                 save_name = './deepec.log',
                 train_source = None,
                 val_source = None, 
                 test_source = None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.n_epochs = n_epochs
        self.device = device
        self.patience = patience
        self.save_name = save_name
        self.train_source = train_source
        self.val_source = val_source
        self.test_source = test_source




def run_neural_net(model, proteinDataloader, pred_thrd, device):
        num_data = len(proteinDataloader.dataset)
        num_ecs = len(proteinDataloader.dataset.map_EC)
        pred_thrd = pred_thrd.to(device)
        model.eval() # training session with train dataset
        with torch.no_grad():
            y_pred = torch.zeros([num_data, num_ecs])
            y_score = torch.zeros([num_data, num_ecs])
            logging.info('Deep leanrning prediction starts on the dataset')
            cnt = 0
            for batch, data in enumerate(tqdm(proteinDataloader)):
                inputs = {key:val.to(device) for key, val in data.items()}
                output = model(**inputs)
                output = torch.sigmoid(output)
                prediction = output > pred_thrd
                prediction = prediction.float()
                step = data['input_ids'].shape[0]
                y_pred[cnt:cnt+step] = prediction.cpu()
                y_score[cnt:cnt+step] = output.cpu()
                cnt += step
            logging.info('Deep learning prediction ended on test dataset')
        return y_pred, y_score


def save_dl_result(y_pred, y_score, input_ids, explainECs, output_dir):
    failed_cases = []
    with open(f'{output_dir}/DL_prediction_result.txt', 'w') as fp:
        fp.write('sequence_ID\tprediction\tscore\n')
        for i, ith_pred in enumerate(y_pred):
            nonzero_preds = torch.nonzero(ith_pred, as_tuple=False)
            if len(nonzero_preds) == 0:
                fp.write(f'{input_ids[i]}\tNone\t0.0\n')
                failed_cases.append(input_ids[i])
                continue
            for j in nonzero_preds:
                pred_ec = explainECs[j]
                pred_score = y_score[i][j].item()
                fp.write(f'{input_ids[i]}\t{pred_ec}\t{pred_score:0.4f}\n')
    return failed_cases


class MultilabelLIMEExplainer:
    def __init__(self, model, device, feature_names, class_names):
        """
        Wrapper for LIME Tabular Explainer to handle multilabel classification.

        Args:
            model: The PyTorch model to explain.
            device: Device to run the model on (CPU/GPU).
            feature_names: List of feature names.
            class_names: List of class (label) names.
        """
        self.model = model
        self.device = device
        self.feature_names = feature_names
        self.class_names = class_names

    def make_classifier_pipeline(self, label_index):
        """
        Creates a classifier pipeline for a specific label.

        Args:
            label_index: Index of the label to explain.

        Returns:
            A prediction function for LIME that outputs probabilities for the specific label.
        """
        def lime_explainer_pipeline(input_data):
            input_tensor = torch.tensor(input_data).to(self.device).long()
            with torch.no_grad():
                outputs = self.model(input_ids=input_tensor)
                outputs = torch.sigmoid(outputs)  # Multilabel probabilities
                prob_true = outputs[:, label_index].cpu().numpy()
                prob_false = 1 - prob_true
                return np.vstack([prob_false, prob_true]).T  # Return probabilities as [P(false), P(true)]

        return lime_explainer_pipeline

    def get_highest_probability_label(self, instance):
        input_tensor = torch.tensor(instance).unsqueeze(0).to(self.device).long()
        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor)
            probabilities = torch.sigmoid(outputs)
        return probabilities.argmax().item()

    def explain_instance(self, instance, explainer, num_features=10, num_samples=100):
        highest_prob_index = self.get_highest_probability_label(instance)
        label_name = self.class_names[highest_prob_index]
        classifier_fn = self.make_classifier_pipeline(highest_prob_index)
        exp = explainer.explain_instance(
            instance,
            classifier_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        return {label_name: dict(exp.local_exp[1])}


    def explain_instanc_all_labels(self, instance, explainer, num_features=10, num_samples=500):
        """
        Explains an instance for all labels using LIME.

        Args:
            instance: Single data instance to explain.
            explainer: LIME Tabular Explainer object.
            num_features: Number of features to include in the explanation.
            num_samples: Number of samples to generate in LIME.

        Returns:
            Dictionary of explanations for each label.
        """
        explanations = {}
        
        # Loop through each label and explain it
        for label_index, label_name in enumerate(self.class_names):
            classifier_fn = self.make_classifier_pipeline(label_index)
            
            exp = explainer.explain_instance(
                instance,
                classifier_fn,
                num_features=num_features,
                num_samples=num_samples
            )
            
            explanations[label_name] = dict(exp.local_exp[1])  # Use index `1` because it's the "true" class
        
        return explanations

def run_neural_net_with_xai(
    model,
    proteinDataloader,
    pred_thrd,
    device,
    num_features=10,
    num_samples=500,
    output_dir='./lime_plots_multilabel'
):
    os.makedirs(output_dir, exist_ok=True)

    num_data = len(proteinDataloader.dataset)
    num_ecs = len(model.explainECs)  # Use the number of labels from model.explainECs
    pred_thrd = pred_thrd.to(device)

    model.eval()

    y_pred = torch.zeros([num_data, num_ecs])
    y_score = torch.zeros([num_data, num_ecs])

    logging.info('Deep learning prediction starts on the dataset')

    cnt = 0
    all_inputs = []
    local_explanations_df = pd.DataFrame()

    with torch.no_grad():
        for batch, data in enumerate(tqdm(proteinDataloader)):
            inputs = {key: val.to(device) for key, val in data.items()}
            output = model(**inputs)
            output = torch.sigmoid(output)  # Use sigmoid for multilabel classification

            prediction = output > pred_thrd
            prediction = prediction.float()
            step = data['input_ids'].shape[0]
            y_pred[cnt:cnt+step] = prediction.cpu()
            y_score[cnt:cnt+step] = output.cpu()

            all_inputs.append(inputs['input_ids'].cpu())

            cnt += step

    logging.info('Deep learning prediction ended on test dataset')

    all_inputs_tensor = torch.cat(all_inputs, dim=0).cpu().numpy()

    # Initialize feature and class names
    feature_names = [f"Feature_{i}" for i in range(all_inputs_tensor.shape[1])]
    ec_names = model.explainECs  # Use label names directly from model.explainECs

    # Initialize Multilabel LIME Explainer
    lime_explainer_wrapper = MultilabelLIMEExplainer(
        model=model,
        device=device,
        feature_names=feature_names,
        class_names=ec_names
    )

    # Create a LIME Tabular Explainer object (shared across all labels)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        all_inputs_tensor,
        feature_names=feature_names,
        class_names=["None", "True"],  # Binary classification per label
        mode='classification'
    )

    logging.info('Computing LIME explanations...')

    global_importance = np.zeros((num_ecs, len(feature_names)))
    lime_explanations = []

    for i in tqdm(range(min(500, len(all_inputs_tensor)))):
        explanations_per_instance = lime_explainer_wrapper.explain_instance(
            instance=all_inputs_tensor[i],
            explainer=explainer,
            num_features=num_features,
            num_samples=num_samples
        )
    
        lime_explanations.append(explanations_per_instance)


        # Save local explanation for this instance to the DataFrame
        instance_expl_df = pd.DataFrame({ 
            "Instance": [i] * len(feature_names), 
            "Feature": feature_names, 
            "Importance": [explanations_per_instance[next(iter(explanations_per_instance))].get(f_idx, 0) for f_idx in range(len(feature_names))]
        })
        local_explanations_df = pd.concat([local_explanations_df, instance_expl_df], ignore_index=True)


        # Aggregate global importance for the highest probability label
        label_name, importance_dict = next(iter(explanations_per_instance.items()))
        ec_idx = ec_names.index(label_name)
        for feature_idx, value in importance_dict.items():
            global_importance[ec_idx, feature_idx] += abs(value)


    logging.info('LIME explanations computed.')

    global_importance /= len(lime_explanations)

    # Save results and plots per label (EC)
    #for ec_idx, ec_name in enumerate(ec_names):
    #    importance_df = pd.DataFrame({
    #        'Feature': feature_names,
    #        'Importance': global_importance[ec_idx]
    #    })

    # Create a single large DataFrame
    importance_df = pd.DataFrame(global_importance, index=ec_names, columns=feature_names)
    
    # Sort columns by the sum of importance across all ECs
    importance_df = importance_df.sort_values(by=importance_df.index.tolist(), axis=1, ascending=False)
    
    # Save global explanations
    csv_path = os.path.join(output_dir, "global_feature_importance_all_ECs.csv")
    importance_df.to_csv(csv_path)

    # Save local explanations
    csv_path_local = os.path.join(output_dir, "local_explanations.csv") 
    local_explanations_df.to_csv(csv_path_local, index=False)
    logging.info(f"Local explanations saved to {csv_path_local}")

    logging.info(f"Global feature importance for all ECs saved to {csv_path}")

    return y_pred, y_score, lime_explanations
