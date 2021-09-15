import pandas as pd
import os,sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class EarlyStopping:
    """Early stops the training if validation loss and validation accuracy don't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='models/checkpoint.pt', trace_func=print, monitor='val_loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print         
            monitor (Mode): If val_loss, stop at maximum mode, else val_accuracy, stop at minimum mode
                            Default: val_loss   
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = monitor
        

    def __call__(self, values, model):

        if self.mode == 'val_loss':
            score = -values

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(values, model)
            elif score <= self.best_score + self.delta:
                self.counter += 1
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(values, model)
                self.counter = 0
        else:
            score = values
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(values, model)
            elif score <= self.best_score + self.delta:
                self.counter += 1
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(values, model)
                self.counter = 0

    def save_checkpoint(self, values, model):
        '''Saves model when validation loss decrease.'''
        if self.mode == 'val_loss':
            if self.verbose:
                self.trace_func(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {values:.6f}).   Saving model to {self.path}')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = values
        elif self.mode == 'val_accuracy':
            if self.verbose:
                self.trace_func(
                    f'Validation accuracy increased ({self.val_acc_max:.3f} --> {values:.3f}).  Saving model to {self.path}')
            torch.save(model.state_dict(), self.path)
            self.val_acc_max = values


def preprocess(data_dir, csv_dir,train_val_split=0.3,train_val_split_status=False,custom_val=False):
    """
    Get training dataframe and testing dataframe from image directory and
    csv description file.

    Args:
        data_dir (String): Directory of image data
        csv_dir (String): Directory of csv description file

    Returns:
        df_train (pandas.DataFrame): Data frame of training set
        df_test (pandas.DataFrame):  Data frame of test set
    """
    data_name = os.listdir(data_dir)
    url_dataframe = pd.read_csv(csv_dir)

    url_dataframe["image_name"] = [str(x) + ".jpg" for x in url_dataframe["image_name"]]

    total_label = url_dataframe["target"]
    total_name = url_dataframe["image_name"]

    if train_val_split_status:
        name_train, name_test, label_train, label_test = train_test_split(
            total_name, total_label, test_size=train_val_split, random_state=42)

        data_train = {'Name': name_train,
                    'Label': label_train
                    }

        data_test = {'Name': name_test,
                    'Label': label_test
                    }

        df_train = pd.DataFrame(data_train)
        df_test = pd.DataFrame(data_test)

        return df_train, df_test
    elif custom_val:
        df_0 = url_dataframe.groupby('target').get_group(0)
        df_1 = url_dataframe.groupby('target').get_group(1)

        df_0_train,df_0_test = train_test_split(df_0,test_size=0.1,random_state=15)
        df_1_train,df_1_test = train_test_split(df_1,test_size=0.2,random_state=15)

        df_train = pd.concat([df_0_train,df_1_train])
        df_test = pd.concat([df_0_test,df_1_test])


        train_target = df_train['target']
        test_target = df_test['target']
        train_data = df_train['image_name']
        test_data = df_test['image_name']

        data_train = {'Name': train_data,
                    'Label': train_target
                    }

        data_test = {'Name': test_data,
                    'Label': test_target
                    }

        df_train = pd.DataFrame(data_train)
        df_test = pd.DataFrame(data_test)

        return df_train, df_test
        # df_train = pd.concat([df_0_train,df_1_train])



    else: 
        data = {'Name': total_name,
                    'Label': total_label} 

        df_data = pd.DataFrame(data)

        return df_data

# Block
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def calculate_metrics(out_gt, out_pred):
    """
    Calculate methics for model evaluation

    Args:
        out_gt (torch.Tensor)   : Grouth truth array
        out_pred (torch.Tensor) : Prediction array

    Returns:
        accuracy (float)    : Accuracy
        precision (float)   : Precision
        recall (float)      : Recall
        f1_score (float)    : F1 Score
        sensitivity (float) : Sensitivity
        specificity (float) : Specificity

    """
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    for i in range(len(out_gt)):
        if ((out_gt[i] == 1) and (out_pred[i] == 1)):
            true_positives += 1
        if ((out_gt[i] == 0) and (out_pred[i] == 0)):
            true_negatives += 1
        if ((out_gt[i] == 0) and (out_pred[i] == 1)):
            false_positives += 1
        if ((out_gt[i] == 1) and (out_pred[i] == 0)):
            false_negatives += 1

    accuracy = (true_positives + true_negatives) / (true_positives +
                                                    true_negatives + false_positives + false_negatives)

    precision = true_positives / \
        (true_positives + false_positives + np.finfo(float).eps)
    recall = true_positives / \
        (true_positives + false_negatives + np.finfo(float).eps)

    f1_score = 2 * precision * recall / \
        (precision + recall + np.finfo(float).eps)

    sensitivity = recall
    specificity = true_negatives / \
        (true_negatives + false_positives + np.finfo(float).eps)

    return accuracy, precision, recall, f1_score, sensitivity, specificity


if __name__ == '__main__':
    df_train,df_test = preprocess('/mnt/data_lab513/dhsang/data/256x256', '../csvFile/train.csv',train_val_split_status=False,custom_val=True)
    print(df_train.head())
    print(df_test.head())
