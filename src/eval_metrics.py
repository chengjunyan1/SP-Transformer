import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    loginfo=''
    loginfo+="F1 score: "+str(f_score)+'\n'
    loginfo+="Accuracy: "+str(accuracy_score(binary_truth, binary_preds))+'\n'
    loginfo+="-" * 50+'\n'
    return f_score,loginfo

def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)

def eval_ur_funny(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    binary_truth = (test_truth > 0.5)
    binary_preds = (test_preds > 0)
    accu=accuracy_score(binary_truth, binary_preds)

    loginfo=''
    loginfo+="Accuracy: "+str(accu)+'\n'
    loginfo+="-" * 50+'\n'
    return accu,loginfo
