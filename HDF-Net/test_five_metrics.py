import os
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from tqdm import tqdm

def preprocess_gt(gt_path):
    # Read GT image and normalize pixel values to 0 or 1
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_img_normalized = gt_img / 255.0
    return gt_img_normalized

def preprocess_pred(pred_path):
    # Read pred image and normalize pixel values to 0-1 range
    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    pred_img_normalized = pred_img / 255.0
    return pred_img_normalized

def calculate_iou(gt_flat, pred_binary):
    intersection = np.logical_and(gt_flat, pred_binary).sum()
    union = np.logical_or(gt_flat, pred_binary).sum()
    iou = intersection / (union + 1e-8)  # Add a small epsilon to avoid division by zero
    return iou

def calculate_fpr(gt_flat, pred_binary):
    true_negative = np.logical_and(np.logical_not(gt_flat), np.logical_not(pred_binary)).sum()
    false_positive = np.logical_and(np.logical_not(gt_flat), pred_binary).sum()
    fpr = false_positive / (false_positive + true_negative + 1e-8)  # Add a small epsilon to avoid division by zero
    return fpr

def calculate_auc_iou_fpr(gt_folder, pred_folder):
    gt_files = os.listdir(gt_folder)
    pred_files = os.listdir(pred_folder)
    num_images = len(gt_files)

    auc_scores = []  # AUC scores for each pred image
    f1_scores = []  # F1 scores for each pred image
    mcc_scores = []  # MCC scores for each pred image
    iou_scores = []  # IoU scores for each pred image
    fpr_scores = []  # FPR scores for each pred image

    for i in tqdm(range(num_images)):
        gt_path = os.path.join(gt_folder, gt_files[i])
        pred_path = os.path.join(pred_folder, pred_files[i])

        # Preprocess GT and pred images
        gt_img_normalized = preprocess_gt(gt_path)
        pred_img_normalized = preprocess_pred(pred_path)

        # Flatten the images to 1D arrays
        gt_flat = gt_img_normalized.flatten()
        pred_flat = pred_img_normalized.flatten()
        gt_flat = gt_flat.astype(int)

        # Calculate AUC for the current pred image
        auc_score = roc_auc_score(gt_flat, pred_flat)
        if auc_score < 0.5:
            auc_score = 1 - auc_score
        auc_scores.append(auc_score)

        # Calculate F1 score for the current pred image
        threshold = 0.5  # Set the threshold for binarization
        pred_binary = (pred_flat > threshold).astype(int)
        f1 = f1_score(gt_flat, pred_binary)
        f1_scores.append(f1)

        # Calculate MCC for the current pred image
        mcc = matthews_corrcoef(gt_flat, pred_binary)
        mcc_scores.append(mcc)

        # Calculate IoU for the current pred image
        iou = calculate_iou(gt_flat, pred_binary)
        iou_scores.append(iou)

        # Calculate FPR for the current pred image
        fpr = calculate_fpr(gt_flat, pred_binary)
        fpr_scores.append(fpr)

        print(f"Image {gt_files[i]} - AUC: {auc_score:.4f}, F1: {f1:.4f}, MCC: {mcc:.4f}, IoU: {iou:.4f}, FPR: {fpr:.4f}")

    # Calculate the average AUC, F1, MCC, IoU, and FPR
    average_auc = np.mean(auc_scores)
    average_f1 = np.mean(f1_scores)
    average_mcc = np.mean(mcc_scores)
    average_iou = np.mean(iou_scores)
    average_fpr = np.mean(fpr_scores)

    print("Average AUC:", average_auc)
    print("Average F1:", average_f1)
    print("Average MCC:", average_mcc)
    print("Average IoU:", average_iou)
    print("Average FPR:", average_fpr)

    return average_auc, average_f1, average_mcc, average_iou, average_fpr

if __name__ == "__main__":
    gt_folder_path = r"F:\ImageManipulationDatasets\Columbia\mask"  # Replace with the path to the folder containing GT images
    pred_folder_path = r"E:\Columbia_gray"  # Replace with the path to the folder containing pred images

    (
        average_auc,
        average_f1,
        average_mcc,
        average_iou,
        average_fpr
    ) = calculate_auc_iou_fpr(gt_folder_path, pred_folder_path)