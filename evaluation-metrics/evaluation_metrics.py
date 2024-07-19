import cc3d
import nibabel as nib
import numpy as np


def nii2numpy(nii_path: str) -> [np.ndarray, float]:
    """Load a nifti file and extract the voxel volume in mm^3"""
    mask_nii = nib.load(nii_path)
    mask = mask_nii.get_fdata()
    pixdim = mask_nii.header["pixdim"]
    voxel_vol = pixdim[1] * pixdim[2] * pixdim[3] / 1000
    return mask, voxel_vol


def count_false_positives(*, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Count the number of false positive pixel, which do not overlap with the ground truth, based on the prediction
    and ground truth arrays. Returns zero if the prediction array is empty.
    """
    if prediction.sum() == 0:
        return 0

    connected_components = cc3d.connected_components(
        prediction.astype(int), connectivity=18
    )
    false_positives = 0

    for idx in range(1, connected_components.max() + 1):
        component_mask = np.isin(connected_components, idx)
        if (component_mask * ground_truth).sum() == 0:
            false_positives += component_mask.sum()

    return float(false_positives)


def count_false_negatives(*, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Count the number of false negative pixel, which do not overlap with the ground truth, based on the prediction
    and ground truth arrays. Returns nan if the ground truth array is empty.
    """
    if ground_truth.sum() == 0:
        return np.nan

    gt_components = cc3d.connected_components(ground_truth.astype(int), connectivity=18)
    false_negatives = 0

    for component_id in range(1, gt_components.max() + 1):
        component_mask = np.isin(gt_components, component_id)
        if (component_mask * prediction).sum() == 0:
            false_negatives += component_mask.sum()

    return float(false_negatives)


def calc_dice_score(*, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate the Dice score between the prediction and ground truth arrays.
    Returns nan if the ground truth array is empty.
    """
    if ground_truth.sum() == 0:
        return np.nan

    intersection = (ground_truth * prediction).sum()
    union = ground_truth.sum() + prediction.sum()
    dice_score = 2 * intersection / union

    return float(dice_score)


def compute_metrics(nii_gt_path: str, nii_pred_path: str) -> [float, float, float]:
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, _ = nii2numpy(nii_pred_path)

    false_neg_vol = (
        count_false_negatives(prediction=pred_array, ground_truth=gt_array) * voxel_vol
    )
    false_pos_vol = (
        count_false_positives(prediction=pred_array, ground_truth=gt_array) * voxel_vol
    )
    dice_sc = calc_dice_score(prediction=pred_array, ground_truth=gt_array)

    return dice_sc, false_pos_vol, false_neg_vol


if __name__ == "__main__":
    nii_gt_path = "../test/orig/psma_95b833d46f153cd2_2018-04-16.nii.gz"
    nii_pred_path = "../test/expected_output_dc/PRED.nii.gz"
    dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, nii_pred_path)
    print(
        f"DiceScore: {dice_sc} \nFalsePositiveVolume: {false_pos_vol} ml \nFalseNegativeVolume: {false_neg_vol} ml"
    )
