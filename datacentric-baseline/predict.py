import os
import random
import time
from typing import Dict, List, Optional, Union

import monai.transforms as mt
import numpy as np
import torch
from autopet3.fixed.dynunet import NNUnet
from autopet3.fixed.evaluation import AutoPETMetricAggregator
from autopet3.fixed.utils import plot_results


class PredictModel:
    """The PredictModel class is responsible for preprocessing input data, loading and evaluating models, and making
    predictions using an ensemble of models. It also supports test-time augmentation (TTA) for improved predictions.
    The class can run the prediction pipeline on CT and PET image files, save the output, and optionally evaluate the
    results.

    Example Usage
    # Create an instance of PredictModel
    model_paths = ["model1.ckpt", "model2.ckpt", "model3.ckpt"]
    predictor = PredictModel(model_paths, sw_batch_size=6, tta=True, random_flips=2)

    # Run the prediction pipeline
    ct_file_path = "ct_image.nii.gz"
    pet_file_path = "pet_image.nii.gz"
    label = "label_image.nii.gz"
    save_path = "output_folder"
    metrics = predictor.run(ct_file_path, pet_file_path, label=label, save_path=save_path, verbose=True)
    """

    def __init__(
        self,
        model_paths: List[str],
        sw_batch_size: int = 6,
        tta: bool = False,
        random_flips: int = 0,
        dynamic_tta: bool = False,
        max_tta_time: int = 300,
    ):
        """Initialize the class with the given parameters.
        Args:
                model_paths (List[str]): List of model paths.
                sw_batch_size (int, optional): Batch size for the model. Defaults to 6.
                tta (bool, optional): Flag for test-time augmentation. Defaults to False.
                random_flips (int, optional): Number of random flips. Defaults to 0.
        Returns:
                None

        """
        self.ckpts = model_paths
        self.transform = None
        self.sw_batch_size = sw_batch_size
        self.tta = tta
        self.dynamic_tta = dynamic_tta
        self.max_tta_time = max_tta_time
        self.tta_flips = [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        self.random_flips = np.clip(random_flips, 0, len(self.tta_flips))

    @staticmethod
    def preprocess(input: Dict[str, str]) -> torch.Tensor:
        """Preprocesses the input data by applying a series of transformations to the CT and PET images.
        Args:
                input (Dict[str, str]): A dictionary containing the paths to the CT and PET images.
        Returns:
                torch.Tensor: The preprocessed input data as a tensor with a batch dimension.

        """
        # Define the percentile values for CT and PET images
        ct_percentiles = (-831.9483642578125, 1127.5189013671843)
        ct_norm = torch.Tensor((107.77672799269146, 286.1725548247846))
        pet_percentiles = (1.0438873767852783, 51.595245361328125)
        pet_norm = torch.Tensor((7.089671984867433, 8.01080984806995))

        # Define the spacing for the images
        spacing = (2.0364201068878174, 2.03642010688781740, 3.0)

        # Define a list of transforms
        keys = ["ct", "pet"]
        transforms = [
            mt.LoadImaged(keys=keys),
            mt.EnsureChannelFirstd(keys=keys),
            mt.EnsureTyped(keys=keys),
            mt.Orientationd(keys=keys, axcodes="LAS"),
            mt.Spacingd(keys=keys, pixdim=spacing, mode="bilinear",),
            mt.ScaleIntensityRanged(
                keys=["ct"],
                a_min=ct_percentiles[0],
                a_max=ct_percentiles[1],
                b_min=0,
                b_max=1,
                clip=True,
            ),
            mt.ScaleIntensityRanged(
                keys=["pet"],
                a_min=pet_percentiles[0],
                a_max=pet_percentiles[1],
                b_min=0,
                b_max=1,
                clip=True,
            ),
            mt.ScaleIntensityRanged(
                keys=["ct"],
                a_min=0,
                a_max=1,
                b_min=ct_percentiles[0],
                b_max=ct_percentiles[1],
                clip=True,
            ),
            mt.ScaleIntensityRanged(
                keys=["pet"],
                a_min=0,
                a_max=1,
                b_min=pet_percentiles[0],
                b_max=pet_percentiles[1],
                clip=True,
            ),
            mt.NormalizeIntensityd(
                keys=["ct"], subtrahend=ct_norm[0], divisor=ct_norm[1]
            ),
            mt.NormalizeIntensityd(
                keys=["pet"], subtrahend=pet_norm[0], divisor=pet_norm[1]
            ),
            mt.ConcatItemsd(keys=keys, name="image", dim=0),
            mt.EnsureTyped(keys=keys),
            mt.ToTensord(keys=keys),
        ]

        # Compose and apply the transforms, add batch dimension
        transform = mt.Compose(transforms)
        output = transform(input)["image"][None, ...]
        return output

    def load_and_evaluate_model(
        self, model_path: str, input: torch.Tensor
    ) -> torch.Tensor:
        """Load a model from a given path and evaluate it on the input tensor.

        Args:
                model_path (str): The path to the model checkpoint.
                input (torch.Tensor): The input tensor for evaluation.

        Returns:
                torch.Tensor: The evaluation result after applying the model.

        """
        net = NNUnet.load_from_checkpoint(
            model_path, sw_batch_size=self.sw_batch_size, strict=False
        )
        net.eval()
        net.cuda()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            if self.tta:
                pred = self.tta_inference(net, input)
            else:
                pred = net.forward_logits(input)
        return torch.sigmoid(pred)

    def predict_ensemble(self, input: torch.Tensor) -> torch.Tensor:
        """Predicts the class label for the given input using an ensemble of models.
        Args:
                input (torch.Tensor): The input tensor to be predicted.
        Returns:
                torch.Tensor: The predicted class labels for the input.

        """
        predictions = [
            self.load_and_evaluate_model(model, input) for model in self.ckpts
        ]
        averaged_predictions = torch.mean(torch.stack(predictions), dim=0)
        return torch.ge(averaged_predictions, 0.5)

    def tta_inference(self, model: NNUnet, input: torch.Tensor) -> torch.Tensor:
        """Perform test-time augmentation (TTA) inference on the given model and input tensor.
        Args:
                model (NNUnet): The model to perform inference on.
                input (torch.Tensor): The input tensor to be augmented and evaluated.
        Returns:
                        torch.Tensor: The predicted output tensor after applying TTA.
        Description:
        This function performs test-time augmentation (TTA) inference on the given model and input tensor.
        It applies a series of transformations to the input tensor and evaluates the model on the augmented tensor.
        The transformations include flipping the input tensor along different axes.
        The augmented predictions are then averaged to obtain the final predicted output tensor. Dynamic TTA allows for


        """
        start_prediction = time.time()
        prediction = model.forward_logits(input)
        prediction_time = time.time() - start_prediction

        if self.dynamic_tta:
            max_time = self.max_tta_time - prediction_time
            random_flips = int(max_time // prediction_time)
            random_flips = int(np.clip(random_flips, 0, len(self.tta_flips)))
            print(
                f"One inference pass takes {prediction_time} s. The time limit is {self.max_tta_time} s. "
                f"Using {random_flips} additional test-time augmentations."
            )
            if random_flips == 0:
                return prediction
            flips_to_apply = random.sample(self.tta_flips, random_flips)
        else:
            if self.random_flips == 0:
                flips_to_apply = self.tta_flips
            else:
                flips_to_apply = random.sample(self.tta_flips, self.random_flips)
        # from tqdm import tqdm
        for flip_idx in flips_to_apply:  # tqdm(flips_to_apply, desc="TTA"):
            start_tta = time.time()
            prediction += self.flip(
                model.forward_logits(self.flip(input, flip_idx)), flip_idx
            )
            print("TTA time: ", time.time() - start_tta, " TTA flip: ", flip_idx)
        prediction /= len(flips_to_apply) + 1
        return prediction

    @staticmethod
    def flip(data, axis):
        return torch.flip(data, dims=axis)

    def run(
        self,
        ct_file_path: str,
        pet_file_path: str,
        label: str = None,
        save_path: str = None,
        verbose: bool = False,
    ) -> Union[int, dict]:
        """Runs the prediction pipeline on the given CT and PET image files.
        Args:
                ct_file_path (str): The path to the CT image file.
                pet_file_path (str): The path to the PET image file.
                label (str, optional): The path to the ground truth label image file. Defaults to None.
                save_path (str, optional): The path to save the output image. Defaults to None.
                verbose (bool, optional): Whether to print the timings. Defaults to False.
        Returns:
                int: 0 if the pipeline is successfully run or dict containing the metrics if evaluation is enabled.
        This function performs the following steps:
        1. Preprocessing: Loads the PET image, applies preprocessing to the CT and PET images, and stores the
        preprocessed data.
        2. Prediction: Predicts the output using an ensemble of models and TTA if TTA is enabled.
        3. Save Output: Resamples the prediction to match the reference image and saves it as a NIfTI file.
        4. Evaluation (optional): If a label image is provided, loads the ground truth label image, performs orientation
        and spacing adjustment, plots the results, and computes metrics.
        5. Print Timings: Prints the timings for preprocessing, prediction, saving, and total time.
        Note: The function assumes that the necessary models and preprocessing steps have been set up before calling
        this function.

        """
        start_time = time.time()

        # Preprocessing
        reference = mt.LoadImage()(pet_file_path)
        data = self.preprocess({"ct": ct_file_path, "pet": pet_file_path})
        preprocessing_time = time.time() - start_time

        # Prediction
        start_prediction = time.time()
        prediction = self.predict_ensemble(data.cuda())
        prediction_time = time.time() - start_prediction

        # Resample and save output
        start_saving = time.time()

        output = mt.ResampleToMatch()(
            prediction[0], reference[None, ...], mode="nearest"
        )
        # clip segmentation values where the suv is smaller than 3
        mask_sum_pre = output.sum()
        output[reference[None, ...] < 1] = 0
        print(
            "Clip segmentation values where the suv is smaller than 1. Difference: ",
            (mask_sum_pre - output.sum()).detach().cpu().numpy(),
        )
        if save_path is not None:
            mt.SaveImage(
                output_dir=save_path,
                output_ext=".nii.gz",
                # output_postfix="pred",
                separate_folder=False,
                output_dtype=np.uint8,
            )(output[0], filename=os.path.join(save_path, "PRED"))
        saving_time = time.time() - start_saving

        if verbose:
            total_time = time.time() - start_time
            print(f"Data preprocessing time: {preprocessing_time} seconds")
            print(f"Prediction time: {prediction_time} seconds")
            print(f"Saving time: {saving_time} seconds")
            print(f"Total time: {total_time} seconds")

        # Evaluation
        if label is not None:
            start_eval = time.time()
            ground_truth = mt.LoadImage()(label)
            test_aggregator = AutoPETMetricAggregator()
            test_aggregator(output[0], ground_truth)
            metrics = test_aggregator.compute()

            # Multiply with volume spacing -> to voxel_vol in ml
            volume = np.prod(reference.meta["pixdim"][1:4]) / 1000
            metrics = {
                "dice_score": metrics["dice_score"],
                "fp_volume": metrics["false_positives"] * volume,
                "fn_volume": metrics["false_negatives"] * volume,
            }
            if verbose:
                plot_results(output[0], ground_truth)
                eval_time = time.time() - start_eval
                print(f"Eval time: {eval_time} seconds")
                print("Spacing:", reference.meta["pixdim"][1:4])
                print(metrics)
            return metrics

        total_time = time.time() - start_time
        if verbose:
            print(f"Total time: {total_time:.2f}s")
            print(f"Preprocessing time: {preprocessing_time:.2f}s")
            print(f"Prediction time: {prediction_time:.2f}s")
            print(f"Saving time: {saving_time:.2f}s")
        return 0


def infer(
    ct_file_path: str,
    pet_file_path: str,
    label: Optional[str] = None,
    save_path: Optional[str] = None,
    verbose: bool = False,
    model_paths: List[str] = [""],
    sw_batch_size: int = 6,
    tta: bool = False,
    random_flips: int = 0,
):
    predict = PredictModel(
        model_paths,
        sw_batch_size,
        tta,
        random_flips,
        dynamic_tta=True,
        max_tta_time=240,
    )
    _ = predict.run(ct_file_path, pet_file_path, label, save_path, verbose=verbose)


if __name__ == "__main__":
    start = time.time()
    ct_path = "../test/orig/psma_95b833d46f153cd2_2018-04-16_0000.nii.gz"
    suv_path = "../test/orig/psma_95b833d46f153cd2_2018-04-16_0001.nii.gz"
    label = "../test/orig/psma_95b833d46f153cd2_2018-04-16.nii.gz"

    print("Predicting: ", label)
    infer(
        ct_file_path=ct_path,
        pet_file_path=suv_path,
        label=label,
        model_paths=["weights/last.ckpt"],
        save_path="../test/expected_output_dc/",
        verbose=True,
        tta=False,
    )
    print("Total time main(): ", time.time() - start)
