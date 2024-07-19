# Datacentric baseline algorithm for autoPETIII challenge

Source code for the datacentric baseline algorithm container for autoPETIII challenge. Information about the 
submission can be found [here](https://autopet-iii.grand-challenge.org/submission/) and in the [grand challenge 
documentation](https://grand-challenge.org/documentation/).

## Task
Best data-handling wins! In real-world applications, especially in the medical domain, data is messy. Improving 
models is not the only way to get better performance. You can also improve the dataset itself rather than treating 
it as fixed. This is the core idea of a popular research direction called Data-Centric AI (DCAI). Examples are 
outlier detection and removal (handling abnormal examples in dataset), error detection and correction (handling 
incorrect values/labels in dataset), data augmentation (adding examples to data to encode prior knowledge)  and many 
more. If you are interested: a good resource to start is [DCAI](https://dcai.csail.mit.edu/).

The rules are: Train a model which generalizes well on FDG and PSMA data but DO NOT alter the model architecture or 
get lost in configuration ablations. For that we will provide a second baseline container and a 
[tutorial](https://github.com/ClinicalDataScience/datacentric-challenge/tree/main) how to use and train the model. You are 
not allowed to use any additional data and the datacentric baseline model will be in competition. This means to be 
eligible for any award you need to reach a higher rank than the baseline. Models submitted in the datacentric 
category will also score in award category 1. Clarification on the rules can be found 
[here](https://autopet-iii.grand-challenge.org/rules/).

## Usage 

In order to use the baseline you first need to download the baseline weights via `bash download_model_weights.sh`. 
After that you can build the container by running `bash build.sh`. In order to participate in the second award 
category your model must be identical to the datacentric-challenge fixed code. We will verify this. The simplest way 
to start is by replacing the baseline weights in the weights folder with your own. If you use different pre- or 
post-processing, you will need to modify the code in `predict.py`. After that, update the `process.py` file to call 
your desired predict function. Finally, adapt the `requirements.txt` file to your needs.
## Testing

Use a python 3.10 based environment and install the requirements.txt file via `pip install -r requirements.txt`. 
Make sure model weights exist in `/weights`. Download the baseline weights by running `bash download_model_weights.sh`. 
Then run `predict.py` to create an expected_output mask. After that you can run `bash test.sh`. The test will 
only pass if your model is deterministic. For this repo you will need to disable test-time augmentation in the 
`process.py` file by setting `self.tta = False` in the init function. 

