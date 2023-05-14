# AutoGluon Object Detection w/ SageMaker Ground Truth Example
This example shows you how to perform AutoGluon-based Object Detection with a manifest file labeled with SageMaker Ground Truth. Note that you do not need SageMaker Ground Truth if you just have a json annotation file in COCO format. 

# Prerequisites
- AutoGluon 0.7 
    - Please make sure to install the nightly build 0.7 version(autogluon==0.7.1b20230513), not the 0.7 full version. Object Detection is not possible in the 0.7 full version!
    - Reference: https://github.com/autogluon/autogluon/issues/3082
- SageMaker Ground Truth 
- Jupyter Notebook (GPU recommended)
    - If you're working in SageMaker, we recommend `ml.g4dn.xlarge` or `ml.g5.xlarge` notebook instance

# Dataset
- [Pika](https://universe.roboflow.com/maxmacstnpublic/pika-wehps) by Roboflow

## License Summary
This sample code is provided under the MIT-0 license. See the LICENSE file.