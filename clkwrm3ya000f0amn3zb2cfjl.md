---
title: "Unifying Deep Learning Models: Deploying TensorFlow and PyTorch Together with ONNX and ONNX Runtime"
datePublished: Fri Aug 04 2023 15:53:11 GMT+0000 (Coordinated Universal Time)
cuid: clkwrm3ya000f0amn3zb2cfjl
slug: unifying-deep-learning-models-deploying-tensorflow-and-pytorch-together-with-onnx-and-onnx-runtime
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1691164551322/5beaaeda-befb-44e8-b809-1405cc7818d8.png
tags: machine-learning, flask, deep-learning, model-deployment, onnx

---

# Index

1. Outline
    
2. Introduction
    
3. Understanding ONNX: Enabling Model Interoperability
    
4. The Need for ONNX in Model Deployment
    
5. Converting TensorFlow Model to ONNX
    
6. Converting PyTorch Model to ONNX
    
7. Preparing the Deployment Environment
    
8. Deploying TensorFlow and PyTorch Models with ONNX Runtime & Flask
    
9. Conclusion
    

# 1\. Outline

1. Introduction
    
    * Deep learning models and their challenges with interoperability across frameworks.
        
    * Motivation for deploying both TensorFlow and PyTorch models in a single environment.
        
    * Solution overview: Utilizing ONNX as an intermediate format for conversion and ONNX Runtime for inference.
        
2. Understanding ONNX: Enabling Model Interoperability
    
    * Explanation of ONNX (Open Neural Network Exchange) and its purpose in the deep learning ecosystem.
        
    * How ONNX acts as a bridge between different deep learning frameworks.
        
    * Key features of ONNX, such as model representation, interoperability, and extensibility.
        
3. The Need for ONNX in Model Deployment
    
    * Challenges of deploying models across different deep learning frameworks.
        
    * Why pure model files (like .h5 for TensorFlow and .pth for PyTorch) are not always sufficient for seamless deployment?
        
    * Discuss the benefits of using ONNX for model conversion and deployment:
        
        **a. Platform Independence:** ONNX enables models to run on different platforms without modifications.
        
        **b. Framework Agnosticism:** ONNX allows models to be converted between various frameworks without loss of functionality.
        
        **c. Simplified Model Exchange:** ONNX facilitates collaboration between researchers and developers by offering a common model format.
        
        **d. Production Efficiency:** ONNX reduces the time and effort required to deploy models across different environments.
        
4. Converting TensorFlow Model to ONNX
    
    * Discuss the limitations of using keras2onnx for non-pure Keras models.
        
    * Demonstrate how to convert the .h5 model to a .pb format using TensorFlow.
        
    * Use the tf2onnx package to convert the .pb model to ONNX format with opset=16.
        
5. Converting PyTorch Model to ONNX
    
    * Clarify the ONNX compatibility range for opset values in TensorFlow and PyTorch models.
        
    * Use the torch.onnx exporter to convert the PyTorch model to ONNX format with opset=16.
        
6. Preparing the Deployment Environment
    
    * Highlight the significance of choosing a common opset value (16) for ONNX conversion.
        
    * Explain the rationale behind selecting ONNX Runtime 1.15.1 as the deployment environment for supporting the chosen opset.
        
7. Deploying TensorFlow and PyTorch Models with ONNX Runtime & Flask
    
    * Provide step-by-step instructions for setting up a Flask-based API to handle requests for both TensorFlow and PyTorch models.
        
    * Showcase the structure of the API for model inference, including necessary input data processing.
        
    * Demonstrate how to load the ONNX models using ONNX Runtime in the Flask API.
        
    * Showcase the code for running inferences with ONNX Runtime for both models.
        
    * Address any challenges faced during the deployment process and their solutions.
        
8. Conclusion
    
    * Summarize the benefits of using ONNX and ONNX Runtime for deploying diverse deep learning models in a unified environment.
        
    * Emphasize the simplicity and efficiency of deploying multiple models without the need for separate environments.
        
    * Discuss potential applications and future work in optimizing the deployment setup
        

# 2\. Introduction

In the rapidly evolving field of deep learning, researchers and developers often find themselves grappling with interoperability challenges when deploying models across different frameworks. TensorFlow and PyTorch, two of the most popular deep learning libraries, have their unique strengths and are preferred choices for various tasks. However, their incompatibility can present significant hurdles when trying to combine and utilize models from both frameworks in a single environment.

The objective of this blog is to address this predicament by demonstrating a seamless approach to deploying both TensorFlow and PyTorch models together in a unified environment. We will leverage the power of ONNX (Open Neural Network Exchange), an open standard for representing deep learning models, to bridge the gap between these frameworks. Additionally, we will use ONNX Runtime, an efficient runtime engine, for performing inference on the deployed models.

The need for such a solution arises due to the differences in model formats and underlying execution mechanisms of TensorFlow and PyTorch. While TensorFlow models are typically saved in .h5 format, PyTorch models use .pth format, making direct deployment in the same environment challenging. Furthermore, TensorFlow and PyTorch models often require different versions of their respective runtimes, which can lead to complex and resource-intensive deployment setups.

ONNX - a format that transcends framework boundaries and enables the conversion of deep learning models between TensorFlow, PyTorch, and other popular libraries. By converting models to the ONNX format, we can create an interoperable representation that seamlessly integrates models from different frameworks, avoiding the need for separate deployment environments.

In this blog, we will walk you through the process of converting a TensorFlow model saved in .h5 format to ONNX and a PyTorch model to ONNX. We will specifically choose opset=16, the latest common version supported by both TensorFlow and PyTorch, ensuring compatibility for deployment. With models successfully converted to ONNX, we will showcase how to implement a unified deployment environment using Flask, a lightweight web application framework.

By deploying both models with ONNX Runtime, we achieve the much-desired compatibility and efficiency, allowing us to handle diverse deep-learning models with ease. The use of a single environment not only simplifies deployment but also streamlines maintenance and reduces operational overhead.

Now, let's dive into the exciting world of ONNX, model conversion, and deployment, as we explore the unification of TensorFlow and PyTorch in a seamless and scalable environment. Let's get started!

# 3\. Understanding ONNX: Enabling Model Interoperability

#### **What is ONNX?**

ONNX (Open Neural Network Exchange) is an open standard designed to foster model interoperability in the deep learning ecosystem. It serves as a bridge between different deep learning frameworks, allowing seamless model exchange and collaboration.

#### **The Purpose of ONNX**

The primary objective of ONNX is to provide a format that enables easy conversion of deep learning models between different frameworks. By doing so, ONNX allows researchers and developers to leverage the strengths of multiple libraries without the constraints of framework-specific model formats.

#### **How ONNX Works**

ONNX models consist of a graph structure, where nodes represent operations and tensors store data. This design enables ONNX to capture the essence of a model and its operations, facilitating smooth conversion between frameworks.

#### **Key Features of ONNX**

* **Platform Independence:** ONNX models can run on diverse platforms without modifications, making them highly portable and deployable across a wide range of devices.
    
* **Framework Agnosticism:** ONNX enables model conversion between various deep learning libraries, such as TensorFlow, PyTorch, and more, without loss of functionality.
    
* **Extensibility:** ONNX is extensible, allowing developers to add custom operators and layers to the standard.
    

#### **ONNX and Deep Learning Frameworks**

ONNX is supported by numerous popular deep-learning frameworks, including TensorFlow, PyTorch, Caffe2, and MXNet, among others. This widespread adoption ensures broad compatibility and fosters collaboration across the community.

#### **ONNX Versions and Opsets**

ONNX evolves with each new release, and its versioning ensures compatibility. Opsets play a crucial role in ONNX conversion, as they specify the version used during the model conversion.

# 4\. The Need for ONNX in Model Deployment

In the dynamic landscape of deep learning, the proliferation of various deep learning frameworks has fueled innovation and progress in artificial intelligence. TensorFlow and PyTorch, two of the most prominent frameworks, have emerged as preferred choices for researchers and developers due to their flexibility, extensive tooling, and strong community support. However, the differences in model formats and execution mechanisms between these frameworks can pose challenges when attempting to combine and deploy models from both ecosystems.

TensorFlow models are typically saved in the .h5 format, while PyTorch models use .pth format. Although these formats are well-suited for their respective frameworks, direct deployment in the same environment becomes cumbersome. This incompatibility creates a siloed approach, wherein models must be deployed separately in different environments, leading to complexities in managing multiple setups.

Furthermore, TensorFlow and PyTorch models may have dependencies on specific versions of their respective runtime environments, which might not be compatible with each other. This divergence necessitates the creation of separate deployment environments for each model, contributing to increased resource utilization and maintenance overhead.

To address these deployment challenges, the deep learning community embraced ONNX as an effective solution. ONNX serves as a standardized intermediate format that facilitates seamless model exchange between different frameworks. By employing ONNX as an interoperable representation, researchers and developers can break free from the constraints of framework-specific formats and seamlessly deploy models across diverse environments.

To ensure the smooth deployment of models from both TensorFlow and PyTorch, it is essential to select an appropriate ONNX Opset. Opsets correspond to specific ONNX versions, and choosing a common value that satisfies the requirements of both frameworks is crucial. In this blog, we opt for Opset 16, the latest common version supported by TensorFlow and PyTorch, ensuring compatibility and seamless conversion between the models.

By leveraging the capabilities of ONNX, we can transcend the limitations of individual deep learning frameworks and unlock the true potential of deploying TensorFlow and PyTorch models together in a unified environment. In the following sections, we will explore the process of converting models to the ONNX format and implementing a single deployment environment using ONNX and ONNX Runtime.

# 5\. Converting TensorFlow Model to ONNX

Having successfully trained our TensorFlow model, we now have the model saved in the .h5 format. To deploy this model alongside other deep learning models in a unified environment, we need to convert it to the ONNX format.

The trained TensorFlow model is saved in the .h5 format, which is the native Keras model format. While this format is suitable for TensorFlow-based deployments, it is not directly compatible with ONNX conversion, especially when the model includes custom or non-pure Keras layers.

While there are tools like keras2onnx available for converting Keras models to ONNX, we encountered limitations in our case. Our model includes custom or non-pure Keras layers, rendering direct conversion with keras2onnx impractical.

To proceed with ONNX conversion, we first convert the .h5 model to TensorFlow's native .pb format. This conversion step is shown below,

```python
import os
import tensorflow as tf
import sys

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

TF_MODEL_PATH = "./Models/Model1.h5"

model = CustomModel()
model.load_weights(filepath=TF_MODEL_PATH, by_name=True)

model.keras_model.save("./Models/Model1")
```

This method bypasses the limitations of keras2onnx and allows us to generate an ONNX representation that includes our custom or non-pure Keras layers.

By converting our TensorFlow model to the ONNX format, we create an interoperable representation that can seamlessly coexist with models from other frameworks. In the next section, we will explore the process of converting the .pb model to ONNX, enabling us to deploy our TensorFlow model alongside the PyTorch model in a unified environment.

To convert the TensorFlow model to the ONNX format, we will use the `tf2onnx` tool, a part of the `tensorflow-onnx` repository. The following steps outline the conversion process:

First, clone the `tensorflow-onnx` repository using the following command:

```python
git clone https://github.com/onnx/tensorflow-onnx
```

Change your current working directory to the `tensorflow-onnx` directory:

```python
cd tensorflow-onnx/
```

Install the `tensorflow-onnx` package by running the setup script:

```python
python setup.py install
```

Now, it's time to convert the TensorFlow model to the ONNX format using the `tf2onnx` converter. Here's the command for the conversion:

```python
python -m tf2onnx.convert --saved-model PATH/TO/MODEL/DIRECTORY/ --output PATH/<NAME>.onnx --opset 16
```

# 6\. Converting PyTorch Model to ONNX

To deploy our PyTorch model alongside the previously converted TensorFlow model, we need to convert it to the ONNX format. We'll use the PyTorch `torch.onnx.export()` function for this purpose. Below are the steps involved in the conversion:

##### We begin by loading the pre-trained PyTorch model. In our case, we have a model is loaded from the file `"Model.pth"`:

```python
import torch
from model import Model

TORCH_MODEL_PATH = "Model.pth"
ONNX_MODEL_PATH = "Model.onnx"

model = Model() # Create the model
model .load_state_dict(torch.load(TORCH_MODEL_PATH))  # Load the model weights
model .eval()   # Set the model to evaluation mode
```

Next, we define a dummy input to the model and specify the input and output names for the ONNX model:

```python
dummy_input = torch.randn(1, 1, 512, 512)   # Dummy input with appropriate dimensions
input_names = ["input image"]   # Name for the input
output_names = ["predicted mask"]   # Name for the output
```

With everything set up, we can now export the PyTorch model to the ONNX format using `torch.onnx.export()`:

```python
torch.onnx.export(model, dummy_input, ONNX_MODEL_PATH, 
                  input_names=input_names, output_names=output_names,
                  export_params=True, opset_version=16)
```

# 7\. Preparing the Deployment Environment

Before we proceed with deploying our converted TensorFlow and PyTorch models using ONNX and ONNX Runtime, let's set up the deployment environment. This involves installing the necessary dependencies and frameworks to ensure a smooth and efficient deployment process.

The first essential component of our deployment environment is the ONNX Runtime. ONNX Runtime is an open-source, high-performance inference engine for ONNX models. To install ONNX Runtime, use the following command:

```python
pip install onnxruntime
```

***Note:*** Check if the onnxruntime version supports the opset version

Depending on the specific requirements of your deployment setup, you might need to install additional dependencies. These may include libraries for image processing, data manipulation, or any other functionalities needed for your inference pipeline. Ensure that you have the required dependencies installed in your environment.

Now, let's create a deployment application that will use ONNX Runtime to load and run our converted models. Depending on your deployment scenario, you may create a web service, a command-line application, or integrate the models into an existing system.

In the deployment application, use ONNX Runtime to load the converted TensorFlow and PyTorch models:

```python
import onnxruntime

# Load the TensorFlow model
tf_model = onnxruntime.InferenceSession("./Models/Model1.onnx")

# Load the PyTorch model
torch_model = onnxruntime.InferenceSession("./Models/Model2.onnx")
```

Now that both models are loaded into the deployment application, you can use them for inference tasks. Depending on your application's inputs and outputs, you can feed input data to the models and obtain their predictions using ONNX Runtime.

# 8\. Deploying TensorFlow and PyTorch Models with ONNX Runtime & Flask

To deploy our converted TensorFlow and PyTorch models in a user-friendly and scalable manner, we'll create a RESTful API using Flask. Flask is a lightweight and flexible web framework in Python that allows us to expose our models as web services.

First, ensure you have Flask installed. If you don't have it installed already, use the following command:

```python
pip install flask
```

Let's create a new Python script, [`app.py`](http://app.py), to build our Flask application:

```python
from flask import Flask, request, jsonify
import onnxruntime
import numpy as np

app = Flask(__name__)

# Load the ONNX Models
tf_model = onnxruntime.InferenceSession("./Models/Model1.onnx")
torch_model = onnxruntime.InferenceSession("./Models/Model2.onnx")

# Define the API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.json

        # Preprocess the data (if needed)
        # ... (add preprocessing steps if required)

        # Convert the input data to a numpy array
        input_data = np.array(data['input'])

        # Perform inference using ONNX Runtime
        tf_output = tf_model.run(None, {'input image': input_data})
        torch_output = torch_model.run(None, {'input image': input_data})

        # Post-process the output (if needed)
        # ... (add post-processing steps if required)

        # Prepare the response
        response = {
            'tf_prediction': tf_output.tolist(),
            'torch_prediction': torch_output.tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})
```

Now that we have created the Flask application, we can run it using the following command:

```python
python app.py
```

With the Flask application running, you can make POST requests to the `/predict` endpoint to obtain predictions from both the TensorFlow and PyTorch models. The input data should be in the JSON format, and the response will contain the predictions from both models.

For example, you can use tools like `curl` or Python's `requests` library to make API requests and receive the predictions.

By building this deployment API with Flask, we have created a user-friendly interface to interact with the deployed models, making it easy to integrate them into other applications or systems.

# 9\. Conclusion

In this blog, we explored the power of ONNX (Open Neural Network Exchange) in enabling seamless interoperability between deep learning frameworks. By converting both TensorFlow and PyTorch models to the ONNX format, we achieved a unified deployment environment using ONNX Runtime and Flask, eliminating the need for multiple deployment setups.

The journey began with the realization that deploying models from different deep learning frameworks could be complex and resource-intensive. ONNX emerged as the perfect solution, acting as a common ground for model representation. We leveraged ONNX to bridge the gap between TensorFlow and PyTorch models, thereby streamlining the deployment process.

We first explored the conversion of a trained TensorFlow model to the ONNX format. Due to the model's non-pure Keras layers, we opted for the `tf2onnx` converter, which allowed us to bypass the limitations of `keras2onnx`. The TensorFlow model was successfully converted to the ONNX format using `opset_version=16`, ensuring compatibility with ONNX Runtime.

Next, we dived into converting a PyTorch model to ONNX using `torch.onnx.export()`. This step allowed us to create an interoperable representation of the PyTorch model, making it compatible with ONNX Runtime and the unified deployment setup.

To ensure a smooth deployment process, we prepared the deployment environment by installing ONNX Runtime and any additional dependencies required for our deployment setup. We also built a deployment API using Flask, enabling users to make POST requests and receive predictions from both models.

With the deployment API ready, we deployed both the TensorFlow and PyTorch models using ONNX Runtime and Flask. The integration of both models in a single environment highlighted the versatility of ONNX and its ability to harmonize deep learning frameworks seamlessly.

By embracing ONNX in our deployment strategy, we unlocked the potential to utilize models from various frameworks, promoting collaboration and exchange among the deep learning community. ONNX facilitated a unified environment for deploying models, significantly simplifying the deployment process and accelerating the journey from model development to production.

As the field of deep learning continues to evolve, ONNX paves the way for further advancements and cross-framework collaborations. It presents exciting opportunities for model sharing, fine-tuning, and exploring novel architectures, enabling AI practitioners to push the boundaries of what's possible.

In conclusion, ONNX serves as a critical enabler of model interoperability, making it an essential tool for modern AI deployments. By adopting ONNX in our deployment strategy, we embraced the spirit of collaboration and innovation, contributing to a more inclusive and thriving AI ecosystem.

# About the Author

Hiii, I'm @[Avinash](@avinash-218), a recent computer science graduate. I am currently working as a Machine Learning Intern at [**Augrade.**](https://www.augrade.com/)

**Connect me through :**

* [**LinkedIn**](https://www.linkedin.com/in/avinash-r-2113741b1/)
    
* [**GitHub**](https://github.com/avinash-218)
    
* [**Instagram**](https://www.instagram.com/_ravinash/)
    

Feel free to correct me !! :)  
Thank you folks for reading. Happy Learning !!! 😊