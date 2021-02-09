## Write-up to explain the work flow that I took to complete the programming task for the Internship role at VOLKSWAGEN GROUP RESEARCH

Note: Due to computing power limitations, I have performed all my works in the Google colab, hence I have done all the code works and implementation in the provided notebook(one copy for Experiment 1 - with STN layer and one copy for Experiment 2 - without STN layer)

But, I have also included the corresponding *.py scripts for your reference, if you need it.

#### Experiment 1:
- As per the instructions provided in the Github README.md for the traffic_sign_recogntion repository (https://github.com/wolfapple/traffic-sign-recognition.git), I have run the pipeline and the results observed are captured in the Excel sheet.

    It is also captured below as an image.
    (Note: The epochs is set to 22 due to computing power limitations for both the experiments)

![Train_and_val_loss](https://user-images.githubusercontent.com/22639337/107316988-13862b80-6a9a-11eb-9267-ce5906806f09.png)

![Val_accuracy](https://user-images.githubusercontent.com/22639337/107317016-20a31a80-6a9a-11eb-8bdf-468083b9db87.png)

- Then, after the evaluation script is run, the following results are observed.

    Test loss: 0.025329;	Test accuracy: 99.287%
    
- The role of STN layer is visualized by visualizing the output of STN layer and understanding the spatial transformations that took place

![Visualization_stn](https://user-images.githubusercontent.com/22639337/107286707-3cd69580-6a61-11eb-98e8-9a669bfd9b47.png)


- I have used some additional performance metrics to evaluate the model and they are the following:

    1) confusion matrix, 
    2) recall, 
    3) precision, and 
    4) F1 score.

- Open the Notebook in the Experiment_1 directory and run the pipeline to print the confusion matrix, recall, precision and F1 Score of the model. (Make sure to upload all the necessary files to the workspace before executing the lines of code)
- Alternatively, you can use the corresponding python script 'performance_evaluation.py' that I included in the files

- Note: When we run the performance_evaluation.py, please set the argument '--model' to './Experiment1/{model_file_name}' or './Experiment2/{model_file_name}' depending upon which model whether you want the one with or without STN.

- Additionally, to visualize the confusion matrix, I have used the heat map to plot it. With this, we will be able to understand where the model gets confused 
between the classes.
    
    Confusion Matrix:
![Confusion_matrix](https://user-images.githubusercontent.com/22639337/107317117-4fb98c00-6a9a-11eb-80e4-ce7adc9d64ef.png)

- Note: For generating the heat map, I have defined the function in the ./resources/plotcm.py

- Performance evaluation results:   
ACCURACY : 0.992874 = 99.29%  
RECALL :     0.874685 = 87.47 %  
PRECISION : 0.821341 = 82.13 %  
F1_SCORE : 0.847174 = 84.72 %  

#### Experiment 2:
- Experiment 2 is more or less the same as Experiment, with the exception of removing STN (Spatial Transformer Network) from the model.

- The corresponding training and evaluation results are plotted as below:

![Train_and_val_loss](https://user-images.githubusercontent.com/22639337/107378032-dea2c480-6aeb-11eb-9941-7a8eefa8d4fc.png)

![Val_accuracy](https://user-images.githubusercontent.com/22639337/107378062-e5313c00-6aeb-11eb-8c92-08ea38b4c7e0.png)

- The evaluation results are:

    Test loss: 0.041385;	Test accuracy: 98.678%
    
    (The reduced accuracy here clearly justifies the role of STN layer for a better accuracy of the model)
    
- Confusion Matrix:
![Confusion_matrix_without_STN](https://user-images.githubusercontent.com/22639337/107287649-870c4680-6a62-11eb-842a-ace045429d06.png)
    
- Performance evaluation results:  
ACCURACY : 0.986778 = 98.68 %
RECALL : 0.986499 = 98.65 %  
PRECISION : 0.977475 = 97.75 %  
F1_SCORE : 0.981967 = 98.20 %
    
#### Experiment 3:
a) Activation output of conv3 layer for the trained models
- Just in the Notebooks for both the experiments, after the performance evaluation code works, I have included the code work for feature visualization.
- Additionally, I have included a python script 'feature_visualisation.py'.
- This allows to peek at the results of conv3 layer, otherwise called feature maps of conv layer.
- As per the instructions, 10 images of at least 5 different classes are taken for this particular task

- Feature Visualisation at Conv3 layer for image0 - class 16:

    With STN:
![image_0](https://user-images.githubusercontent.com/22639337/107318165-6e208700-6a9c-11eb-8574-e6d001a93e59.png)

    Without STN:
![image_0](https://user-images.githubusercontent.com/22639337/107287974-0437bb80-6a63-11eb-96be-670496a1d935.png)

- The feature map clearly captures only the central element of the traffic sign, which increases the spatial invariance for the end result 
(It is mainly due to the spatial transformations that took place at the start of the network).

- But, the influence of STN layer in the spatial invariable is not perceivable very much in this conv layer.

b) Gradient calculation of Conv3 layer for the trained models
- To my understanding, only during the training, the model would be possessing the gradient tracking with it. Hence, during the prediction stage, the gradient value would be set to zero, as there is no gradient descent.
- Please correct me if I am wrong. I am humbly open to learn, if I am mistaken. I will correct, update this section upon learning it. 

- The observations are summarized also in .ppt and .pdf document 'Programming_task_Summarized_report'

#### Thank you for your patience and your consideration!!!
