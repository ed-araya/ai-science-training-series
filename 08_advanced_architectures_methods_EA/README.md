# Advanced AI Architectures and Methods: See homework at the end

AI-Driven Science on Supercomputers @ ALCF 2022

**Contact:** Venkat Vishwanath ([venkat@anl.gov](mailto:///venkat@anl.gov)), Bethany Lusch ([blusch@anl.gov](mailto:///blusch@anl.gov)), Carlo Graziani ([cgraziani@anl.gov](mailto:///cgraziani@anl.gov)) 


[AI Accelerators for Science](https://github.com/argonne-lcf/ai-science-training-series/blob/main/08_advanced_architectures_methods/ALCF_AI_Testbed_Vishwanath.pdf)
    
[Gaussian Process Modeling](Gaussian_Process_Modeling.ipynb)

[Advanced AI Methods](https://github.com/argonne-lcf/ai-science-training-series/blob/main/08_advanced_architectures_methods/AITrainingSeries-AdvancedMethods.pdf)

**Useful Links**

 [Getting started on AI Testbed](https://www.alcf.anl.gov/support/ai-testbed-userdocs/index.html)
 
 [Useful AI Testbed Resources](https://github.com/argonne-lcf/AIaccelerators-SC22-tutorial)
 
 **Homework**
 
 Submit a paragraph about: 
 
- How could you use AI for a problem that interests you? 
- What is the task? 
- What kind of data would you use? 
- What kind of method or model might be appropriate? 
- What kind of metric would you use to measure success? 

Feel free to consult the Internet for ideas.

This paragraph can be placed in a README in git and you can submit the link. 

*E. D. Araya Homework:*

My area of research (astrophysics, specifically observational radio astronomy) has many topics in which AI is useful, as new observatories generate very large data sets. There are many tasks to which AI could be applied in radio astronomy, including direct applications of image classification such as to automatically identify populations of radio sources in large surveys. Such application would be based not only on morphology (like in the standard image classification discussed in the training) but also requires spectral information (a generalization of using RGB information). A specific dataset that could be used is the VLASS (VLA Sky Survey). In principle, the ResNet convolutional neural network codes used in the training could be adapted to classify VLASS images. Some preparation work would be needed (generate cut-outs of the fields), moreover, some type of unsupervised learning may be needed if we would like to avoid creating a very large labeled dataset for training and validation. To measure success, the model would be applied to a (labeled) validation dataset (labeled by the PI as well as by trained graduate and undergraduate students). Success would be judged by comparing the accuracy of the image classification of the validation dataset achieved by the IA with respect to the accuracy of classification achieved by manual classification (citizen science project and/or undergraduate students with minimal training) and by comparing the achieved accuracy with respect to image classification results in other areas (e.g., IA used in classification of galaxies in optical surveys). 
