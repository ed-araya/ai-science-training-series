# AI Accelerators


Scientific applications are increasingly adopting Artificial Intelligence (AI) techniques to advance science. There are specialized hardware accelerators designed and built to run AI applications efficiently. With a wide diversity in the hardware architectures and software stacks of these systems, it is challenging to understand the differences between these accelerators, their capabilities, programming approaches, and how they perform, particularly for scientific applications. 

We will cover an overview of the AI accelerators landscape with a focus on Cerebras-CS3 for training, and Sambanova SN40-L (Metis) for inference jobs, along with architectural features and details of their software stacks. We will have hands-on exercises that will help attendees understand how to program these systems by learning how to refactor codes written in standard AI framework implementations and compile, run and evaluate the models on these systems. 



## Slides

* [Intro to AI Series: AI Accelerators](./ALCF_HandsOn_AIScience-training-series-Nov2025.pdf) 
 

## Hands-On Sessions


* [Cerebras](./Cerebras/README.md)
* [SambaNova](./SambaNova/README.md)                                    

## Homework 
* Cerebras Homework : Run the Llama-7B example for different batch sizes and compare the performance.

### Results and challenges:
Running the code was initially unsuccessful, the following error keep appearing when running: cszoo fit configs/params_llama2_7b.yaml --job_labels name=llama2_7b --model_dir model_dir_llama2_7b |& tee mytest.log:

File "/home/earaya/R_2.6.0/venv_cerebras_pt/lib64/python3.11/site-packages/peft/tuners/lora/model.py", line 26, in <module>
    from transformers.modeling_layers import GradientCheckpointingLayer
ModuleNotFoundError: No module named 'transformers.modeling_layers'

The program worked fine the day of the lecture. The R_2.6.0 directory was removed and the instructions were followed again, but the error continued occurring. The error was fixed by:

pip uninstall peft
pip install peft==0.17.1

After the compatibility issue was solved, the code seemed to work but it got stocked at “INFO:   Beginning appliance run”.


* Sambanova Homework : Use your choice of huggingface dataset and compare the performance on GptOSS model using both Metis and Sophia, reason out the possible differences. 

### Results and challenges: 
Accessing SambaNova through the WebUI worked out fine, I was able to have Sophia:openai/gpt-oss-120b and Metis:gpt-oss-120b-131072 working simultaneously to compare results. However, based on the information provided in the lecture, it was unclear how to select a hugginface dataset and compare performance on GptOSS. Thus, that part of the homework was not accomplished, nevertheless, the SambaNova WebUI was used with Gpt-OSS in Sophia and Metis to ask how to do so: “Could you provide guidance to accomplish the following: “Use your choice of huggingface dataset and compare the performance on GptOSS model”. Metis provided the instructions significantly faster than Sophia (>2sec vs <1sec), but it was unclear how to implement the instructions in the WebUI for the homework exercise, I assume the it had to be done through the API, but there was not enough guidance in https://github.com/ed-araya/ai-science-training-series/blob/main/05-AITestbed/SambaNova/README.md to complete the task. 



## Documentation 
* [ALCF Documentation](https://docs.alcf.anl.gov/ai-testbed/)
* [Cerebras Public Documentation on Training](https://training-docs.cerebras.ai/rel-2.5.0/getting-started/overview)
* [Sambanova Public Documentation on Inference](https://docs.sambanova.ai/docs/en/get-started/overview)

## Director’s Discretionary Allocation Program

To gain access to AI Testbeds at ALCF after current allocation expires apply for [Director’s Discretionary Allocation Program](https://www.alcf.anl.gov/science/directors-discretionary-allocation-program)

The ALCF Director’s Discretionary program provides “start up” awards to researchers working to achieve computational readiness for for a major allocation award.
