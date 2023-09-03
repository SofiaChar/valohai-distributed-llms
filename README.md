#  Distributed Fine-Tuning of Language Models with Valohai 

This GitHub repository demonstrates how to perform fine-tuning of large language models (LLM) in a distributed manner using [Valohai][vh]. The repository provides detailed guides and code examples to help you get started with distributed training, showcasing the use of Valohai for efficient model fine-tuning.

 Learn how to distribute the fine-tuning process of large language models across multiple machines or GPUs for improved training efficiency.


[vh]: https://valohai.com/
[app]: https://app.valohai.com
[hfe]: https://github.com/huggingface/transformers/blob/bc2571e61c985ec82819cf01ad038342771c94d0/examples/pytorch/summarization/run_summarization_no_trainer.py

## Distributed Training

In this section, we explore the various approaches to distributed training of large language models (LLMs) within the Valohai GitHub repository. Distributed training is crucial when dealing with computationally intensive tasks, 
and this repository provides multiple methods to achieve it. 

**___Note: To utilize the distributed training features outlined below, you will need a machine equipped with at least 2 or more GPUs. For setup assistance, please contact our support team.___**

### 1. Torchrun (Elastic Launch)

#### Script: train-torchrun.py

In the second approach, we leverage the `torchrun` (Elastic Launch) functionality, which extends the capabilities of `torch.distributed.launch`. We employ the Hugging Face Transformers Trainer to fine-tune the language model using our dataset. With `torchrun`, you can distribute the training process without making any modifications to your existing code. This method provides a straightforward introduction to distributed training for LLMs.

##### Benefits:

* Utilizes Transformers Trainer for model fine-tuning.
* No code changes needed for distributed training.
* Easy setup for those new to distributed training.


### 2. Accelerate Library

#### Script: train-accelerator.py

We employ the Accelerate library to facilitate distributed training. The `train-accelerator.py` script is based on the [Hugging Face example][hfe] for summarization without using the Trainer class. It grants you complete control over the training loop, allowing for more flexibility in customizing training processes. Meanwhile, the Accelerate library takes care of the distribution aspects, making it an efficient choice for distributed training.

##### Benefits:

* Fine-grained control over the training loop.
* Accelerate library handles distribution seamlessly.
* Ideal for custom training approaches.


### 3. Distributed Training Across Multiple Machines (Under Development)

The third approach, currently under development within this repository, tackles distributed training across multiple machines simultaneously. To achieve this, we employ Valohai's `valohai.distributed`, along with `torch.distributed` and `torch.multiprocessing` to establish communication between multiple machines during the training process. While still in development, this approach aims to provide a robust solution for training large language models across a distributed infrastructure.

##### Benefits:

* Distributes training across multiple machines.
* Uses Valohai's distributed capabilities.
* Enables efficient scaling for demanding training workloads.

## Configure the repository:
To get started login to the [Valohai app][app] and create a new project.

<details open>
<summary>Using UI</summary>

Configure this repository as the project's repository, by following these steps:

1. Go to your project's page.
2. Navigate to the Settings tab.
3. Under the Repository section, locate the URL field.
4. Enter the URL of this repository.
5. Click on the Save button to save the changes.
</details>

<details open>
<summary>Using terminal</summary>

To run the code on Valohai using the terminal, follow these steps:

1. Install Valohai on your machine by running the following command:

```bash
pip install valohai-cli valohai-utils
```

2. Log in to Valohai from the terminal using the command:

```bash
vh login
```

3. Create a project for your Valohai workflow.
   Start by creating a directory for your project:

```bash
mkdir valohai-distributed-llms
cd valohai-distributed-llms
```

Then, create the Valohai project:

```bash
vh project create
```

4. Clone the repository to your local machine:

```bash
git clone https://github.com/valohai/distributed-llms-example.git .
```

</details>

### **Running Executions:**
<details open>
<summary>Using UI</summary>

1. Go to the Executions tab in your project.
2. Create a new execution by selecting the predefined step.
3. Customize the execution parameters if needed.
4. Start the execution to run the selected step.

</details>

<details open>
<summary>Using terminal</summary>
To run individual steps, execute the following command:

```bash
vh execution run <step-name> --adhoc
```

For example, to run the _train-torchrun_ step, use the command:

```bash
vh execution run train-torchrun --adhoc
```
</details>

## <div align="center">Contact</div>
For bug reports and feature requests please visit GitHub Issues.

If you need any help, feel free to contact our support team!