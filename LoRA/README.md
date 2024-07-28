# 🏷️Implementation of LoRA [Low Rank Adaptation]

🎤 LoRA is implementated on 2021 by Microsoft. It is currently widely used in language models like LLM.

## 📝 Knowledge

🛞Problems with normal FineTuning : 
  * For Large model we usually finetune the pretrained model for our custom dataset. But finetuning the whole model is computationally expensive for most of the users/researchers.
  * For finetuning we need to store checkpoint of every epoch, this also take lot of space.
  * If we want to use our model for 2 different task, we need to finetune the model seperately on 2 different dataset and save 2 different models. and in production pipeline we need to load each model at a time for each usecase. This is also time consuming.

🧚🏻 Solution : 
  * Using LoRA.
  *  Instead of finetuning on the full model weights we can apply LoRA matrix on the base model and train the LoRA matrix.

📝 Paper  : [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

🎥 For better understanding I took help from : [LoRA: Low-Rank Adaptation of Large Language Models - Explained visually + PyTorch code from scratch](https://youtu.be/PXWYUTMt-AU?si=c6eLTWErwkf67R5T)


## 👨🏻‍🎨 Implementation :

🐳 Suppose we are having a 2 layer model of large number of parametes and we have to add LoRA matrix on this model.
![code2image](https://github.com/user-attachments/assets/21ebc99f-7273-40c3-b372-003964971e47)

🐳 All the Trainable parameters of the model :
![code2image-2](https://github.com/user-attachments/assets/543eba34-987e-4679-9ecb-3c1f1cb5f2ae)

🐳 For a pre-trained weight matrix W0 ∈ Rd×k, we constrain its update by representing the latter with a low-rank de-composition W0 + ∆W = W0 + BA, where B ∈ R^(d×r) ,A ∈ R^(r×k), and the rank r ≪ min(d,k). 

During training, W0 is frozen and does not receive gradient updates, while A and B contain trainable parameters. Note both W0 and ∆W = BA are multiplied with the same input, and their respective output vectors are summed coordinate-wise.

The LoRA Implementation : 
![code2image-3](https://github.com/user-attachments/assets/184cbd36-517f-4a79-bcbd-10e057d3bb6a)

🐳 For adding the LoRA matrix to our original model we used pytorch's `register_parametrization`.
![code2image-4](https://github.com/user-attachments/assets/c7615253-1d25-4ffe-a015-47d7824bd7bd)

🐳 After applying LoRA to the model, we can see the LoRA Parameters is only `0.24%` of the original weights. So if we are training the model with LoRA, we need to train only 0.24% of the model weights.
![code2image-5](https://github.com/user-attachments/assets/f6f1539e-5e9a-41bc-a696-e3fddd9d1a50)


### 🥷🏻 Regards :
[@yota](https://github.com/yotaAI)
