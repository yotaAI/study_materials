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

