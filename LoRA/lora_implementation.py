import torch
import torch.nn as nn


_ = torch.manual_seed(0)


class BaseNeuralNet(nn.Module):
	def __init__(self,hidden_state_0=1000,hidden_state_1=2000):
		super().__init__()

		self.lin1 = nn.Linear(28*28,hidden_state_0)
		self.lin2 = nn.Linear(hidden_state_0,hidden_state_1)
		self.lin3  = nn.Linear(hidden_state_1,10)
		self.relu = nn.ReLU()

	def forward(self,x):
		x = self.relu(self.lin1(x))
		x = self.relu(self.lin2(x))
		return self.lin3(x)

net = BaseNeuralNet()

#Printing total parameter of the neural network [All trainable]



print("Before Applying LoRA ....")
total_parameters_org = 0

for layer in net.state_dict().keys():
	name = '.'.join(layer.split('.')[:-1])
	kind = layer.split('.')[-1]
	print(f'Layer {name} : {kind} : Shape : {net.state_dict()[layer].shape}')
	total_parameters_org +=net.state_dict()[layer].nelement()
print("Total number of parameters [Train]:",total_parameters_org)
	# print(net.state_dict()[layer].shape)


#--------LoRA Parameterization -------

class LoRAParameterization(nn.Module):
	def __init__(self,features_in:int,features_out:int,rank=1,alpha=1,device='cpu'):
		super().__init__()

		self.lora_A = nn.Parameter(torch.zeros(rank,features_out).to(device))
		self.lora_B = nn.Parameter(torch.zeros(features_in,rank).to(device))

		#Random Gautian initialization of Lora_A [As said in paper]
		nn.init.normal_(self.lora_A,mean=0,std=1)

		self.scale = alpha / rank

		self.enable=True

	def forward(self,original_weights):
		if self.enable:
			return original_weights + torch.matmul(self.lora_B,self.lora_A).view(original_weights.shape) * self.scale
		
		else :
			return original_weights

def linear_layer_parameterize(layer,device='cpu',rank=1,lora_alpha=1):
	feature_in,features_out = layer.weight.shape

	return LoRAParameterization(feature_in,features_out,rank=rank,alpha=lora_alpha,device=device)


for n,m in net.named_children():
	if n.find("lin")==0:
		torch.nn.utils.parametrize.register_parametrization(m,'weight',linear_layer_parameterize(m))

# print(net)


#Printing Total Parameters after LORA
print("\n\nAfter Applying LoRA ...... ")
total_param_lora = 0
total_param_non_lora=0
for n,layer in net.named_children():
	if n.find("lin")==0:
		weight = layer.weight.shape
		bias = layer.bias.shape
		lora_A = layer.parametrizations['weight'][0].lora_A.shape
		lora_B = layer.parametrizations['weight'][0].lora_B.shape
		total_param_non_lora += layer.weight.nelement() + layer.bias.nelement()
		total_param_lora +=layer.parametrizations['weight'][0].lora_A.nelement() + layer.parametrizations['weight'][0].lora_B.nelement()
		print(f'Layer {n} : W :{weight} + B : {bias} + Lora_A : {lora_A} + Lora_B : {lora_B}')

print(f'Total Lora Parameters [Train]: {total_param_lora}\nTotal Non Lora Parameters : {total_param_non_lora}\nTotal Increment of parameters : {(total_param_lora/total_param_non_lora)*100:.3f}%')


