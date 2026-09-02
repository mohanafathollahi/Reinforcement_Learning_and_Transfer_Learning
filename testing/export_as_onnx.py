#this script will give us onnx model
import  os
import torch
from custom_env.clipaction_env import CustomEnv
from models.sac_model import SacModel
from models.td3_model import Td3Model

containing_dir = os.path.dirname(os.path.realpath(__file__)) #path of the current working directory
onnx_path = os.path.join(containing_dir, 'onnx_model','export_model_test.onnx')
print(onnx_path)

class OnnxableTD3Policy(torch.nn.Module):
    def __init__(self, 
                 actor: torch.nn.Module):
        super().__init__()
        self.actor = actor
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)

class OnnxableSACPolicy(torch.nn.Module):
    def __init__(self,
                 actor: torch.nn.Module):
        super().__init__()
        self.actor = actor
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation, deterministic=True)

#take the trained model and convert it to onnx
def create_onnx(model_name,
                volt_lb,
                volt_up,
                episode_length,
                pref_range,
                directory_pretrain_model,
                base_path_to_save_onnx):

    onnx_dir = os.path.join(base_path_to_save_onnx, model_name)
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, f"{model_name}.onnx")

    env = CustomEnv(volt_lb=volt_lb, 
                    volt_up=volt_up,
                    episode_length = episode_length,
                    pref_range=pref_range)

    if "td3" in model_name.lower():
        print(f"model_name:{model_name}")
        print("convert td3 to onnx")
        model = Td3Model().get_pretrained_output(env,
                                                device="cpu",
                                                directory_pretrain_model= directory_pretrain_model,
                                                return_model_only = True)
        onnxable_model = OnnxableTD3Policy(model.policy.actor)

    else:
        print(f"model_name:{model_name}")
        print("convert SAC model to onnx")
        model = SacModel().get_pretrained_output(env,
                                                 device="cpu",
                                                 deterministic=True,
                                                 directory_pretrain_model=directory_pretrain_model,
                                                 return_model_only=True)

        onnxable_model = OnnxableSACPolicy(model.policy.actor)

    observation_size = model.observation_space.shape        #it return 4 which is correct
    dummy_input = torch.randn(1, *observation_size)
    print("Saving ONNX to:", onnx_path)
    torch.onnx.export(onnxable_model,
                      dummy_input, 
                      onnx_path,
                      opset_version = 10,
                      input_names = ["input"],
                     )