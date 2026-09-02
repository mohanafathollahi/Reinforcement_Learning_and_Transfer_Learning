import os

from config.env_config import env_config
from testing.export_as_onnx import create_onnx

dir_pretrained_models =  "./pre_trained_models_dirctory"


def create_onnx_for_all_models(dir_pretrained_models= dir_pretrained_models):
        print(f"pretrained_model:{dir_pretrained_models}")
        for file in os.listdir(dir_pretrained_models):
            print(f"file:{file}")
            dir_model  = os.path.join(dir_pretrained_models,file)
            print(f"dir_model:{dir_model}")

            env_detail = env_config[0]
            create_onnx(model_name = "model_name",
                        volt_lb= env_detail['volt_lb'],
                        volt_up= env_detail['volt_up'],
                        episode_length= 1,
                        pref_range= list(range(env_detail['pref_lb'],
                                               env_detail['pref_ub'],
                                               env_detail['pref_step'])),

                        directory_pretrain_model= dir_model,
                        base_path_to_save_onnx='./onnx_models')

create_onnx_for_all_models(dir_pretrained_models= dir_pretrained_models)