import torch as th

#we are considering the penultimate layer of first critic network as an embedding
#possible consideration: consider the mean of two penultimate layers of two critic networks!

def get_source_embeddings(env_id,pref,embedding_type):
    pen_qf0_dict = th.load(f"source_embedding/embedding_sources/penultimate_{embedding_type}.pt") #size for each key: torch.Size([10, 256])
    q_value_qf0_dict = th.load(f"source_embedding/embedding_sources/q_val_{embedding_type}.pt") #size of each key: torch.Size([10, 256])

    for key in pen_qf0_dict.keys():
        if int(key[0].split("_")[1]) == env_id and int(key[1]) == int(pref):
            if key in q_value_qf0_dict:
                pen_tensor_embedding = pen_qf0_dict[key]
                q_value_tensor = q_value_qf0_dict[key]
                return pen_tensor_embedding, q_value_tensor

#to check the size of embedding layer
# pen_tensor_embedding, q_value_tensor = get_source_embeddings(env_id=1, pref = 200)
# print(len(pen_tensor_embedding[0]))
# print(len(q_value_tensor[0]))