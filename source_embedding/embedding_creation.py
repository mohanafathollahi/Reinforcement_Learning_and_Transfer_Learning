from source_embedding.embedding_helper import *
#1: create sample_df by calling create_random_samples
sample_df = create_random_samples(number_of_samples = 10)
# print(sample_df)
#2: create an embedding vector
create_source_embedding_tensor(sample_df = sample_df)

