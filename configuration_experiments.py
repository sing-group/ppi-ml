import embeddings as em
import models.models as md

random_state = 2024
test_size = 0.2 # Use 0.0 to skip the initial train/test split
n_jobs = -1 # None means 1; -1 means all processors;
shuffle = False # Use True to run the sanity check
use_GPU = False # Use True to use the GPU
nested_cv_outer_splits = 5
nested_cv_inner_splits = 5
per_fold = True # True means that metris are computed for each fold separately and then averaged
print_debug_messages = False

datasets = ['dataset_clean_wei_seqvec.h5', 'dataset_clean_wei_protbert.h5']
models_to_exec = ['PPIIBM_first_item', 'PPIIBM_both_items']
models = md.prepare_models(models_to_exec, random_state, use_GPU)
embeddings_combinators = [
    em.ConcatEmbeddings(add_inverted_interactions=False)
]
