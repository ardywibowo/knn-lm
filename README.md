# K-Nearest Neighbors Augmented Language Models

This is a HuggingFace's 🤗 `transformers` + Lightning ⚡️ implementation of K-Nearest Neighbors Augmented Language Models, designed to be easy to read & understand, useful in research, and for experimenting with new kNN-based model ideas.

The implementation is originally based on the [k-NN Transformers](https://github.com/neulab/knn-transformers) repository. I found the originally implementation difficult to work with, especially for distributed environments. I reimplemented the method and made it compatible with Lightning ⚡️, and allows parallelization along multiple nodes and GPUs, as well as training using DeepSpeed through Lightning ⚡️.

The repository currently implements [k-nearest-neighbor language model (kNN-LM)](https://arxiv.org/pdf/1911.00172.pdf) (Khandelwal et al., ICLR'2020). Efforts to implement [k-nearest-neighbor machine translation (kNN-MT)](https://arxiv.org/pdf/2010.00710) (Khandelwal et al., ICLR'2021) and [Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval](https://arxiv.org/pdf/2201.12431.pdf) (ICML'2022), as well as decoder-style architectures (GPT-based) is planned in the future.

## Quickstart

There are 4 main files in `knnlm/training`:

- `generate.py` Generates a `.arrow` tokenized dataset from a `.jsonl` file of input-output pairs.
- `train.py` Trains the model on the generated dataset.
- `store.py` Generates a `faiss` Approximate Nearest Neighbor (ANN) index from the training set.
- `eval.py` Evaluates the model with/without the index.

All of these steps is controlled by a single config file `knnlm/configs/main.yaml`. Simply specify the path to your data (`train_path`, `val_path`), the path to save the store (`store_dir`), the path to the checkpoint to your finetuned model (`checkpoint`), and all of the other typical training params (base model name, training parameters). Then, you can run the code in the sequence above (generate, train, store, and eval).

## Acknowledgements

- [k-NN Transformers](https://github.com/neulab/knn-transformers) for the original code.
- [k-NN LM](https://github.com/urvashik/knnlm) for the original KNN-LM implementation.
