**Sparse Phased Transformer (SPT)**

This is the official repo for the paper "[Multimodal Phased Transformer for Sentiment Analysis](https://aclanthology.org/2021.emnlp-main.189.pdf)"

Preprocessed MOSI and MOSEI dataset by MulT download in https://github.com/yaohungt/Multimodal-Transformer 
UR-FUNNY dataset download in https://github.com/ROC-HCI/UR-FUNNY 

Use run.py to run the model, use Optuna to search hyper-params.


### Citation
If you use this code in your research, please cite the following paper:


``` bibtex
@inproceedings{cheng-etal-2021-multimodal,
    title = "Multimodal Phased Transformer for Sentiment Analysis",
    author = "Cheng, Junyan  and
      Fostiropoulos, Iordanis  and
      Boehm, Barry  and
      Soleymani, Mohammad",
    editor = "Moens, Marie-Francine  and
      Huang, Xuanjing  and
      Specia, Lucia  and
      Yih, Scott Wen-tau",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.189/",
    doi = "10.18653/v1/2021.emnlp-main.189",
    pages = "2447--2458",
    abstract = "Multimodal Transformers achieve superior performance in multimodal learning tasks. However, the quadratic complexity of the self-attention mechanism in Transformers limits their deployment in low-resource devices and makes their inference and training computationally expensive. We propose multimodal Sparse Phased Transformer (SPT) to alleviate the problem of self-attention complexity and memory footprint. SPT uses a sampling function to generate a sparse attention matrix and compress a long sequence to a shorter sequence of hidden states. SPT concurrently captures interactions between the hidden states of different modalities at every layer. To further improve the efficiency of our method, we use Layer-wise parameter sharing and Factorized Co-Attention that share parameters between Cross Attention Blocks, with minimal impact on task performance. We evaluate our model with three sentiment analysis datasets and achieve comparable or superior performance compared with the existing methods, with a 90{\%} reduction in the number of parameters. We conclude that (SPT) along with parameter sharing can capture multimodal interactions with reduced model size and improved sample efficiency."
}
```
