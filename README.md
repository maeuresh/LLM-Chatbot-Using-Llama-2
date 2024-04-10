# LLM-Chatbot-Using-Llama-2
A LLM chatbot made using quantized Llama 2 model and Cricket World Cup Dataset. 

**Important**

Download the Quantized LLama 2 Model form [here]( https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q4_K_M.bin) or click this link ->  https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q4_K_M.bin

>**Run the code using:>chainlit run model.py -w**

>**Read the important.txt file first before running the code**

>**Install the requiremnts from the requirements.txt file**

>**Install FAISS package using the following command:** conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

For this LLM chatbot I have used:
1) Sentence Transformers for embeddings
2) Faiss CPU for vector storage
3) Quantized Llama 2 large language model
4) the Chainlit library for a conversational interface
