# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/blob/main/weaviate-features/multi-vector/multi-vector-colipali-rag.ipynb)

# %% [markdown]
# # Multimodal RAG over PDFs using ColQwen2, Qwen2.5, and Weaviate
# 
# This notebook demonstrates [Multimodal Retrieval-Augmented Generation (RAG)](https://weaviate.io/blog/multimodal-rag) over PDF documents.
# We will be performing retrieval against a collection of PDF documents by embedding both the individual pages of the documents and our queries into the same multi-vector space, reducing the problem to approximate nearest-neighbor search on ColBERT-style multi-vector embeddings under the MaxSim similarity measure.
# 
# For this purpose, we will use
# 
# - **A multimodal [late-interaction model](https://weaviate.io/blog/late-interaction-overview)**, like ColPali and ColQwen2, to generate
# embeddings. This tutorial uses the publicly available model
# [ColQwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) with a permissive Apache 2.0 license.
# - **A Weaviate [vector database](https://weaviate.io/blog/what-is-a-vector-database)**, which  has a [multi-vector feature](https://docs.weaviate.io/weaviate/tutorials/multi-vector-embeddings) to effectively index a collection of PDF documents and support textual queries against the contents of the documents, including both text and figures.
# - **A vision language model (VLM)**, specifically [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), to support multimodal Retrieval-Augmented Generation (RAG).
# 
# Below, you can see the multimodal RAG system overview:
# 
# <img src="https://github.com/weaviate/recipes/blob/main/weaviate-features/multi-vector/figures/multimodal-rag-diagram.png?raw=1" width="700px"/>
# 
# First, the ingestion pipeline processes the PDF documents as images with the multimodal late-interaction model. The multi-vector embeddings are stored in a vector database.
# Then at query time, the text query is processed by the same multimodal late-interaction model to retrieve the relevant documents.
# The retrieved PDF files are then passed as visual context together with the original user query to the vision language model, which generates a response based on this information.
# 

# %% [markdown]
# ## Prerequisites
# 
# To run this notebook, you will need a machine capable of running neural networks using 5-10 GB of memory.
# The demonstration uses two different vision language models that both require several gigabytes of memory.
# See the documentation for each individual model and the general PyTorch docs to figure out how to best run the models on your hardware.
# 
# For example, you can run it on:
# 
# - Google Colab (using the free-tier T4 GPU)
# - or locally (tested on an M2 Pro Mac).
# 
# Furthermore, you will need an instance of Weaviate version >= `1.29.0`.
# 

# %% [markdown]
# ## Step 1: Install required libraries
# 
# Let's begin by installing and importing the required libraries.
# 
# Note that you'll need Python `3.13`.

# %%
%%capture
%pip install colpali_engine weaviate-client qwen_vl_utils
%pip install -q -U "colpali-engine[interpretability]>=0.3.2,<0.4.0"

# %%
import os
import torch
import numpy as np

from google.colab import userdata
from datasets import load_dataset

from transformers.utils.import_utils import is_flash_attn_2_available
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from colpali_engine.models import ColQwen2, ColQwen2Processor
#from colpali_engine.models import ColPali, ColPaliProcessor # uncomment if you prefer to use ColPali models instead of ColQwen2 models

import weaviate
from weaviate.classes.init import Auth
import weaviate.classes.config as wc
from weaviate.classes.config import Configure
from weaviate.classes.query import MetadataQuery

from qwen_vl_utils import process_vision_info
import base64
from io import BytesIO

import matplotlib.pyplot as plt

from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
    plot_similarity_map,
)


# %% [markdown]
# ## Step 2: Load the PDF dataset
# 
# Let's start with the data.
# We're going to first load a PDF document dataset of the [top-40 most
# cited AI papers on arXiv](https://arxiv.org/abs/2412.12121) from Hugging Face from the period 2023-01-01 to 2024-09-30.

# %%
dataset = load_dataset("weaviate/arXiv-AI-papers-multi-vector", split="train")

# %%
dataset

# %%
dataset[398]

# %% [markdown]
# Let's take a look at a sample document page from the loaded PDF dataset.

# %%
display(dataset[289]["page_image"])

# %% [markdown]
# ![Retrieved page](./figures/retrieved_page.png)

# %% [markdown]
# ## Step 3: Load the ColVision (ColPali or ColQwen2) model
# 
# The approach to generate embeddings for this tutorial is outlined in the paper [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449). The paper demonstrates that it is possible to simplify traditional approaches to preprocessing PDF documents for retrieval:
# 
# Traditional PDF processing in RAG systems involves using OCR (Optical Character Recognition) and layout detection software, and separate processing of text, tables, figures, and charts. Additionally, after text extraction, text processing also requires a chunking step. Instead, the ColPali method feeds images (screenshots) of entire PDF pages to a Vision Language Model that produces a ColBERT-style multi-vector embedding.
# 
# <img src="https://github.com/weaviate/recipes/blob/main/weaviate-features/multi-vector/figures/colipali_pipeline.jpeg?raw=1" width="700px"/>
# 
# There are different ColVision models, such as ColPali or ColQwen2, available, which mainly differ in the used encoders (Contextualized Late Interaction over Qwen2 vs. PaliGemma-3B). You can read more about the differences between ColPali and ColQwen2 in our [overview of late-interaction models](https://weaviate.io/blog/late-interaction-overview).
# 
# Let's load the [ColQwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) model for this tutorial.

# %%
# Get rid of process forking deadlock warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
if torch.cuda.is_available(): # If GPU available
    device = "cuda:0"
elif torch.backends.mps.is_available(): # If Apple Silicon available
    device = "mps"
else:
    device = "cpu"

if is_flash_attn_2_available():
    attn_implementation = "flash_attention_2"
else:
    attn_implementation = "eager"

print(f"Using device: {device}")
print(f"Using attention implementation: {attn_implementation}")

# %%
model_name = "vidore/colqwen2-v1.0"

# About a 5 GB download and similar memory usage.
model = ColQwen2.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation=attn_implementation,
).eval()

# Load processor
processor = ColQwen2Processor.from_pretrained(model_name)

# %% [markdown]
# This notebook uses the ColQwen2 model because it has a permissive Apache 2.0 license.
# Alternatively, you can also use [ColPali](https://huggingface.co/vidore/colpali-v1.2), which has a Gemma license, or check out other available [ColVision models](https://github.com/illuin-tech/colpali). For a detailed comparison, you can also refer to [ViDoRe: The Visual Document Retrieval Benchmark](https://huggingface.co/spaces/vidore/vidore-leaderboard)
# 
# If you want to use ColPali instead of ColQwen2, you can comment out the above code cell and uncomment the code cell below.

# %%
#model_name = "vidore/colpali-v1.2"

# Load model
#colpali_model = ColPali.from_pretrained(
#    model_name,
#    torch_dtype=torch.bfloat16,
#    device_map=device,
#    attn_implementation=attn_implementation,
#).eval()

# Load processor
#colpali_processor = ColPaliProcessor.from_pretrained(model_name)

# %% [markdown]
# Before we go further, let's familiarize ourselves with the ColQwen2 model. It can create multi-vector embeddings from both images and text queries. Below you can see examples of each.
# 

# %%
# Sample image inputs
images = [
    dataset[0]["page_image"],
    dataset[1]["page_image"],
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)

# Forward pass
with torch.no_grad():
    query_embedding = model(**batch_images)

print(query_embedding)
print(query_embedding.shape)

# %%
# Sample query inputs
queries = [
    "A table with LLM benchmark results.",
    "A figure detailing the architecture of a neural network.",
]

# Process the inputs
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    query_embedding = model(**batch_queries)

print(query_embedding)
print(query_embedding.shape)

# %% [markdown]
# Let's write a class to wrap the multimodal late-interaction model and its embedding functionalities for convenience.
# 
# 

# %%
# A convenience class to wrap the embedding functionality 
# of ColVision models like ColPali and ColQwen2 
class ColVision:
    def __init__(self, model, processor):
        """Initialize with a loaded model and processor."""
        self.model = model
        self.processor = processor

    # A batch size of one appears to be most performant when running on an M4.
    # Note: Reducing the image resolution speeds up the vectorizer and produces
    # fewer multi-vectors.
    def multi_vectorize_image(self, img):
        """Return the multi-vector image of the supplied PIL image."""
        image_batch = self.processor.process_images([img]).to(self.model.device)
        with torch.no_grad():
            image_embedding = self.model(**image_batch)
        return image_embedding[0]

    def multi_vectorize_text(self, query):
        """Return the multi-vector embedding of the query text string."""
        query_batch = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            query_embedding = self.model(**query_batch)
        return query_embedding[0]

# Instantiate the model to be used below.
colvision_embedder = ColVision(model, processor) # This will be instantiated after loading the model and processor

# %% [markdown]
# Let's verify that the embedding of images and queries works as intended.
# 

# %%
# Sample image inputs
images = dataset[0]["page_image"]

page_embedding = colvision_embedder.multi_vectorize_image(images)
print(page_embedding.shape)  # torch.Size([755, 128])

queries = [
    "A table with LLM benchmark results.",
    "A figure detailing the architecture of a neural network.",
]

query_embeddings = [colvision_embedder.multi_vectorize_text(q) for q in queries]
print(query_embeddings[0].shape)  # torch.Size([20, 128])

# %% [markdown]
# ## Step 4: Connect to a Weaviate vector database instance
# 
# Now, you will need to connect to a running Weaviate vector database cluster.
# 
# You can choose one of the following options:
# 
# 1. **Option 1:** You can create a 14-day free sandbox on the managed service [Weaviate Cloud (WCD)](https://console.weaviate.cloud/)
# 2. **Option 2:** [Embedded Weaviate](https://docs.weaviate.io/deploy/installation-guides/embedded)
# 3. **Option 3:** [Local deployment](https://docs.weaviate.io/deploy/installation-guides/docker-installation)
# 4. [Other options](https://docs.weaviate.io/deploy)

# %%
# Option 1: Weaviate Cloud
WCD_URL = os.environ["WEAVIATE_URL"] # Replace with your Weaviate cluster URL
WCD_AUTH_KEY = os.environ["WEAVIATE_API_KEY"] # Replace with your cluster auth key

# Uncomment if you are working in a Google Colab environment
#WCD_URL = userdata.get("WEAVIATE_URL")
#WCD_AUTH_KEY = userdata.get("WEAVIATE_API_KEY")

# Weaviate Cloud Deployment
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WCD_URL,
    auth_credentials=weaviate.auth.AuthApiKey(WCD_AUTH_KEY),
)

# Option 2: Embedded Weaviate instance
# use if you want to explore Weaviate without any additional setup
#client = weaviate.connect_to_embedded()

# Option 3: Locally hosted instance of Weaviate via Docker or Kubernetes
#!docker run --detach -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.29.0
#client = weaviate.connect_to_local()

print(client.is_ready())

# %% [markdown]
# For this tutorial, you will need the Weaviate `v1.29.0` or higher.
# Let's make sure we have the required version:

# %%
client.get_meta()['version']

# %% [markdown]
# ## Step 5: Create a collection
# 
# Next, we will create a collection that will hold the embeddings of the images of the PDF document pages.
# 
# We will not define a built-in vectorizer but use the [Bring Your Own Vectors (BYOV) approach](https://docs.weaviate.io/weaviate/starter-guides/custom-vectors), where we manually embed queries and PDF documents at ingestions and query stage.
# 
# Additionally, if you are interested in using the [MUVERA encoding algorithm](https://weaviate.io/blog/muvera) for multi-vector embeddings, you can uncomment it in the code below.

# %%
collection_name = "PDFDocuments"

# %%
# Delete the collection if it already exists
# Note: in practice, you shouldn't rerun this cell, as it deletes your data
# in "PDFDocuments", and then you need to re-import it again.
#if client.collections.exists(collection_name):
#  client.collections.delete(collection_name)

# Create a collection
collection = client.collections.create(
    name=collection_name,
    properties=[
        wc.Property(name="page_id", data_type=wc.DataType.INT),
        wc.Property(name="dataset_index", data_type=wc.DataType.INT),
        wc.Property(name="paper_title", data_type=wc.DataType.TEXT),
        wc.Property(name="paper_arxiv_id", data_type=wc.DataType.TEXT),
        wc.Property(name="page_number", data_type=wc.DataType.INT),
    ],
    vector_config=[
        Configure.MultiVectors.self_provided(
            name="colqwen",
            #encoding=Configure.VectorIndex.MultiVector.Encoding.muvera(),
            vector_index_config=Configure.VectorIndex.hnsw(
                multi_vector=Configure.VectorIndex.MultiVector.multi_vector()
            )
    )]
)

# %% [markdown]
# ## Step 6: Uploading the vectors to Weaviate
# 
# In this step, we're indexing the vectors into our Weaviate Collection in batches.
# 
# For each batch, the images are processed and encoded using the ColPali model, turning them into multi-vector embeddings.
# These embeddings are then converted from tensors into lists of vectors, capturing key details from each image and creating a multi-vector representation for each document.
# This setup works well with Weaviate's multivector capabilities.
# 
# After processing, the vectors and any metadata are uploaded to Weaviate, gradually building up the index.
# You can lower or increase the `batch_size` depending on your available GPU resources.

# %%
# Map of page ids to images to support displaying the image corresponding to a
# particular page id.
page_images = {}

with collection.batch.dynamic() as batch:
    for i in range(len(dataset)):
        p = dataset[i]
        page_images[p["page_id"]] = p["page_image"]

        batch.add_object(
            properties={
                "page_id": p["page_id"],
                "paper_title": p["paper_title"],
                "paper_arxiv_id": p["paper_arxiv_id"],
                "page_number": p["page_number"],
                },
            vector={"colqwen": colvision_embedder.multi_vectorize_image(p["page_image"]).cpu().float().numpy().tolist()})

        if i % 25 == 0:
            print(f"Added {i+1}/{len(dataset)} Page objects to Weaviate.")

    batch.flush()

# Delete dataset after creating page_images dict to hold the images
del dataset

# %%
len(collection)

# %% [markdown]
# ## Step 7: Multimodal Retrieval Query
# 
# As an example of what we are going to build, consider the following actual demo query and resulting PDF page from our collection (nearest neighbor):
# 
# - Query: "How does DeepSeek-V2 compare against the LLaMA family of LLMs?"
# - Nearest neighbor:  "DeepSeek-V2: A Strong Economical and Efficient Mixture-of-Experts Language Model" (arXiv: 2405.04434), Page: 1.
# 

# %%
query = "How does DeepSeek-V2 compare against the LLaMA family of LLMs?"

# %% [markdown]
# By inspecting the first page of the [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434), we see that it does indeed contain a figure that is relevant for answering our query:
# 
# <img src="https://github.com/weaviate/recipes/blob/main/weaviate-features/multi-vector/figures/deepseek_efficiency.jpeg?raw=1" width="700px"/>

# %% [markdown]
# Note: To avoid `OutOfMemoryError` on freely available resources like Google Colab, we will only retrieve a single document. If you have resources with more memory available, you can set the `limit`parameter to a higher value, like e.g., `limit=3` to increase the number of retrieved PDF pages.

# %%
response = collection.query.near_vector(
    near_vector=colvision_embedder.multi_vectorize_text(query).cpu().float().numpy(),
    target_vector="colqwen",
    limit=1,
    return_metadata=MetadataQuery(distance=True), # Needed to return MaxSim score
)

print(f"The most relevant documents for the query \"{query}\" by order of relevance:\n")
result_images = []
for i, o in enumerate(response.objects):
    p = o.properties
    print(
        f"{i+1}) MaxSim: {-o.metadata.distance:.2f}, "
        + f"Title: \"{p['paper_title']}\" "
        + f"(arXiv: {p['paper_arxiv_id']}), "
        + f"Page: {int(p['page_number'])}"
    )
    result_images.append(page_images[p["page_id"]])

# %% [markdown]
# The retrieved page with the highest MaxSim score is indeed the page with the figure we mentioned earlier.

# %%
closest_page_id = response.objects[0].properties['page_id']
image = page_images[closest_page_id]
display(image)

# %% [markdown]
# ![Retrieved page](./figures/retrieved_page.png)
# 
# Let's visualize the similarity maps for the retrieved PDF document page to see the semantic similarity between each token in the user query and the image patches. This is an optional step.

# %%

# Preprocess inputs
batch_images = processor.process_images([image]).to(device)
batch_queries = processor.process_queries([query]).to(device)

# Forward passes
with torch.no_grad():
    image_embeddings = model.forward(**batch_images)
    query_embeddings = model.forward(**batch_queries)

# Get the number of image patches
n_patches = processor.get_n_patches(
    image_size=image.size,
    spatial_merge_size=model.spatial_merge_size,
)

# Get the tensor mask to filter out the embeddings that are not related to the image
image_mask = processor.get_image_mask(batch_images)

# Generate the similarity maps
batched_similarity_maps = get_similarity_maps_from_embeddings(
    image_embeddings=image_embeddings,
    query_embeddings=query_embeddings,
    n_patches=n_patches,
    image_mask=image_mask,
)

# Get the similarity map for our (only) input image
similarity_maps = batched_similarity_maps[0]  # (query_length, n_patches_x, n_patches_y)

print(f"Similarity map shape: (query_length, n_patches_x, n_patches_y) = {tuple(similarity_maps.shape)}")

# %%
# Remove the padding tokens and the query augmentation tokens
query_content = processor.decode(batch_queries.input_ids[0])
query_content = query_content.replace(processor.tokenizer.pad_token, "")
query_content = query_content.replace(processor.query_augmentation_token, "").strip()

# Retokenize the cleaned query
query_tokens = processor.tokenizer.tokenize(query_content)

# Use this cell output to choose a token using its index
for idex, val in enumerate(query_tokens):
    print(f"{idex}: {val}")

# %% [markdown]
# Let's check the similarity plot for the token "MA" in "LLaMA". (Note that similarity maps are created for each token separately.)

# %%
token_idx = 13

fig, ax = plot_similarity_map(
    image=image,
    similarity_map=similarity_maps[token_idx],
    figsize=(18, 18),
    show_colorbar=False,
)

max_sim_score = similarity_maps[token_idx, :, :].max().item()
ax.set_title(f"Token #{token_idx}: `{query_tokens[token_idx]}`. MaxSim score: {max_sim_score:.2f}", fontsize=14)

plt.show()

# %% [markdown]
# ![Similarity map](./figures/similarity_map.png)

# %%
# Delete variables used for visualization
del batched_similarity_maps, similarity_maps, n_patches, query_content, query_tokens, token_idx

# %% [markdown]
# ## Step 8: Extension to Multimodal RAG using Qwen2.5
# 
# The above example gives us the most relevant pages to begin looking at to answer our query. Let's extend this multimodal document retrieval pipeline to a multimodal RAG pipeline.
# 
# Vision language models (VLMs) are Large Language Models with vision capabilities. They are now powerful enough that we can give the query and relevant pages to such a model and have it produce an answer to our query in plain text.
# 
# To accomplish this we are going to feed the top results into the
# state-of-the-art VLM [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

# %%
# Setting up Qwen2.5-VL-3B-Instruct for generating answers from a query string
# plus a collection of (images of) PDF pages.

class QwenVL:
    def __init__(self):
        # Adjust the settings to your available architecture, see the link
        # https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct for examples.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_implementation,
        )

        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

    def query_images(self, query, images):
        """Generate a textual response to the query (text) based on the information in the supplied list of PIL images."""
        # Preparation for inference.
        # Convert the images to base64 strings.
        content = []

        for img in images:
            buffer = BytesIO()
            img.save(buffer, format="jpeg")
            img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            content.append({"type": "image", "image": f"data:image;base64,{img_base64}"})

        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output.
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

# Instantiate the model to be used below.
qwenvl = QwenVL()

# %% [markdown]
# The response from `Qwen2.5-VL-3B-Instruct` based on the retrieved PDF pages:

# %%
qwenvl.query_images(query, result_images)

# %% [markdown]
# As you can see, the multimodal RAG pipeline was able to answer the original query: "How does DeepSeek-V2 compare against the LLaMA family of LLMs?". For this, the ColQwen2 retrieval model retrieved the correct PDF page from the 
# "DeepSeek-V2: A Strong Economical and Efficient Mixture-of-Experts Language Model" paper and used both the text and visual from the retrieved PDF page to answer the question.

# %% [markdown]
# ## Summary
# 
# This notebook demonstrates a multimodal RAG pipeline over PDF documents using ColQwen2 for multi-vector embeddings, a Weaviate vector database for storage and retrieval, and Qwen2.5-VL-3B-Instruct for generating answers.

# %% [markdown]
# ## References
# 
# - Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., Colombo, P. (2024). ColPali: Efficient Document Retrieval with Vision Language Models. arXiv. https://doi.org/10.48550/arXiv.2407.01449
# - [ColPali GitHub repository](https://github.com/illuin-tech/colpali)
# - [ColPali Cookbook](https://github.com/tonywu71/colpali-cookbooks)


