from llama_index import ServiceContext, SimpleDirectoryReader, LLMPredictor, PromptHelper, GPTVectorStoreIndex, StorageContext, load_index_from_storage

from langchain import OpenAI
import os

os.environ["OPENAI_API_KEY"] = ""
max_input = 4096
tokens = 256
chunk_size = 600
max_chunk_overlap = 20
promp_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

llmpredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))
service_context = ServiceContext.from_defaults(llm_predictor=llmpredictor, prompt_helper=promp_helper)

def createVector(path):

  promp_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

  llmpredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))
  service_context = ServiceContext.from_defaults(llm_predictor=llmpredictor, prompt_helper=promp_helper)

  documents = SimpleDirectoryReader(path).load_data()
  index = GPTVectorStoreIndex.from_documents(documents=documents, service_context=service_context)
  print('index', index)
  index.storage_context.persist(persist_dir="./index")
  return index

def answerMe():
  # rebuild storage context
  storage_context = StorageContext.from_defaults(persist_dir="./index")
  # load index
  index = load_index_from_storage(storage_context)

  while True:
      prompt = input("Ask the farm-gpt  ")
      query_engine = index.as_query_engine(
        response_mode="tree_summarize"
      )
      response = query_engine.query(prompt)
      print(f"Resonse: {response} \n")
# createVector("./data")
answerMe()