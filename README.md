# :speech_balloon: Chat, Talk, Learn and Analyse your spending habits with your Personal Finance Assistant

# Demo

TBD

# :bar_chart: Project diagram 

![rag-dag](project-info/rag_dag.png)

* Dataset source - [Kaggle](https://www.kaggle.com/datasets/rajatsurana979/comprehensive-credit-card-transactions-dataset)
* Database - SQLite
* LLM - OpenAI's GPT
* Audio Model - openai/whisper-small from [HuggingFace](https://huggingface.co/openai/whisper-small)
* RAG pipeline - [LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/)
* Monitoring/Tracing - [Azire Pheonix](https://github.com/Arize-ai/phoenix)
* User Interface - Streamlit


# :evergreen_tree: Folder Tree 

```
├── app/
│   ├── app.py (streamlit)
│   ├── table_index_dir/
│   │   └── Ivanov_Transactions/
│   │       ├── (data index files) *.json
│   ├── table_info_directory/
│   │   └── (data) 0_Ivanov_Transactions.json
│   └── (data) transactions.db
├── dev/
│   ├── (files used for setup and during development)
├── Makefile
├── project-info/
├── README.md
└── requirements.txt
```

# Reproduction

1. Clone the repo `git clone https://github.com/divakaivan/finance_voice_assistant.git` 
2. Run `make` in the terminal, and you should see:
```
Usage: make [option]

Options:
  help                 Show this help message
  install              Install dependencies
  data                 Prepare data for the app (will overwrite any existing data)
  start                Start app
  del-audio            Delete any generated audio files
```
If running for the first time, run the setup:
* `make install`
* `make data`

Otherwise:
* `make start` - this will open the streamlit UI in your browser on `http://localhost:8501`

