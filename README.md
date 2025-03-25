# Insurance Conversational Assistant

## Project Structure

The standard structure of a Rasa project is as follows:
```
.
├── actions
│   ├── models                (directory with trained models)
│   ├── data
│   │    ├── FAQ.csv         (data)
│   │    └── selection.yaml   (config to select questions and answers)
│   ├── __init__.py
│   ├── config.py             (pipeline configs)
│   ├── retrieval_pipeline.py (NLU/FAQ retrieval pipeline code)
│   └── actions.py           (custom actions)
├── config.yml
├── credentials.yml
├── experiments_notebooks     (experiment notebooks)
|    ├── 1.Data exploration.ipynb   
|    └── 2.Intent Classification Modelling.ipynb
├── data
│   ├── nlu.yml
│   └── stories.yml
├── domain.yml
├── endpoints.yml
├── models
│   └── <timestamp>.tar.gz
└── tests
│   └── test_stories.yml
└── requirements.txt
```
## How to Run

- You need a GPU (with CUDA installed)
```console
$ pip install -r requirements.txt
```
- Modify the ```data/selection.yaml``` file with the selection of question and answer indices (between 0 and total number of entries)
```console
$ rasa train
$ rasa run actions --port 5056
```
- In another terminal
```console
$ rasa train
$ rasa shell
```
## Project Overview

This project was inspired by the Haystack REST API/Rasa Action Server framework

(see more details at [Chatbot integration](https://haystack.deepset.ai/guides/chatbots))

It includes:
- Quantitative baselines for intent classification models (knowledge questions)
- Ability to add new questions and answers on the fly with very few examples per intent
- Code that integrates with local custom actions and the Haystack REST API framework
- Experiment notebooks with analyses

### Technical Considerations
The chatbot is developed using the *Rasa Open Source* framework.

Here are some useful links:

- The [Rasa Open Source GitHub repository](https://github.com/RasaHQ/rasa)
- The [Official Rasa Open Source documentation](https://rasa.com/docs/rasa/)
  - If you're new to Rasa, you can check out the [Rasa Playground](https://rasa.com/docs/rasa/playground)
  - You can find details about [Rasa Open Source installation](https://rasa.com/docs/rasa/installation)
  - And about the [Command Line Interface](https://rasa.com/docs/rasa/command-line-interface) including the ```rasa init``` command to start your project
- The [official tutorials](https://rasa.com/blog/category/tutorials/) available on the Rasa blog

- The chatbot can be developed using either Rasa Open Source *2.x* or *3.x*
- No graphical interface is required, you can interact with the chatbot either:
  - Via the ``rasa shell`` command
  - Via the [Rasa X](https://rasa.com/docs/rasa-x/) graphical interface
