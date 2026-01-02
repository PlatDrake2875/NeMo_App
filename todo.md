# General rules:
- We've got Docker and hot reload. Do not rebuild or restart any container without my permission
- For each task, we need to work cleanly on branches (that are not the 'main' branch, of course), so please stick with that. We'll make branches, PRs and we'll review them
- Follow SOLID, DRY and other clean software development principles

# Tasks
## Critical
[] We've got a 'Document Manager' page that should be in fact a hub for benchmarking RAG methods. Currently, it's a mix between everything. The workflow should be the following:
- Create a 'raw dataset'. This should be a space into which one can only put documents, be them pdfs, JSONs, md's, txt's, csv's that we don't apply any processing to
- The 'dataset' preprocessing pipeline, in which we take a 'raw dataset' and we apply cleaning, adding metadata (how would we do this? Via LLMs?), chunking methods and indexing in a vector DB. The first two steps should be optional, but the latter are mandatory
- A 'dataset' editor in which one can see each document (also provide a hyperlinked preview with a react-pdf modal and equivalent mediums to the original document; do that as well in the raw dataset section).

The document manager should actually be a dataset manager in which one can see the datasets, group them via DB and chunking method. 

[] I want a feature from which we could download datasets from HuggingFace (ex.: https://huggingface.co/datasets/microsoft/wiki_qa)

## Minor
[] The light mode button should be present on all pages, not just the ChatInterface