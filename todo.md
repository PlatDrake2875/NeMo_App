# General rules:
- We've got Docker and hot reload. Do not rebuild or restart any container without my permission
- For each task, we need to work cleanly on branches (that are not the 'main' branch, of course), so please stick with that. We'll make branches, PRs and we'll review them
- Follow SOLID, DRY and other clean software development principles
- Everything (that is reasonable in regards of a systems design perspective) should be kept in the PostgresDB we're using

# Tasks
### Critical
[x] Processing datasets isn't working. They either fail or are kept indefinitely in a 'processing' state. (FIXED: parameter name mismatch in preprocessing_pipeline.py)
[x] Remove the frontend/src/components/rag-benchmark-hub/RAGHubSidebar.jsx and replace it with a dropdown in the header in the render view. (DONE: Replaced sidebar with Select dropdown in header)
[x] Add more preprocessing methods (cleaning and metadata extraction methods) (DONE: Added text cleaners + lightweight metadata extractor)
[x] Change the interface to be more compact (buttons are too spaced out and are too small, chunk size cursors are not precise). We should keep a scientific, pragmatic, usable approach. (DONE: Compact 2-col layouts, smaller controls, numeric inputs + sliders for precision)

## Minor
[x] The light mode button should be present on all pages, not just the ChatInterface. It is currently bugged, as there are three or so different places from which you can change the light mode. We should have a general button for that (DONE: Consolidated to single toggle in Sidebar, components use useTheme hook directly)