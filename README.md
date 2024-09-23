### Idea:
1. Create new Project
    1. Sets up SQL Database

2. Input first cases
    1. Save cases to database
    2. Create vectors for each case
        A: Prompt/Model optional (if not, use default), Also vectors as input possible -> Make it modular comparable BERTopic
    3. Save vectors to database
    4. Option to generate synthetic reasoning for each case (optional)

3. Select Model to classify new cases
    1. Use OpenAI API / Ollama / Custom Huggingface Model / Plugin own model (must be capable of handling list of dicts as input)
    2. Optional to use synthetic reasoning or even generate new synthetic reasoning for each case at this point and save it to database
    3. Save run to database

4. Automate Evaluation
    1. Use Gold Standard Annotations vs. predicted annotations
    2. Visualize Results
    3. Visualize "train" cases using their vectors and bertopic

5. Export Results
    1. Export Results to CSV/EXCEL

Train own model (optional)
Track Approaches (optional)
    - Track different combinations of models/prompts/reasoning/metrics etc. 
    - Moonshot: Used by different users, track their results and compare them

### Input Datasets needs:
1. Input (Text for the beggining) to annotate/classify etc.
2. Gold Standard annotations/classifications of each case
3. Reasoning for each case (optional)


### Challenges
- Make a structure general enough to be parsed and used by any kind of case.

- Make project handling easy and intuitive for the user.
  This includes:
    - CLI and Code capable
    - Connection to database (SQL) and handling of data easily | Idea: Connect to database
    - Adding new cases easily | Idea: Using batch identifier


### Open Questions:
- Only save on explicit command or save automatically?
   - Specify during test/addig cases etc.


To Do Monday:
- [X] query and insertionn of meta for full database
- [X] generation of synthetic reasoning
- [ ] use custom embedding model