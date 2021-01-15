Everything is anchored to paper_experiment_sdcfr*

mo_FHP_2: 
- 40 workers (2x paper)
- SINGLE only
- 1/3 n_traversals
- 1/2 n_batches
- 1/2 max_buffer_size

mo_FHP_3:
- Same as mo_FHP_2 but with modified game that allows for 3 raises in the first street

mo_FHP_cluster:

mo_leduc:

mo_HULH:
- everything as mo_FHP_2 except
- 1/10 mo_FHP_2 = 500 traversals/LA
- for some reason maxbuffer size = 2e6 not 1e6

