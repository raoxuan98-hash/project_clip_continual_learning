# Local and Generated Artifacts

This directory consolidates generated outputs, raw tool sessions, local logs,
and historical backups that are not part of the maintained source tree.

## Layout

- `logs/`: experiment and launcher logs;
- `results/`: local or historical experiment outputs that are not yet part of
  a frozen publication package;
- `figures/`: generated analysis plots; final paper figures belong under
  `paper_writing/paper-template/figures/`;
- `tool-runs/arbor/`: raw Arbor sessions and coordinator state;
- `archive/backups/`: historical code backups retained temporarily.

The contents are ignored by Git by default. Any result used to support a paper
claim must also be registered in chat history and the latest claim-evidence
ledger with its remote source, configuration, and code version.
