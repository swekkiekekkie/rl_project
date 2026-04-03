Yes. The main tool for this is **Jupytext**. It can convert `.ipynb` to `.py`, represent notebook cells with `# %%`, keep Markdown as commented text, and convert the `.py` form back into `.ipynb`. It also supports **paired notebooks**, where the `.ipynb` and `.py` stay in sync. ([Jupytext][1])

Typical commands:

```bash
# notebook -> python script
jupytext --to py notebook.ipynb

# notebook -> python script with explicit cell markers
jupytext --to py:percent notebook.ipynb

# python script -> notebook
jupytext --to notebook notebook.py

# keep a paired .ipynb + .py version in sync
jupytext --set-formats ipynb,py notebook.ipynb
jupytext --sync notebook.ipynb
```

Jupytext’s `py:percent` format is usually the most practical choice because it makes cell boundaries explicit with `# %%`, and many editors understand that format. In the script-based formats, Jupyter magics can be commented out by default, and that behavior is configurable. ([Jupytext][1])

There is also **`nbconvert`**, which can flatten a notebook to a script:

```bash
jupyter nbconvert --to script notebook.ipynb
```

That is good for **exporting** a notebook to a runnable script, but Jupytext is the better tool when you want reliable **round-tripping** between `.ipynb` and `.py`. ([nbconvert.readthedocs.io][2])

My recommendation:

* Use **Jupytext** for `.ipynb` ⇄ `.py`
* Use **`py:percent`** if you want comments plus recoverable notebook cells
* Use **`nbconvert`** only for one-way export

I can also give you a minimal setup for Jupytext with VS Code, JupyterLab, or pre-commit.

[1]: https://jupytext.readthedocs.io/en/latest/formats-scripts.html?utm_source=chatgpt.com "Notebooks as code — Jupytext documentation - Read the Docs"
[2]: https://nbconvert.readthedocs.io/_/downloads/en/latest/pdf/?utm_source=chatgpt.com "nbconvert Documentation"

`--sync` is **effectively bidirectional, but not merge-based**. Jupytext decides the source of truth by **which paired file is newer**. The CLI docs say `--sync` updates paired representations **based on timestamps**, and the paired-notebooks docs say inputs are loaded from **the most recent file in the pair**. So if you run `jupytext --sync notebook.ipynb` or `jupytext --sync notebook.py`, the filename you pass is mostly just the entry point; Jupytext still checks the pair and updates the **outdated** side from the **newer** side. ([Jupytext][1])

That means: if you usually edit `notebook.py` and it is the newer file, `--sync` will push those inputs into the `.ipynb`. If the `.ipynb` was saved later and is now newer, `--sync` can go the other direction and update the `.py` instead. It does **not** do a content-aware or three-way merge of concurrent edits; the timestamp winner wins. That is the practical behavior implied by the docs’ timestamp-based sync model. ([Jupytext][1])

For simultaneous editing, the docs are pretty explicit: you **can** keep the notebook open in Jupyter while editing the text file, but you should **not let both editors save simultaneously**. They recommend disabling Jupyter autosave and reloading the notebook when switching back to Jupyter. If you forget to reload and then save in Jupyter after the text file changed, Jupyter will prompt you about which version to keep. ([Jupytext][2])

One subtlety that matters for your workflow: on reload/open, Jupytext uses the **newer file for inputs**, while **outputs come from the `.ipynb`** if it exists. So “py as source of inputs, ipynb as source of outputs” is the normal paired-notebook model. ([Jupytext][2])

For your setup, I would treat `--sync` as safe when there is a clear editor-of-record at any moment, but **not** as protection against true concurrent edits. If you want a **deterministic one-way refresh from `py` to `ipynb`**, use:

```bash
jupytext --update --to notebook notebook.py
```

The CLI docs describe that as updating the input cells in the `.ipynb` while preserving outputs and metadata, which is often a better fit than `--sync` for “I edited the script; now refresh the notebook.” ([Jupytext][1])

A practical rule set for your case:

* edit mainly in `py:percent`
* keep `.ipynb` for outputs/UI
* avoid simultaneous saves from both sides
* use `--sync` when you want “newer side wins”
* use `--update --to notebook notebook.py` when you want “script wins, refresh notebook” regardless of ambiguity in the paired state. ([Jupytext][1])

Also worth noting: Jupytext has continued refining timestamp behavior; for example, the changelog notes fixes around paired-notebook timestamps and `--sync` not touching timestamps when content is unchanged. ([Jupytext][3])

I can sketch a robust git/pre-commit workflow for `ipynb + py:percent` that avoids accidental reversals.

[1]: https://jupytext.readthedocs.io/en/latest/using-cli.html "Jupytext CLI — Jupytext  documentation"
[2]: https://jupytext.readthedocs.io/en/latest/paired-notebooks.html "Paired notebooks — Jupytext  documentation"
[3]: https://jupytext.readthedocs.io/en/latest/changelog.html "Jupytext ChangeLog — Jupytext  documentation"
