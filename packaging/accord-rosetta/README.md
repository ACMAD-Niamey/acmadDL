# accord-rosetta (deprecated)

This distribution has been renamed to **[acmadDL](https://pypi.org/project/acmadDL/)**.

```bash
pip install acmadDL
```

```python
import acmaddl                      # new import name
ds = acmaddl.fetch("c3s/ecmwf", variable="precip", init="2025-02")
```

Installing `accord-rosetta` now installs `acmadDL` and nothing else. Existing
code that does `import rosetta` keeps working via a deprecated alias shipped
inside `acmadDL`, which emits a `DeprecationWarning`. That alias will be
removed in a future release — switch to `import acmaddl`.

Source: <https://github.com/ACMAD-Niamey/acmadDL>
