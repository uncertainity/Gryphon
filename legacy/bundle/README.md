# Gryphon dashboard bundle

Generate the self-contained implementation and simulation dashboard from the
project root:

```bash
python bundle/generate_dashboard.py
```

The generated file is `bundle/output/dashboard_gryphon.html`. It has no
external web dependencies and can be opened directly in a browser.

The generator reads the current Python configuration and selects the newest
compatible base-game, Hold-and-Spin, and full-game NPZ archives. Missing
simulation types are shown clearly until their result files exist.
