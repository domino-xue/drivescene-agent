# Minimal Demo

This demo runs the real DriveScene Plan-and-Execute runtime against three
embedded event-index rows. It does not need an API key, the full Argoverse 2
dataset, or generated review assets.

From the repository root:

```powershell
& 'D:\app\envs\agent\python.exe' scripts\run_demo.py
& 'D:\app\envs\agent\python.exe' scripts\run_demo.py --question '找 2 个有效急刹案例'
& 'D:\app\envs\agent\python.exe' -m streamlit run scripts\demo_app.py --server.headless true
```

The CLI prints the planned step, deterministic tool result, evaluation status,
and final answer. The Streamlit page shows the same result with a compact chat
view, plan table, metrics, and execution digest.
