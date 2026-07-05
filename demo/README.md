# SMD Smart Monitoring Demo

Run all commands from the `UniTS` directory.

```powershell
C:\Users\Manh\.conda\envs\ptorch\python.exe scripts\smd\prepare_smd.py `
  --raw-dir "dataset\NetManAIOps OmniAnomaly master ServerMachineDataset" `
  --output-dir dataset\SMD

C:\Users\Manh\.conda\envs\ptorch\python.exe demo\generate_artifacts.py --backend baseline
C:\Users\Manh\.conda\envs\MyEnv\python.exe -m streamlit run demo\app.py
```

UniTS artifacts use the same dashboard contract:

```powershell
C:\Users\Manh\.conda\envs\ptorch\python.exe demo\generate_artifacts.py `
  --backend units --checkpoint checkpoints\smd_ptune_checkpoint.pth
```

The dashboard is offline after artifacts have been generated. Ground-truth
labels are only visible when evaluation mode is enabled.
