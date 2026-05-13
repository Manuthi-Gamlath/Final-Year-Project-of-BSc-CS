import importlib.metadata

for dist in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
    print(f"{dist.metadata['Name']}=={dist.version}")

