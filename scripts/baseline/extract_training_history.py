"""Export local W&B epoch history without network access."""
import json
from pathlib import Path
import pandas as pd
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2
root=Path(__file__).resolve().parents[2]/'results/sccvae_baseline/fresh_20261021_214807'
out=root/'evaluation'
out.mkdir(exist_ok=True)
records=[]
for path in (root/'outs/ood_split_0_full/wandb').rglob('*.wandb'):
    store=DataStore(); store.open_for_scan(str(path))
    while True:
        raw=store.scan_data()
        if raw is None: break
        record=wandb_internal_pb2.Record(); record.ParseFromString(raw)
        if record.HasField('history'):
            row={item.key or '.'.join(item.nested_key):json.loads(item.value_json) for item in record.history.item}
            records.append(row)
history=pd.DataFrame(records)
history=history[[c for c in history if c.startswith('_') or c.endswith(' train') or c.endswith(' val')]]
history.to_csv(out/'training_validation_history.csv',index=False)
print('History records:',len(history))
print(history.tail(2).to_string(index=False))
