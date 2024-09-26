#%%
import labelbox as lb
import os

client = lb.Client(api_key=os.getenv("LABELBOX_API_KEY"))
dataset = client.create_dataset(name="clicking_dataset")
#%%
import json
file_path = "output_tasks/descriptions_labelbox.json_results.json"
with open(file_path, 'r') as file:
    rows = json.load(file)

task = dataset.upsert_data_rows(rows)
task.wait_till_done()
print(task.errors)
#%%
dataset.delete()