#%%
import os

BUCKET_NAME = 'clicking_dataset'  # specify your bucket name here

from dotenv import load_dotenv

load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS_FILE = "/Users/sarathmenon/clicking_service_account.json"

with open(GOOGLE_APPLICATION_CREDENTIALS_FILE) as f:
    credentials = f.read()
# %%
from label_studio_sdk.client import LabelStudio
LABEL_STUDIO_URL = 'http://localhost:8080'
API_KEY = 'ca457375ea8657a8b43acd99d5255a4200fbfcd7'

ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)

# %%

LABEL_STUDIO_URL = os.getenv('LABEL_STUDIO_URL')
API_KEY = os.getenv('API_KEY')

# Connect to the Label Studio API and check the connection
ls_client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=API_KEY)
projects_list = ls_client.projects.list()

# Get the project
project_id =1
project = ls_client.projects.get(project_id)
print(f"Project title: {project.title}")

# get the project tasks
tasks = ls_client.tasks.list(project=1)
# %%

PREFIX = 'gameplay_images'  # specify your prefix here
# assuming you service account key is stored in GOOGLE_APPLICATION_CREDENTIALS

storage = ls.import_storage.gcs.create(
    project=project_id,
    bucket=BUCKET_NAME,
    prefix=PREFIX,
    regex_filter='.*jpg',
    google_application_credentials=credentials,
    use_blob_urls=True,
    presign=True
)
# %%
