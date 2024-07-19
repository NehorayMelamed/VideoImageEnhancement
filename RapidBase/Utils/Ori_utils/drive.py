import pickle
import os.path
import zipfile
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

def setup_drive_service(SCOPES):
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(creds.refresh_token)
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret_169265982200-k7k5usl59vh84e4qad71uvn440h7mvaj.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)


# Download ZIP file function
def download_zip(file_id, destination):
    request = drive_service.files().get_media(fileId=file_id)
    with open(destination, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Downloaded {int(status.progress() * 100)}%")


# Extract ZIP file function
def extract_zip(zip_filepath, extract_to_dir):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to_dir)
    print(f"Extracted to {extract_to_dir}")




if __name__=="__main__":
    # Setup the Drive API client
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    drive_service = setup_drive_service(SCOPES)

    # Replace with your file ID and desired extraction directory
    file_id = "1S-lKFS82MAh16J9yUohY7fvq5fag6Bvf"
    zip_path = "zip_file.zip"
    extraction_directory = "/home/dudy/Nehoray/Ori"

    download_zip(file_id, zip_path)
    extract_zip(zip_path, extraction_directory)

    # Optionally, delete the ZIP file after extraction
    os.remove(zip_path)
