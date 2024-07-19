import zipfile
import os
import subprocess
import drive
# Step 3 & 4: Navigate to SystemBuilding and run SystemBuildingForUser.py
def run_system_building_for_user(base_directory):
    os.chdir(os.path.join(base_directory, 'SystemBuilding'))  # navigate to the directory
    subprocess.run(['python', 'SystemBuildingForUser.py'])

# Step 5: Run the main script
def run_main_script(base_directory):
    os.chdir(base_directory)  # navigate back to the base directory
    subprocess.run(['python', 'gui_video_processing_via_tkniret_3.py'])

# Execute the functions in sequence
def main(base_directory):
    # Setup the Drive API client
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    drive_service = drive.setup_drive_service(SCOPES)

    # Replace with your file ID and desired extraction directory
    file_id = "1S-lKFS82MAh16J9yUohY7fvq5fag6Bvf"
    zip_path = "zip_file.zip"
    extraction_directory = "/home/dudy/Nehoray/Ori"

    drive.download_zip(file_id, zip_path)
    drive.extract_zip(zip_path, extraction_directory)

    # Optionally, delete the ZIP file after extraction
    os.remove(zip_path)


    run_system_building_for_user(base_directory)
    run_main_script(base_directory)

# Run the main function
if __name__ == "__main__":
    base_dir = input("Enter the path to the base directory: ")
    main(base_dir)
