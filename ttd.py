import os
import shutil

folder_path = "recon_frames"

if os.path.exists(folder_path) and os.path.isdir(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Delete file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Delete subfolder
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    print("All contents deleted successfully.")
else:
    print("Folder does not exist.")
