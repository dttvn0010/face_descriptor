import os
from utils.gdrive_util import download_file_from_google_drive
from utils.zip_util import extractZip


print('Downloading face_db.zip ...')
download_file_from_google_drive('1giDTs_j-eb6oIyCqgp641GYZf8On0GCt', 'face_db.zip')

print('Extracting zip file ...')
extractZip('face_db.zip', '.')

os.remove('face_db.zip')