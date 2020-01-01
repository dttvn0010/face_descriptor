import os
from gdrive_util import download_file_from_google_drive
from zip_util import extractZip


print('Downloading model_weights.zip ...')
download_file_from_google_drive('1CcAOeuN130xSJOXbyVZgVaarGx8pONbh', 'model_weights.zip')

print('Extracting zip file ...')
extractZip('model_weights.zip', '.')

os.remove('model_weights.zip')