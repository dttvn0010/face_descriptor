import os
from utils.gdrive_util import download_file_from_google_drive
from utils.zip_util import extractZip


print('Downloading model_weights.zip ...')
download_file_from_google_drive('1SpEPINaw8cq8dk2EstOfsZ_OUh-x58D6', 'model_weights.zip')

print('Extracting zip file ...')
extractZip('model_weights.zip', './model_weights')

os.remove('model_weights.zip')