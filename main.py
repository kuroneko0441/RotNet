from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

from utils import save_output, RotNetDataGenerator, angle_error
from data.street_view import get_filenames

input_dir = sys.argv[1]
output_dir = sys.argv[2]

print('Loading model...')
print()
model_location = os.path.join('models', 'rotnet_street_view_resnet50.hdf5')
model = load_model(model_location, custom_objects={'angle_error': angle_error})
print()
print('Model loaded.')

filenames = [
    os.path.join(input_dir, f) for f in os.listdir(input_dir)
]

save_output(
    model,
    filenames,
    output_dir,
)
