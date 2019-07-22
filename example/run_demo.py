import numpy as np
from PIL import Image

from densecrf_np import util
from densecrf_np.densecrf import DenseCRF

image_source = 'example/image.jpg'
# These unaries were obtained by running the FCN part of the model https://github.com/sadeepj/crfasrnn_keras
unary_logits_source = 'example/unary_logits.npy'
logits = np.load(unary_logits_source)

# Labels before the CRF
before_crf_labels = util.get_label_image(logits)

img = np.array(Image.open(image_source))
crf = DenseCRF(img, spatial_weight=1.5, bilateral_weight=5.0)
probs = crf.infer(logits, 5)

# Labels after the CRF
after_crf_labels = util.get_label_image(probs)

before_crf_labels.show()
after_crf_labels.show()
