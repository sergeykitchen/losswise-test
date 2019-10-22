import time
import random
import losswise
import numpy as np
from PIL import Image


losswise.set_api_key("IE11WZOSM")
max_iter = 60
session = losswise.Session(max_iter=max_iter,
    params={'max_iter': max_iter, 'dropout': 0.3, 'lr': 0.01, 'rnn_sizes': [256, 512]})
graph = session.graph('loss', kind='min')
for x in range(max_iter):
    train_loss = 1. / (0.1 + x + 0.1 * random.random())
    test_loss = 1.5 / (0.1 + x + 0.2 * random.random())
    graph.append(x, {'train_loss': train_loss, 'test_loss': test_loss})
    time.sleep(0.5)
    if x % 5 == 0:
        seq = session.image_sequence(x=x, name="Test")
        for img_id in range(5):
            pil_image = Image.open("./image.png")
            seq.append(pil_image,
                    metrics={'accuracy': 1},
                    image_id=str(img_id) + "_img")
session.done()