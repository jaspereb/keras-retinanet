def get_d_layers():
    layers = []
    with open('/home/jasper/git/CEIG/keras-retinanet/keras_retinanet/models/depthLayers.txt', 'r') as f:
        cont = f.readlines()
        for i in range(0, len(cont)):
            layers.append(cont[i].rstrip())

    return layers


def get_rgb_layers():
    layers = []
    with open('/home/jasper/git/CEIG/keras-retinanet/keras_retinanet/models/rgbLayers.txt', 'r') as f:
        cont = f.readlines()
        for i in range(0, len(cont)):
            layers.append(cont[i].rstrip())

    return layers
