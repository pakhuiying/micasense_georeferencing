import numpy as np

def cosine_similarity(v1,v2):
    """ similarity between 2 vectors"""
    return np.dot(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2))

def correlation_coefficient(im1,im2):
    """
    :param im1 (np.array): one channel image
    :param im2 (np.array): one channel image, with a black background
    Subtracting the means make the resulting vectors insensitive to image brightness, 
    and dividing by the vector norms makes them insensitive to image contrast
    """
    assert im1.shape == im2.shape, "im1 and im2 must have the same shape"
    im1_prime = im1[im2>0]
    im2_prime = im2[im2>0]
    assert im1_prime.shape == im2_prime.shape

    im1_norm = im1_prime.flatten() - im1_prime.mean()
    im1_norm = im1_norm/np.linalg.norm(im1_norm)

    im2_norm = im2_prime.flatten() - im2_prime.mean()
    im2_norm = im2_norm/np.linalg.norm(im2_norm)

    return np.dot(im1_norm,im2_norm)
