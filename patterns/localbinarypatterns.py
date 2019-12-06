# Importa as bibliotecas necessárias.
from skimage import feature
import numpy as np
import cv2
import matplotlib.pyplot as plt

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# Guarda o número de pontos e o raio.
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# Computa a representação LBP da imagem e
		# usa a representação para gerar um histograma de padrões.
		# LBP é uma função da biblioteca skimage.
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# cv2.imshow("LBP", lbp)
		# ax = plt.hist(lbp.ravel(), bins = 256)
		# plt.show()

		# cv2.waitKey(0)

		# Normaliza o histograma.
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# Retorna o histograma de LBP.
		return hist # Feature