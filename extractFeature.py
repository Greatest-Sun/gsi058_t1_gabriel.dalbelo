##-----------------------------------------------------------------------------
##  Importações
##-----------------------------------------------------------------------------
from cv2 import imread

from Functions.segment import segment
from Functions.normalize import normalize
from Functions.encode import encode


##-----------------------------------------------------------------------------
##  Parâmetros para a extração de features
##	(Os parâmetros a seguir são padrões do banco de dados CASIA1)
##-----------------------------------------------------------------------------
# Parâmetros da segmentação.
eyelashes_thres = 80

# Parâmetros para a normalização.
radial_res = 20
angular_res = 240

# Parâmetros para a codificação das features.
minWaveLength = 18
mult = 1
sigmaOnf = 0.5


##-----------------------------------------------------------------------------
##  Função
##-----------------------------------------------------------------------------
def extractFeature(im_filename, eyelashes_thres=80, use_multiprocess=True):
	"""
	Descrição:
		Extrai as features de uma imagem de iris.

	Entrada:
		im_filename			- A imagem de iris para entrada.
		use_multiprocess	- Usa multiprocessamento para rodar.

	Saída:
		template			- O template extraído
		mask				- A máscara extraída
		im_filename			- A imagem de iris da entrada.
	"""
	# 
	im = imread(im_filename, 0)
	ciriris, cirpupil, imwithnoise = segment(im, eyelashes_thres, use_multiprocess)

	# Realiza a normalização.
	polar_array, noise_array = normalize(imwithnoise, ciriris[1], ciriris[0], ciriris[2],
										 cirpupil[1], cirpupil[0], cirpupil[2],
										 radial_res, angular_res)

	# Realiza a codificação.
	template, mask = encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf)

	# Retorno.
	return template, mask, im_filename