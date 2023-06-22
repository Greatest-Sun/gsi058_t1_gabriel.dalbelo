##-----------------------------------------------------------------------------
##  Importações
##-----------------------------------------------------------------------------
import numpy as np


##-----------------------------------------------------------------------------
##  Função
##-----------------------------------------------------------------------------
def encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf):
	"""
	Descrição:
		Gera o template da iris e a máscara de ruídos partindo da região normalizada da iris.

	Entrada:
		polar_array		- Região da iris normalizada.
		noise_array		- Região de ruídos normalizada.
		minWaveLength	- Comprimento base de onda.
		mult			- Fator multiplicativo entre cada filtro.
		sigmaOnf		- Parâmetro de largura de banda.

	Saída:
		template		- O template biométrico da iris binarizada.
		mask			- A máscara de ruídos da iris binarizada.
	"""

	# Convoluciona a região normalizada com filtros de Gabor.
	filterbank = gaborconvolve(polar_array, minWaveLength, mult, sigmaOnf)

	length = polar_array.shape[1]
	template = np.zeros([polar_array.shape[0], 2 * length])
	h = np.arange(polar_array.shape[0])

	# Cria o template da iris.
	mask = np.zeros(template.shape)
	eleFilt = filterbank[:, :]

	# Quantização de fase.
	H1 = np.real(eleFilt) > 0
	H2 = np.imag(eleFilt) > 0

	# Se a amplitude é perto de zero então os dados de fase são inúteis, então marque na máscara de ruídos.
	H3 = np.abs(eleFilt) < 0.0001
	for i in range(length):
		ja = 2 * i

		# Constrói o template biométrico.
		template[:, ja] = H1[:, i]
		template[:, ja + 1] = H2[:, i]

		# Cria a máscara de ruídos.
		mask[:, ja] = noise_array[:, i] | H3[:, i]
		mask[:, ja + 1] = noise_array[:, i] | H3[:, i]

	# Retorno
	return template, mask


#------------------------------------------------------------------------------
def gaborconvolve(im, minWaveLength, mult, sigmaOnf):
	"""
	Descrição:
		Convoluciona cada linha da imagem com filtros logarítmicos 1D de Gabor.

	Entrada:
		im   			- A imagem que será convolucionada.
		minWaveLength   - Comprimento de onda do filtro base.
		mult   			- Fator multiplicativo entre cada filtro.
		sigmaOnf   		- Razão do desvio padrão Gaussiano descrevendo a função de transferência do filtro logarítmico de Gabor no domínio da frequência para o centro do filtro.

	Saída:
		filterbank		- O arranjo de células das coordenadas da convolução valorizada complexamente de resultsCircle.
	"""

	# Pré-alocado
	rows, ndata = im.shape					# Tamanho
	logGabor = np.zeros(ndata)				# Logarítimica de Gabor
	filterbank = np.zeros([rows, ndata], dtype=complex)

	# Valores de frequência 0 - 0.5
	radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
	radius[0] = 1

	# Inicializa o filtro de comprimento de onda.
	wavelength = minWaveLength

	# Calcula o componente de filtro radial. 
	fo = 1 / wavelength 		# Centraliza a frequência da filtragem
	logGabor[0 : int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigmaOnf)**2))
	logGabor[0] = 0

	# Para cada linha da imagem de entrada, convolucione.
	for r in range(rows):
		signal = im[r, 0:ndata]
		imagefft = np.fft.fft(signal)
		filterbank[r , :] = np.fft.ifft(imagefft * logGabor)

	# Retorno
	return filterbank