##-----------------------------------------------------------------------------
##  Importações
##-----------------------------------------------------------------------------
import numpy as np


##-----------------------------------------------------------------------------
##  Função
##-----------------------------------------------------------------------------
def normalize(image, x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil,
			  radpixels, angulardiv):
	"""
	Descrição:
		Normaliza a região da iris ao deformar a região circular em um bloco retangular de dimensões constantes.

	Entrada:
		image		- Imagem de iris de entrada.

		x_iris		- Coordenada x do circulo definindo o contorno da iris.
		y_iris		- Coordenada y do circulo definindo o contorno da iris.
		r_iris		- Raio do círculo definindo o contorno da iris.

		x_pupil		- Coordenada x do circulo definindo o contorno da pupila.
		y_pupil		- Coordenada y do circulo definindo o contorno da pupila.
		r_pupil		- Raio do círculo definindo o contorno da pupila.

		radpixels	- Resolução radial (dimensão vertical).
		angulardiv	- Resolução angular (dimensão horizontal).

	Saída:
		polar_array	- Forma normalizada da região da iris.
		polar_noise	- Forma normalizada da região de ruído.
	"""
	radiuspixels = radpixels + 2
	angledivisions = angulardiv-1

	r = np.arange(radiuspixels)
	theta = np.linspace(0, 2*np.pi, angledivisions+1)

	# Calcula o deslocamento do centro da pupila a partir do centro da iris.
	ox = x_pupil - x_iris
	oy = y_pupil - y_iris

	if ox <= 0:
		sgn = -1
	elif ox > 0:
		sgn = 1

	if ox==0 and oy > 0:
		sgn = 1

	a = np.ones(angledivisions+1) * (ox**2 + oy**2)

	# Necessário tratar a possibilidade de ox = 0.
	if ox == 0:
		phi = np.pi/2
	else:
		phi = np.arctan(oy/ox)

	b = sgn * np.cos(np.pi - phi - theta)

	# Calcula o raio ao redor da iris como uma função do ângulo.
	r = np.sqrt(a)*b + np.sqrt(a*b**2 - (a - r_iris**2))
	r = np.array([r - r_pupil])

	rmat = np.dot(np.ones([radiuspixels,1]), r)

	rmat = rmat * np.dot(np.ones([angledivisions+1,1]),
							np.array([np.linspace(0,1,radiuspixels)])).transpose()
	rmat = rmat + r_pupil

	# Exclui valores no contorno da borda entre iris e pupila, e a borda da esclera da iris...
	# ...pois estes podem não corresponder à areas na região da iris e resultar em ruído.
	# Não considere os aneis externos como dados da iris.
	rmat = rmat[1 : radiuspixels-1, :]

	# Calcula as coordenadas cartesianas de cada ponto de dados ao redor da região circular da iris.
	xcosmat = np.dot(np.ones([radiuspixels-2,1]), np.array([np.cos(theta)]))
	xsinmat = np.dot(np.ones([radiuspixels-2,1]), np.array([np.sin(theta)]))

	xo = rmat * xcosmat
	yo = rmat * xsinmat

	xo = x_pupil + xo
	xo = np.round(xo).astype(int)
	coords = np.where(xo >= image.shape[1])
	xo[coords] = image.shape[1] - 1
	coords = np.where(xo < 0)
	xo[coords] = 0
	
	yo = y_pupil - yo
	yo = np.round(yo).astype(int)
	coords = np.where(yo >= image.shape[0])
	yo[coords] = image.shape[0] - 1
	coords = np.where(yo < 0)
	yo[coords] = 0

	# Extrai valores de intensidade em uma representação polar normalizada através de interpolação.
	# x,y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
	# f = interpolate.interp2d(x, y, image, kind='linear')
	# polar_array = f(xo, yo)
	# polar_array = polar_array / 255

	polar_array = image[yo, xo]
	polar_array = polar_array / 255

	# Cria um arranjo de ruídos com a localização dos NaNs em polar_array
	polar_noise = np.zeros(polar_array.shape)
	coords = np.where(np.isnan(polar_array))
	polar_noise[coords] = 1

	# Livre-se dos pontos externos para escrever o padrão circular.
	image[yo, xo] = 255

	# Obtém as coordenadas dos pixels do círculo ao redor da iris.
	x,y = circlecoords([x_iris,y_iris], r_iris, image.shape)
	image[y,x] = 255

	# Obtém as coordenadas dos pixels do círculo ao redor da pupila.
	xp,yp = circlecoords([x_pupil,y_pupil], r_pupil, image.shape)
	image[yp,xp] = 255

	# Substitui os NaNs antes de realizar a codificação das features.
	coords = np.where((np.isnan(polar_array)))
	polar_array2 = polar_array
	polar_array2[coords] = 0.5
	avg = np.sum(polar_array2) / (polar_array.shape[0] * polar_array.shape[1])
	polar_array[coords] = avg

	return polar_array, polar_noise.astype(bool)


#------------------------------------------------------------------------------
def circlecoords(c, r, imgsize, nsides=600):
	"""
	Descrição:
		Encontra as coordenadas de um círculo baseado em seu centro e raio.

	Entrada:
		c   	- Centro do círculo.
		r  		- Raio do círculo.
		imgsize - Tamanho da imagem no qual o círculo será plotado.
		nsides 	- Número de lados do casco convexo contornando o círculo (o padrão é 600)

	Output:
		x,y     - Coordenadas do círculo.
	"""
	a = np.linspace(0, 2*np.pi, 2*nsides+1)
	xd = np.round(r * np.cos(a) + c[0])
	yd = np.round(r * np.sin(a) + c[1])

	# Se livra de valores maiores do que a imagem.
	xd2 = xd
	coords = np.where(xd >= imgsize[1])
	xd2[coords[0]] = imgsize[1] - 1
	coords = np.where(xd < 0)
	xd2[coords[0]] = 0

	yd2 = yd
	coords = np.where(yd >= imgsize[0])
	yd2[coords[0]] = imgsize[0] - 1
	coords = np.where(yd < 0)
	yd2[coords[0]] = 0

	x = np.round(xd2).astype(int)
	y = np.round(yd2).astype(int)
	return x,y