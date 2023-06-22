##-----------------------------------------------------------------------------
##  Importações
##-----------------------------------------------------------------------------
import numpy as np
from Functions.boundary import searchInnerBound, searchOuterBound
from Functions.line import findline, linecoords
import multiprocessing as mp


##-----------------------------------------------------------------------------
##  Função
##-----------------------------------------------------------------------------
def segment(eyeim, eyelashes_thres=80, use_multiprocess=True):
	"""
	Descrição:
		Segmenta a região da iris da imagem de olho. Indica a região com ruídos.

	Entrada:
		eyeim				- Imagem de olho
		eyelashes_thres   	- Limiar dos cílios.
		use_multiprocess   	- Usa multiprocessamento para rodar.

	Saída:
		ciriris		- Centraliza as coordenadas e o raio do contorno da iris.
		cirpupil	- Centraliza as coordenadas e o raio do contorno da pupila.
		imwithnoise	- Imagem original com a localização dos ruídos marcados como NaN.
	"""
	# Encontra o contorno da Iris através da equação integro-diferencial de Daugman.
	rowp, colp, rp = searchInnerBound(eyeim)
	row, col, r = searchOuterBound(eyeim, rowp, colp, rp)

	# Package pupil and iris boundaries.
	# Armazena os contornos da iris e da pupila.
	rowp = np.round(rowp).astype(int)
	colp = np.round(colp).astype(int)
	rp = np.round(rp).astype(int)
	row = np.round(row).astype(int)
	col = np.round(col).astype(int)
	r = np.round(r).astype(int)
	cirpupil = [rowp, colp, rp]
	ciriris = [row, col, r]

	# Encontra o topo e a base da pálpebra.
	imsz = eyeim.shape
	irl = np.round(row - r).astype(int)
	iru = np.round(row + r).astype(int)
	icl = np.round(col - r).astype(int)
	icu = np.round(col + r).astype(int)
	if irl < 0:
		irl = 0
	if icl < 0:
		icl = 0
	if iru >= imsz[0]:
		iru = imsz[0] - 1
	if icu >= imsz[1]:
		icu = imsz[1] - 1
	imageiris = eyeim[irl: iru + 1, icl: icu + 1]

	# Se usar multiprocessamento:
	if use_multiprocess:
		ret_top = mp.Manager().dict()
		ret_bot = mp.Manager().dict()
		p_top = mp.Process(
			target=findTopEyelid,
			args=(imsz, imageiris, irl, icl, rowp, rp, ret_top),
		)
		p_bot = mp.Process(
			target=findBottomEyelid,
			args=(imsz, imageiris, rowp, rp, irl, icl, ret_bot),
		)
		p_top.start()
		p_bot.start()
		p_top.join()
		p_bot.join()
		mask_top = ret_top[0]
		mask_bot = ret_bot[0]

	# Se não usar multiprocessamento:
	else:
		mask_top = findTopEyelid(imsz, imageiris, irl, icl, rowp, rp)
		mask_bot = findBottomEyelid(imsz, imageiris, rowp, rp, irl, icl)

	# Mascara a imagem do olho. Regiões de ruído são mascaradas por valores NaN.
	imwithnoise = eyeim.astype(float)
	imwithnoise = imwithnoise + mask_top + mask_bot

	# Se usar CASIA, elimina cílios através de limiarização
	ref = eyeim < eyelashes_thres
	coords = np.where(ref == 1)
	imwithnoise[coords] = np.nan

	return ciriris, cirpupil, imwithnoise


#------------------------------------------------------------------------------
def findTopEyelid(imsz, imageiris, irl, icl, rowp, rp, ret_top=None):
	"""
	Descrição:
		Máscara para o topo da região da pálpebra.

	Entrada:
		imsz		- Tamanho da imagem de olho.
		imageiris	- Imagem da região da iris.

		irl		    -
		icl		    -

		rowp		- Coordenada y do centro do circulo interior.
		rp		    - Raio do centro do circulo interior.

		ret_top		- Retorna o resultado quando usando multiprocessamento.

	Saída:
		mask    	- Mapeia os ruídos que serão mascarados com valores NaN.
	"""
	topeyelid = imageiris[0: rowp - irl - rp, :]
	lines = findline(topeyelid)
	mask = np.zeros(imsz, dtype=float)

	if lines.size > 0:
		xl, yl = linecoords(lines, topeyelid.shape)
		yl = np.round(yl + irl - 1).astype(int)
		xl = np.round(xl + icl - 1).astype(int)

		yla = np.max(yl)
		y2 = np.arange(yla)

		mask[yl, xl] = np.nan
		grid = np.meshgrid(y2, xl)
		mask[grid] = np.nan

	# Retorno
	if ret_top is not None:
		ret_top[0] = mask
	return mask


#------------------------------------------------------------------------------
def findBottomEyelid(imsz, imageiris, rowp, rp, irl, icl, ret_bot=None):
	"""
	Descrição:
		Máscara para a base da região da pálpebra.

	Input:
		imsz		- Imagem de olho.
		imageiris	- Imagem da região da iris.

		rowp		- Coordenada y do centro do circulo interior.
		rp		    - Raio do centro do circulo interior.

		irl		    -
		icl		    -

		ret_top		- Retorna o resultado quando usando multiprocessamento.

	Saída:
		mask    	- Mapeia os ruídos que serão mascarados com valores NaN.
	"""
	bottomeyelid = imageiris[rowp - irl + rp - 1 : imageiris.shape[0], :]
	lines = findline(bottomeyelid)
	mask = np.zeros(imsz, dtype=float)

	if lines.size > 0:
		xl, yl = linecoords(lines, bottomeyelid.shape)
		yl = np.round(yl + rowp + rp - 3).astype(int)
		xl = np.round(xl + icl - 2).astype(int)
		yla = np.min(yl)
		y2 = np.arange(yla-1, imsz[0])

		mask[yl, xl] = np.nan
		grid = np.meshgrid(y2, xl)
		mask[grid] = np.nan

	# Retorno
	if ret_bot is not None:
		ret_bot[0] = mask
	return mask