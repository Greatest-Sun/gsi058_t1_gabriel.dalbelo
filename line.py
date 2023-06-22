##-----------------------------------------------------------------------------
##  Importações
##-----------------------------------------------------------------------------
import numpy as np
from scipy.ndimage import convolve
from skimage.transform import radon


##-----------------------------------------------------------------------------
##  Função
##-----------------------------------------------------------------------------
def findline(img):
    """
	Descrição:
		Encontra as linhas em uma imagem.
		A transformada linear de Hough e a detecção de bordas de Canny foram usadas.

	Entrada:
		img     - A imagem de entrada

	Saída:
		lines   - Parâmetros da detecção de linha em forma polar.
	"""
    # Pré-processamento
    I2, orient = canny(img, 2, 0, 1)
    I3 = adjgamma(I2, 1.9)
    I4 = nonmaxsup(I3, orient, 1.5)
    edgeimage = hysthresh(I4, 0.2, 0.15)

    # Transformada de Radon
    theta = np.arange(180)
    R = radon(edgeimage, theta, circle=False)
    sz = R.shape[0] // 2
    xp = np.arange(-sz, sz+1, 1)

    # Encontra a borda mais forte
    maxv = np.max(R)
    if maxv > 25:
        i = np.where(R.ravel() == maxv)
        i = i[0]
    else:
        return np.array([])

    R_vect = R.ravel()
    ind = np.argsort(-R_vect[i])
    u = i.shape[0]
    k = i[ind[0: u]]
    y, x = np.unravel_index(k, R.shape)
    t = -theta[x] * np.pi / 180
    r = xp[y]

    lines = np.vstack([np.cos(t), np.sin(t), -r]).transpose()
    cx = img.shape[1] / 2 - 1
    cy = img.shape[0] / 2 - 1
    lines[:, 2] = lines[:,2] - lines[:,0]*cx - lines[:,1]*cy
    return lines


# ------------------------------------------------------------------------------
def linecoords(lines, imsize):
    """
	Descrição:
		Encontra as coordenadas x- y- das posições ao longo da linha.

	Entrada:
		lines   - Parâmetros (forma polar) da linha.
		imsize  - Tamanho da imagem.

	Saída:
		x,y     - Coordenadas resultantes.
	"""
    xd = np.arange(imsize[1])
    yd = (-lines[0,2] - lines[0,0] * xd) / lines[0,1]

    coords = np.where(yd >= imsize[0])
    coords = coords[0]
    yd[coords] = imsize[0]-1
    coords = np.where(yd < 0)
    coords = coords[0]
    yd[coords] = 0

    x = xd
    y = yd
    return x, y


# ------------------------------------------------------------------------------
def canny(im, sigma, vert, horz):
    """
	Descrição:
		Detecção de bordas de Canny.

	Entrada:
		im      - A imagem de entrada.
		sigma   - O desvio padrão do filtro Gaussiano.
		vert    - Peso dos gradientes verticais.
		horz    - Peso dos gradientes horizontais.

	Saída:
		grad    - Força da borda (amplitude do gradiente).
		orient  - Orientação da imagem (0-180, positiva, anti-horário)
	"""

    def fspecial_gaussian(shape=(3, 3), sig=1):
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        f = np.exp(-(x * x + y * y) / (2 * sig * sig))
        f[f < np.finfo(f.dtype).eps * f.max()] = 0
        sum_f = f.sum()
        if sum_f != 0:
            f /= sum_f
        return f

    hsize = [6 * sigma + 1, 6 * sigma + 1]  # O tamanho do filtro
    gaussian = fspecial_gaussian(hsize, sigma)
    im = convolve(im, gaussian, mode='constant')  # Imagem suavizada
    rows, cols = im.shape

    h = np.concatenate([im[:, 1:cols], np.zeros([rows,1])], axis=1) - \
        np.concatenate([np.zeros([rows, 1]), im[:, 0: cols - 1]], axis=1)

    v = np.concatenate([im[1: rows, :], np.zeros([1, cols])], axis=0) - \
        np.concatenate([np.zeros([1, cols]), im[0: rows - 1, :]], axis=0)

    d11 = np.concatenate([im[1:rows, 1:cols], np.zeros([rows - 1, 1])], axis=1)
    d11 = np.concatenate([d11, np.zeros([1, cols])], axis=0)
    d12 = np.concatenate([np.zeros([rows-1, 1]), im[0:rows - 1, 0:cols - 1]], axis=1)
    d12 = np.concatenate([np.zeros([1, cols]), d12], axis=0)
    d1 = d11 - d12

    d21 = np.concatenate([im[0:rows - 1, 1:cols], np.zeros([rows - 1, 1])], axis=1)
    d21 = np.concatenate([np.zeros([1, cols]), d21], axis=0)
    d22 = np.concatenate([np.zeros([rows - 1, 1]), im[1:rows, 0:cols - 1]], axis=1)
    d22 = np.concatenate([d22, np.zeros([1, cols])], axis=0)
    d2 = d21 - d22

    X = (h + (d1 + d2) / 2) * vert
    Y = (v + (d1 - d2) / 2) * horz

    gradient = np.sqrt(X * X + Y * Y)  # Amplitude do gradiente.

    orient = np.arctan2(-Y, X)  # Ângulos (-pi até +pi)
    neg = orient < 0  # Mapeia os ângulos de 0 até pi
    orient = orient * ~neg + (orient + np.pi) * neg
    orient = orient * 180 / np.pi  # Converte para degraus

    return gradient, orient


# --------------------------------------------------------------------------------
def adjgamma(im, g):
    """
	Descrição
		Ajusta o gamma da imagem.

	Entrada:
		im      - A entrada da imagem.
		g       - Valor gamma da imagem.
				  Intervalo (0, 1] melhora o contraste da região clara.
				  Range (1, inf) melhora o contraste da região escura.

	Saída:
		newim   - A imagem ajustada.
	"""
    newim = im
    newim = newim - np.min(newim)
    newim = newim / np.max(newim)
    newim = newim ** (1 / g)  # Aplica a função gamma.
    return newim


# ------------------------------------------------------------------------------
def nonmaxsup(in_img, orient, radius):
    """
    Descrição:
        Efetua supressão não-máxima em uma imagem usando uma imagem de orientação.

    Entrada:
        in_img  - A imagem de entrada.
        orient  - Imagem contendo os ângulos de orientação normais da feature.
        radius  - Distância a ser olhada em cada lado de cada pixel quando determinando se ele é um máximo local ou não  (1.2 - 1.5)
    Saída:
        im_out  - A imagem suprimida.
    """
    # Pré-aloca memória para a imagem de saída, aumentando a velocidade. 
    rows, cols = in_img.shape
    im_out = np.zeros([rows, cols])
    iradius = np.ceil(radius).astype(int)

    # Pré-calcula os endereços de x e y referentes ao pixel central para cada ângulo de orientação.
    angle = np.arange(181) * np.pi / 180  # Angulos em incrementos de 1 degrau (em radianos).
    xoff = radius * np.cos(angle)  # endereços x e y dos pontos em raio e ângulo especificos. 
    yoff = radius * np.sin(angle)  # para cada posição de referência.

    hfrac = xoff - np.floor(xoff)  # Endereço fracional de xoff relativo à localização do inteiro.
    vfrac = yoff - np.floor(yoff)  # Endereço fracional de yoff relativo à localização do inteiro.


    orient = np.fix(orient)

    # Agora executa através da interpolação dos valores de cinza da imagem em cada lado do centro do pixel usado para supressão de não-maximos.
    col, row = np.meshgrid(np.arange(iradius, cols - iradius),
                           np.arange(iradius, rows - iradius))

    # Índice em arranjos pré-computados. 
    ori = orient[row, col].astype(int)

    # Localização de x e y em cada lado do ponto em questão.
    x = col + xoff[ori]
    y = row - yoff[ori]

    # Obtém as localizações em integer dos pixels que circulam a localização x,y
    fx = np.floor(x).astype(int)
    cx = np.ceil(x).astype(int)
    fy = np.floor(y).astype(int)
    cy = np.ceil(y).astype(int)

    # Valor nas localizações em integer dos pixels.
    tl = in_img[fy, fx]  # superior esquerdo
    tr = in_img[fy, cx]  # superior direito
    bl = in_img[cy, fx]  # inferior esquerdo
    br = in_img[cy, cx]  # inferior direito

    # Interpolação bi-linear para estimar os valores em x,y.
    upperavg = tl + hfrac[ori] * (tr - tl)
    loweravg = bl + hfrac[ori] * (br - bl)
    v1 = upperavg + vfrac[ori] * (loweravg - upperavg)

    # Checa o valor do outro lado.
    map_candidate_region = in_img[row, col] > v1

    x = col - xoff[ori]
    y = row + yoff[ori]

    fx = np.floor(x).astype(int)
    cx = np.ceil(x).astype(int)
    fy = np.floor(y).astype(int)
    cy = np.ceil(y).astype(int)

    tl = in_img[fy, fx]
    tr = in_img[fy, cx]
    bl = in_img[cy, fx]
    br = in_img[cy, cx]

    upperavg = tl + hfrac[ori] * (tr - tl)
    loweravg = bl + hfrac[ori] * (br - bl)
    v2 = upperavg + vfrac[ori] * (loweravg - upperavg)

    # Maximo local
    map_active = in_img[row, col] > v2
    map_active = map_active * map_candidate_region
    im_out[row, col] = in_img[row, col] * map_active

    return im_out

# ------------------------------------------------------------------------------
def hysthresh(im, T1, T2):
    """
	Descrição:
		Limiar de histerese.

	Entrada:
		im  - Imagem de entrada.
		T1  - O valor de limiar superior.
		T2  - O valor de limiar inferior.

	Output:
		bw  - A imagem binarizada.
	"""
    # Pré-computa alguns valores por velocidade e conveniência.
    rows, cols = im.shape
    rc = rows * cols
    rcmr = rc - rows
    rp1 = rows + 1

    bw = im.ravel()  # Transforma uma imagem em um vetor de colunas.
    pix = np.where(bw > T1) # Encontra os índices de todos os pixels com valor > T1
    pix = pix[0]
    npix = pix.size         # Encontra o número de pixels com valor > T1

    # Cria um array de pilha (que nunca deve transbordar).
    stack = np.zeros(rows * cols)
    stack[0:npix] = pix         # Coloca todos os pontos de borda em uma pilha.
    stp = npix      # Define o ponteiro da pilha.
    for k in range(npix):
        bw[pix[k]] = -1         # Marca pontos como bordas.

    # Pré-computa um array, 0, se os valores de endereço dos indíces corresponderem aos
    # oito pixels ao redor de qualquer ponto. Note que a imagem foi transformada em
    # um vetor de coluna, então se redefinirmos ela de volta á um quadrado os índices
    # circulando um pixel com índice, n, serão:
    #              n-linhas-1   n-1   n+linhas-1
    #
    #               n-linhas     n     n+linhas
    #
    #              n-linhas+1   n+1   n+linhas+1

    O = np.array([-1, 1, -rows - 1, -rows, -rows + 1, rows - 1, rows, rows + 1])

    while stp != 0:  # Enquanto a pilha não estiver vazia
        v = int(stack[stp-1])  # Retira o próximo índice da pilha
        stp -= 1

        if rp1 < v < rcmr:  # Previne a geração de índices ilegais
            # Agora olhe os pixels ao redor para ver se eles deveriam ser colocados na
            # pilha para também serem processados
            index = O + v  # Calcula os indices dos pontos ao redor desse pixel.
            for l in range(8):
                ind = index[l]
                if bw[ind] > T2:  # se valor > T2,
                    stp += 1  # coloca o índice na pilha.
                    stack[stp-1] = ind
                    bw[ind] = -1  # marca esse como um ponto de borda. 

    bw = (bw == -1)  # Finalmente, zera tudo aquilo que não é uma borda.
    bw = np.reshape(bw, [rows, cols])  # Remodela a imagem.
    return bw
