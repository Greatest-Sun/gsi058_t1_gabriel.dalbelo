##-----------------------------------------------------------------------------
##  Importações
##-----------------------------------------------------------------------------
import numpy as np
from scipy import signal


##-----------------------------------------------------------------------------
##  Função
##-----------------------------------------------------------------------------
def searchInnerBound(img):
    """
    Descrição:
        Busca pelo contorno interno da iris.

    Entrada:
        img		- A imagem de iris de entrada.

    Saída:
        inner_y	- Coordenada-y do círculo interior central.
        inner_x	- Coordenada-x do círculo interior central.
        inner_r	- Raio do círculo interno.
    """

    # Operador Integro-Diferencial grosseiro (precisão em nível de salto)
    Y = img.shape[0]
    X = img.shape[1]
    sect = X/4 		# Largura da margem externa para a qual a pesquisa é excluída.
    minrad = 10
    maxrad = sect*0.8
    jump = 4 		# Precisão da busca grosseira, em pixels.

    # Espaço de Hough (y,x,r)
    sz = np.array([np.floor((Y-2*sect)/jump),
                    np.floor((X-2*sect)/jump),
                    np.floor((maxrad-minrad)/jump)]).astype(int)

    # Resolução da integração circular
    integrationprecision = 1
    angs = np.arange(0, 2*np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(sz[1]),
                          np.arange(sz[0]),
                          np.arange(sz[2]))
    y = sect + y*jump
    x = sect + x*jump
    r = minrad + r*jump
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Derivada Parcial R do Espaço de Hough.
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # Blur .
    sm = 3 		# Tamanho da máscara de blur.
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = sect + y*jump
    inner_x = sect + x*jump
    inner_r = minrad + (r-1)*jump

    # Operador Integro-Diferencial fino (precisão em nível de pixel)
    integrationprecision = 0.1 		# Resolução da integração circular.
    angs = np.arange(0, 2*np.pi, integrationprecision)
    x, y, r = np.meshgrid(np.arange(jump*2),
                          np.arange(jump*2),
                          np.arange(jump*2))
    y = inner_y - jump + y
    x = inner_x - jump + x
    r = inner_r - jump + r
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # Derivada Parcial R do Espaço de Hough.
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # Bluring
    sm = 3 		# Tamanho da máscara de blurring
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")
    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    inner_y = inner_y - jump + y
    inner_x = inner_x - jump + x
    inner_r = inner_r - jump + r - 1

    return inner_y, inner_x, inner_r


#------------------------------------------------------------------------------
def searchOuterBound(img, inner_y, inner_x, inner_r):
    """
    Descrição:
        Busca pelo contorno externo da iris.

    Entrada:
        img		- A imagem de iris da entrada. 
        inner_y	- Coordenada-y do círculo interior central.
        inner_x	- Coordenada-x do círculo interior central.
        inner_r	- Raio do círculo interno.

    Saída:
        outer_y	- Coordenada-y do círculo externo central.
        outer_x	- Coordenada-x do círculo externo central.
        outer_r	- Raio do círculo externo.
    """
    # Deslocamento máximo 15# (Daugman 2004)
    maxdispl = np.round(inner_r*0.15).astype(int)

    # 0.1 - 0.8 (Daugman 2004)
    minrad = np.round(inner_r/0.8).astype(int)
    maxrad = np.round(inner_r/0.3).astype(int)

    # # Espaço de Hough (y,x,r)
    # hs = np.zeros([2*maxdispl, 2*maxdispl, maxrad-minrad])

    # Região de Integração, evitando cílios
    intreg = np.array([[2/6, 4/6], [8/6, 10/6]]) * np.pi

    # Resolução da integração circular.
    integrationprecision = 0.05
    angs = np.concatenate([np.arange(intreg[0,0], intreg[0,1], integrationprecision),
                            np.arange(intreg[1,0], intreg[1,1], integrationprecision)],
                            axis=0)
    x, y, r = np.meshgrid(np.arange(2*maxdispl),
                          np.arange(2*maxdispl),
                          np.arange(maxrad-minrad))
    y = inner_y - maxdispl + y
    x = inner_x - maxdispl + x
    r = minrad + r
    hs = ContourIntegralCircular(img, y, x, r, angs)

    # # Derivada Parcial R do Espaço de Hough.
    hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2]-1), 0, 0)]

    # Blur
    sm = 7 	# Tamanho da máscara de blurring
    hspdrs = signal.fftconvolve(hspdr, np.ones([sm,sm,sm]), mode="same")

    indmax = np.argmax(hspdrs.ravel())
    y,x,r = np.unravel_index(indmax, hspdrs.shape)

    outer_y = inner_y - maxdispl + y + 1
    outer_x = inner_x - maxdispl + x + 1
    outer_r = minrad + r - 1

    return outer_y, outer_x, outer_r


#------------------------------------------------------------------------------
def ContourIntegralCircular(imagen, y_0, x_0, r, angs):
    """
    Descrição:
        Efetua o contorno (circular) integral.
        Usa a aproximação discreta de Rie-mann.

    Entrada:
        imagen  - A imagem de iris de entrada.
        y_0     - A coordenada-y do círculo central.
        x_0     - A coordenada-x do círculo central.
        r       - O raio do círculo.
        angs    - A região do círculo considerando sentido horário 0-2pi.

    Saída:
        hs      - Resultado integral.
    """
    # Obtém y, x
    y = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    x = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
    for i in range(len(angs)):
        ang = angs[i]
        y[i, :, :, :] = np.round(y_0 - np.cos(ang) * r).astype(int)
        x[i, :, :, :] = np.round(x_0 + np.sin(ang) * r).astype(int)

    # Adapta y
    ind = np.where(y < 0)
    y[ind] = 0
    ind = np.where(y >= imagen.shape[0])
    y[ind] = imagen.shape[0] - 1

    # Adapta x
    ind = np.where(x < 0)
    x[ind] = 0
    ind = np.where(x >= imagen.shape[1])
    x[ind] = imagen.shape[1] - 1

    # Retorna
    hs = imagen[y, x]
    hs = np.sum(hs, axis=0)
    return hs.astype(float)
