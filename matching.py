##-----------------------------------------------------------------------------
##  Importações
##-----------------------------------------------------------------------------
import numpy as np
from os import listdir
from fnmatch import filter
import scipy.io as sio
from multiprocessing import Pool, cpu_count
from itertools import repeat

import warnings
warnings.filterwarnings("ignore")


##-----------------------------------------------------------------------------
##  Função
##-----------------------------------------------------------------------------
def matching(template_extr, mask_extr, temp_dir, threshold=0.38):
	"""
	Descrição:
		Compara os templates extraídos com o banco de dados.

	Entrada:
		template_extr	- Template extraído.
		mask_extr		- Máscara extraída.
		threshold		- Limiar de distância.
		temp_dir		- Diretório contendo os templates.

	Saída:
		Lista de strings dos arquivos identificados.
		0 se não houver, -1 se não houver amostra registrada.
	"""
	#Pega o número de contas no banco de dados.
	n_files = len(filter(listdir(temp_dir), '*.mat'))
	if n_files == 0:
		return -1

	# Usa todos os números para calcular as distâncias de Hamming
	args = zip(
		sorted(listdir(temp_dir)),
		repeat(template_extr),
		repeat(mask_extr),
		repeat(temp_dir),
	)
	with Pool(processes=cpu_count()) as pools:
		result_list = pools.starmap(matchingPool, args)

	filenames = [result_list[i][0] for i in range(len(result_list))]
	hm_dists = np.array([result_list[i][1] for i in range(len(result_list))])

	# Remove os elementos não-numéricos (NaN)
	ind_valid = np.where(hm_dists>0)[0]
	hm_dists = hm_dists[ind_valid]
	filenames = [filenames[idx] for idx in ind_valid]

	# Limiariza e dá o ID resultante.
	ind_thres = np.where(hm_dists<=threshold)[0]

	# Retorno.
	if len(ind_thres)==0:
		return 0
	else:
		hm_dists = hm_dists[ind_thres]
		filenames = [filenames[idx] for idx in ind_thres]
		ind_sort = np.argsort(hm_dists)
		return [filenames[idx] for idx in ind_sort]


#------------------------------------------------------------------------------
def calHammingDist(template1, mask1, template2, mask2):
	"""
	Descrição:
		Calcula a distância de Hamming entre dois templates de Iris.

	Entrada:
		template1	- O 1º template.
		mask1		- A 1º máscara de ruídos.
		template2	- O 2º template.
		mask2		- A 2º máscara de ruídos.

	Saída:
		hd			- A distância de Hamming como uma razão.
	"""
	# Inicializa
	hd = np.nan

	# Troca os templates da esquerda e direita, usando o menor distância de Hamming.
	for shifts in range(-8,9):
		template1s = shiftbits(template1, shifts)
		mask1s = shiftbits(mask1, shifts)

		mask = np.logical_or(mask1s, mask2)
		nummaskbits = np.sum(mask==1)
		totalbits = template1s.size - nummaskbits

		C = np.logical_xor(template1s, template2)
		C = np.logical_and(C, np.logical_not(mask))
		bitsdiff = np.sum(C==1)

		if totalbits==0:
			hd = np.nan
		else:
			hd1 = bitsdiff / totalbits
			if hd1 < hd or np.isnan(hd):
				hd = hd1

	# Retorno
	return hd


#------------------------------------------------------------------------------
def shiftbits(template, noshifts):
	"""
	Descrição:
		Troca os padrões de iris bit a bit.

	Entrada:
		template	- O template a ser trocado.
		noshifts	- O número de operadores de troca. Valores positivos para direita, e negativos para esquerda.

	Saída:
		templatenew	- O template trocado.
	"""
	# Inicializa
	templatenew = np.zeros(template.shape)
	width = template.shape[1]
	s = 2 * np.abs(noshifts)
	p = width - s

	# Troca
	if noshifts == 0:
		templatenew = template

	elif noshifts < 0:
		x = np.arange(p)
		templatenew[:, x] = template[:, s + x]
		x = np.arange(p, width)
		templatenew[:, x] = template[:, x - p]

	else:
		x = np.arange(s, width)
		templatenew[:, x] = template[:, x - s]
		x = np.arange(s)
		templatenew[:, x] = template[:, p + x]

	# Retorno
	return templatenew


#------------------------------------------------------------------------------
def matchingPool(file_temp_name, template_extr, mask_extr, temp_dir):
	"""
	Descrição:
		Perform matching session within a Pool of parallel computation
		Realiza uma sessão de correspondência dentro de um pool de computação paralela.

	Entrada:
		file_temp_name	- Nome do arquivo com o template à ser examinado.
		template_extr	- Template extraído.
		mask_extr		- Máscara de ruídos extraída.

	Saída:
		hm_dist			- Distância de Hamming.
	"""
	# Carrega cada conta.
	data_template = sio.loadmat('%s%s'% (temp_dir, file_temp_name))
	template = data_template['template']
	mask = data_template['mask']

	# Calcula a distância de Hamming
	hm_dist = calHammingDist(template_extr, mask_extr, template, mask)
	return (file_temp_name, hm_dist)