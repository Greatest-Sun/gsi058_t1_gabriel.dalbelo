##-----------------------------------------------------------------------------
##  Importações
##-----------------------------------------------------------------------------
import argparse
from time import time

from Functions.extractFeature import extractFeature
from Functions.matching import matching


#------------------------------------------------------------------------------------
#	Conversão de argumentos. Cria as flags à serem identificadas via linha de comando
#------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--file", type=str,
                    help="Caminho até o arquivo que você quer verificar.")

parser.add_argument("--temp_dir", type=str, default="./templates/",
					help="Caminho até o diretório que contém os templates.")

parser.add_argument("--thres", type=float, default=0.38,
					help="Limiar para a compatibilidade.")

args = parser.parse_args()


##-----------------------------------------------------------------------------
##  Execução
##-----------------------------------------------------------------------------
# Extração das features passadas via linha de comando.
start = time()
print('>>> Iniciando verificação {}\n'.format(args.file))
template, mask, file = extractFeature(args.file)


# Função de verificação de compatibilidade.
result = matching(template, mask, args.temp_dir, args.thres)

if result == -1:
	print('>>> Nenhuma amostra registrada.')
elif result == 0:
	print('>>> Nenhuma amostra compatível.')
else:
	print('>>> {} amostras compatíveis (em ordem decrescente de confiabilidade):'.format(len(result)))
	for res in result:
		print("\t", res)


#Mensura o tempo total de execução.
end = time()
print('\n>>> Tempo de verificação: {} [s]\n'.format(end - start))