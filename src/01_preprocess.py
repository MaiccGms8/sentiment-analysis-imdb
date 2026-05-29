import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import limpar_texto

if __name__ == '__main__':
    exemplos = [
        "Esse filme foi incrível, amei cada cena!",
        "Péssimo filme, total perda de tempo.",
        "Não gostei nada do final, decepcionante.",
    ]
    for frase in exemplos:
        print(f"Original:  {frase}")
        print(f"Limpa:     {limpar_texto(frase)}")
        print()
