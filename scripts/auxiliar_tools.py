import json

def print_hello_world():
    """
    A função print_hello_world() devolve a string "Hello World". 
    """
    return "Hello World!"


def jprint(obj, hierarchical_levels=6):
    """
    A função jprint() tem como objetivo imprimir o json fornecido por uma chamada a uma API. 
    O formato .json é um formato para representação de dados na especificação "chave-valor" 
    - numa estrutura de dados similar a um dicionário da linguagem Python. 
    
    Os dados, tal como numa base de dados usual, podem ter uma estrutura relacional.
    É usual que estas relações possam ser representadas num formato hierárquico. 
    O formato json é assim compatível com esta representação relacional hierárquica, 
    sendo a navegação entre os diferentes níveis definida pelas chaves, 
    que correspondem aos nós da representação hierárquica em rede dos referidos dados.
    Esta função permite representar a estrutura hierárquica do json, através da "impressão" em 
    formato textual, utilizando tabulações por forma a alinhar as chaves que se encontram
    em cada nível hierárquico"    

    Args:
        obj (json): objeto json.
        hierarchical_levels (int): número de níveis hierárquicos a serem impressos.

    Returns:
        return: return the content of the json as a string with a scheme of line breaks and intendents to
        represent the hierarchical nature of data stored in the json file.
        
    Examples:
        Example usage of the function:
        >>> jprint(json_object)
        
    """
    
    text = json.dumps(obj, sort_keys=True, indent=hierarchical_levels)
    
    return text