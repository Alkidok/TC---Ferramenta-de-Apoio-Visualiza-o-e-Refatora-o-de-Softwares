import ast
from matplotlib.font_manager import font_scalings
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Any
import re
from numpy import size

class AnalisadorCodigo:
    def __init__(self): # Inicializa analisador com estruturas básicas
        self.grafo = nx.DiGraph() # Grafo direcionado para dependências
        self.funcoes = {} # Armazena informações das funções
        self.variaveis_globais = set() # Conjunto de variáveis globais
        self.imports = set() # Conjunto de imports utilizados
        self.arquivo_analisado = "" # Nome do arquivo em análise
    
    def analisar_arquivo(self, caminho_arquivo: str) -> Dict[str, Any]: # Analisa arquivo 
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as arquivo: # Abre arquivo
                codigo = arquivo.read() # Lê código completo
            
            self.arquivo_analisado = caminho_arquivo # Registra arquivo
            arvore = ast.parse(codigo) # Faz parsing para AST
            visitante = VisitanteCodigo() # Cria visitante especializado
            visitante.visit(arvore) # Percorre árvore
            
            # Processa dados coletados
            self.funcoes = visitante.funcoes
            self.variaveis_globais = visitante.variaveis_globais
            self.imports = visitante.imports
            
            # Constrói grafo de dependências
            self._construir_grafo_dependencias()
            
            return {
                'funcoes': self.funcoes,
                'variaveis_globais': self.variaveis_globais,
                'imports': self.imports,
                'grafo': self.grafo
            }
            
        except Exception as e: # Captura erros de análise
            print(f"Erro ao analisar {caminho_arquivo}: {e}")
            return {}
    
    def _construir_grafo_dependencias(self) -> None: # Constrói grafo com base nos dados coletados
        # Adiciona nós para funções
        for nome_funcao, info in self.funcoes.items():
            self.grafo.add_node(nome_funcao, tipo='funcao', **info) # Adiciona nó função
        
        # Adiciona nós para variáveis globais
        for var_global in self.variaveis_globais:
            self.grafo.add_node(var_global, tipo='variavel_global') # Adiciona nó variável
        
        # Adiciona nós para imports
        for import_item in self.imports:
            self.grafo.add_node(import_item, tipo='import') # Adiciona nó import
        
        # Adiciona arestas de chamada entre funções
        for nome_funcao, info in self.funcoes.items():
            for funcao_chamada in info.get('funcoes_chamadas', []):
                if funcao_chamada in self.funcoes: # Verifica se função existe
                    self.grafo.add_edge(nome_funcao, funcao_chamada, tipo='chamada_funcao') # Aresta de chamada
            
            # Adiciona arestas para variáveis globais usadas
            for var_usada in info.get('variaveis_globais_usadas', []):
                if var_usada in self.variaveis_globais: # Verifica se variável existe
                    self.grafo.add_edge(nome_funcao, var_usada, tipo='usa_variavel') # Aresta de uso
            
            # Adiciona arestas para imports usados
            for import_usado in info.get('imports_usados', []):
                if import_usado in self.imports: # Verifica se import existe
                    self.grafo.add_edge(nome_funcao, import_usado, tipo='usa_import') # Aresta de uso

    #Formata o label para quebrar linhas longas
    def _formatar_label(self, node: str) -> str: 
        if len(node) > 20:
            # Quebra o texto a cada 15 caracteres ou no último underscore
            parts = []
            current = node
            while len(current) > 20:
                # Tenta quebrar no último underscore
                underscore_pos = current[:20].rfind('_')
                if underscore_pos > 0:
                    parts.append(current[:underscore_pos])
                    current = current[underscore_pos + 1:]
                else:
                    parts.append(current[:20])
                    current = current[20:]
            parts.append(current)
            return '\n'.join(parts)
        return node

    def _calcular_layout_organizado(self) -> Dict[Any, List[float]]:
        pos = {}
        
        # Separa nós por tipo
        funcoes = [node for node in self.grafo.nodes() if self.grafo.nodes[node].get('tipo') == 'funcao']
        variaveis = [node for node in self.grafo.nodes() if self.grafo.nodes[node].get('tipo') == 'variavel_global']
        imports = [node for node in self.grafo.nodes() if self.grafo.nodes[node].get('tipo') == 'import']
        
        # Posiciona em camadas verticais com mais espaço
        camada_x = 0  # Import na esquerda
        for i, import_node in enumerate(imports):
            pos[import_node] = [camada_x, i - len(imports)/2]
        
        camada_x = 3  # Funções no meio
        for i, funcao in enumerate(funcoes):
            pos[funcao] = [camada_x, i - len(funcoes)/2]
        
        camada_x = 6  # Variáveis na direita
        for i, variavel in enumerate(variaveis):
            pos[variavel] = [camada_x, i - len(variaveis)/2]
        
        # Aplica um layout de força para espalhar os nós e evitar sobreposição
        pos = nx.spring_layout(self.grafo, pos=pos, k=2, iterations=50)
        
        return pos

    def gerar_visualizacao_organizada(self, arquivo_saida: str = 'Grafo-Test') -> None:
        if len(self.grafo.nodes()) == 0:
            print("Nenhuma dependência encontrada para visualizar.")
            return
        
        plt.figure(figsize=(16, 14))  # Tamanho da figura
        
        # Layout organizado por camadas (funções, variáveis, imports)
        pos = self._calcular_layout_organizado()
        
        # Cores por tipo de nó
        cores = []
        for node in self.grafo.nodes():
            tipo = self.grafo.nodes[node].get('tipo', 'desconhecido')
            if tipo == 'funcao': 
                cores.append('lightblue')
            elif tipo == 'variavel_global': 
                cores.append('lightgreen')
            elif tipo == 'import': 
                cores.append('lightcoral')
            else: 
                cores.append('gray')
        
        # Tamanhos baseados no grau de conexão
        graus = dict(self.grafo.degree())
        tamanhos = [400 + graus[node] * 80 for node in self.grafo.nodes()]  # Tamanho base
        
        # Desenha nós
        nx.draw_networkx_nodes(self.grafo, pos, node_color=cores, node_size=tamanhos, alpha=0.9)
        
        # Desenha arestas com cores por tipo
        arestas_chamada = [(u, v) for u, v, d in self.grafo.edges(data=True) if d.get('tipo') == 'chamada_funcao']
        arestas_variavel = [(u, v) for u, v, d in self.grafo.edges(data=True) if d.get('tipo') == 'usa_variavel']
        arestas_import = [(u, v) for u, v, d in self.grafo.edges(data=True) if d.get('tipo') == 'usa_import']
        
        nx.draw_networkx_edges(self.grafo, pos, edgelist=arestas_chamada, edge_color='blue', 
                              arrows=True, arrowsize=25, alpha=0.7, width=1.5)
        nx.draw_networkx_edges(self.grafo, pos, edgelist=arestas_variavel, edge_color='green', 
                              arrows=True, arrowsize=20, alpha=0.6, width=1.2)
        nx.draw_networkx_edges(self.grafo, pos, edgelist=arestas_import, edge_color='red', 
                              arrows=True, arrowsize=20, alpha=0.6, width=1.2)
        
        # Labels com melhor formatação
        labels = {node: self._formatar_label(node) for node in self.grafo.nodes()}
        nx.draw_networkx_labels(self.grafo, pos, labels, font_size=9, font_weight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                       edgecolor="gray", alpha=0.8))
        
        # Legenda
        self._adicionar_legenda()
        
        plt.title(f"Grafo de Dependências\n{self.arquivo_analisado}", #
                 fontsize=16, pad=25, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{arquivo_saida}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Visualização salva como: {arquivo_saida}.png")
    
    def _adicionar_legenda(self) -> None:
        from matplotlib.patches import Patch
        elementos_legenda = [
            Patch(facecolor='lightblue', label='Funções'),
            Patch(facecolor='lightgreen', label='Variáveis Globais'),
            Patch(facecolor='lightcoral', label='Imports'),
            Patch(facecolor='white', edgecolor='blue', label='Chamada de Função'),
            Patch(facecolor='white', edgecolor='green', label='Uso de Variável'),
            Patch(facecolor='white', edgecolor='red', label='Uso de Import')
        ]
        plt.legend(handles=elementos_legenda, loc='upper left', bbox_to_anchor=(0, 1))
    
    def gerar_relatorio_detalhado(self) -> None:
        print("="*70)
        print("RELATÓRIO DE ANÁLISE")
        print("="*70)
        
        # Estatísticas gerais
        print(f"\n📊 ESTATÍSTICAS GERAIS:")
        print(f"   Funções encontradas: {len(self.funcoes)}")
        print(f"   Variáveis globais: {len(self.variaveis_globais)}")
        print(f"   Imports utilizados: {len(self.imports)}")
        print(f"   Total de dependências: {len(self.grafo.edges())}")
        
        # Funções mais importantes (mais conectadas)
        if self.grafo.nodes():
            graus = dict(self.grafo.degree())
            funcoes_importantes = sorted([(n, d) for n, d in graus.items() 
                                        if self.grafo.nodes[n].get('tipo') == 'funcao'], 
                                       key=lambda x: x[1], reverse=True)[:5]
            
            print(f"\n🔗 FUNÇÕES MAIS CONECTADAS:")
            for funcao, conexoes in funcoes_importantes:
                print(f"   {funcao}: {conexoes} conexões")
        
        # Análise de dependências por função
        print(f"\n📋 ANÁLISE DETALHADA POR FUNÇÃO:")
        for nome_funcao, info in self.funcoes.items():
            print(f"\n   🎯 {nome_funcao}:")
            print(f"      📍 Linha: {info.get('linha', 'N/A')}")
            
            if info.get('funcoes_chamadas'):
                print(f"      🔄 Chama: {', '.join(info['funcoes_chamadas'])}")
            
            if info.get('variaveis_globais_usadas'):
                print(f"      💾 Usa variáveis: {', '.join(info['variaveis_globais_usadas'])}")
            
            if info.get('imports_usados'):
                print(f"      📦 Usa imports: {', '.join(info['imports_usados'])}")

# Visita nós do AST para coletar informações
class VisitanteCodigo(ast.NodeVisitor):

    # Inicializa estruturas de dados
    def __init__(self):
        self.funcoes = {}
        self.variaveis_globais = set()
        self.imports = set()
        self.funcao_atual = None
        self.variaveis_locais = set()

    # Visita definições de função
    def visit_FunctionDef(self, node): 
        info_funcao = {
            'nome': node.name,
            'linha': node.lineno,
            'funcoes_chamadas': set(),
            'variaveis_globais_usadas': set(),
            'imports_usados': set(),
            'variaveis_locais': set(),  # captura nomes de vars locais
            'variaveis_locais_detalhadas': [] 
        }
        
        self.funcao_atual = node.name
        self.variaveis_locais = set()
        self.funcoes[node.name] = info_funcao
        
        # Analisa argumentos da função
        for arg in node.args.args:
            self.variaveis_locais.add(arg.arg)
            info_funcao['variaveis_locais'].add(arg.arg)
            info_funcao['variaveis_locais_detalhadas'].append(f"arg: {arg.arg}")
        
        self.generic_visit(node)
        self.funcao_atual = None
    
    def visit_Assign(self, node):
        if self.funcao_atual:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variaveis_locais.add(target.id)
                    self.funcoes[self.funcao_atual]['variaveis_locais'].add(target.id)
                    self.funcoes[self.funcao_atual]['variaveis_locais_detalhadas'].append(f"var: {target.id}")
        else:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variaveis_globais.add(target.id)
        
        self.generic_visit(node)
        self.funcao_atual = None # Limpa função atual
    
    def visit_Assign(self, node): # Visita atribuições
        if self.funcao_atual: # Se está dentro de função
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variaveis_locais.add(target.id) # Adiciona como variável local
        else: # Se é global
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variaveis_globais.add(target.id) # Adiciona como variável global
        
        self.generic_visit(node) # Continua análise
    
    def visit_Name(self, node): # Visita nomes (variáveis, funções)
        if isinstance(node.ctx, ast.Load): # Se é uso (leitura)
            if self.funcao_atual and node.id not in self.variaveis_locais:
                # É variável global ou função
                if node.id in self.variaveis_globais:
                    self.funcoes[self.funcao_atual]['variaveis_globais_usadas'].add(node.id)
                elif node.id in self.funcoes and node.id != self.funcao_atual:
                    self.funcoes[self.funcao_atual]['funcoes_chamadas'].add(node.id)
        
        self.generic_visit(node) # Continua análise
    
    def visit_Call(self, node): # Visita chamadas de função
        if isinstance(node.func, ast.Name): # Chamada direta
            funcao_chamada = node.func.id
            if self.funcao_atual and funcao_chamada in self.funcoes and funcao_chamada != self.funcao_atual:
                self.funcoes[self.funcao_atual]['funcoes_chamadas'].add(funcao_chamada)
        elif isinstance(node.func, ast.Attribute): # Chamada de método
            if isinstance(node.func.value, ast.Name):
                modulo = node.func.value.id
                if self.funcao_atual and modulo in self.imports:
                    self.funcoes[self.funcao_atual]['imports_usados'].add(f"{modulo}.{node.func.attr}")
        
        self.generic_visit(node) # Continua análise
    
    def visit_Import(self, node): # Visita imports simples
        for alias in node.names:
            self.imports.add(alias.name) # Adiciona import
            if alias.asname:
                self.variaveis_globais.add(alias.asname) # Adiciona alias como variável global
    
    def visit_ImportFrom(self, node): # Visita imports from
        for alias in node.names:
            nome_completo = f"{node.module}.{alias.name}" if node.module else alias.name
            self.imports.add(nome_completo) # Adiciona import completo
            self.variaveis_globais.add(alias.asname or alias.name) # Adiciona como variável global

def main(): # Função principal
    import sys # Import para argumentos
    
    if len(sys.argv) > 1: # Verifica se há argumento
        arquivo = sys.argv[1] # Pega primeiro argumento
    else:
        arquivo = "testeplus.py" # Arquivo padrão
    
    print(f"🔍 Analisando arquivo: {arquivo}") # Feedback
    analisador = AnalisadorCodigo() # Cria analisador
    resultado = analisador.analisar_arquivo(arquivo) # Executa análise
    
    if resultado: # Se análise foi bem sucedida
        analisador.gerar_visualizacao_organizada('Grafo-Test') # Gera visualização
        analisador.gerar_relatorio_detalhado() # Gera relatório
    else:
        print("❌ Falha na análise do arquivo.") # Mensagem de erro

if __name__ == "__main__": # Ponto de entrada
    main() # Executa análise