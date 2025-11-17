from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains 
import re

# ---- Constantes ----
contador = 0
contador_metacritic = 0

# ---- Arrays ----
dic_link_jogos = ["https://store.epicgames.com/pt-BR/"]
dic_nomes_jogos = []
dic_descricao_jogo = []
dic_preco_jogo = []
dic_fim_promoção_jogo = []
dic_capa_jogos = []
dic_genero_jogos = []
dic_nota_metacritic = []

# ---- Webdriver config ----

options = Options()
service = Service()
options = webdriver.EdgeOptions()
options.add_argument("--ignore-certificate-errors")

driver = webdriver.Edge(service=service, options=options)
driver.set_window_size(800, 600)
WebDriverWait(driver, 5)




# ---- Defs ----

def pegar_links():
    conteiner = driver.find_element(By.CLASS_NAME, "css-1vu10h2")
    links = conteiner.find_elements(By.CLASS_NAME, "css-g3jcms")

    for link in links:
        link_jogo = link.get_attribute("href")
        dic_link_jogos.append(link_jogo)

    return dic_link_jogos

def pagina_jogo(contador):
    driver = webdriver.Edge(service=service, options=options)
    driver.get(dic_link_jogos[contador])

    # ----Tela de conteudo inapropriado caso tenha ----

    span_conteudo_inapropriado = "Para continuar, forneça sua data de nascimento"
    try:
        elemento_span = driver.find_element(By.XPATH, f"//span[text()='{span_conteudo_inapropriado}']")
        print(elemento_span)
        # Dia
        conteudo_inapropriado_click_dia = driver.find_element(By.ID, "day_toggle")
        conteudo_inapropriado_click_dia.click()
        ActionChains(driver).send_keys(Keys.ARROW_UP).send_keys(Keys.ENTER).perform()
        WebDriverWait(driver, 2)
        print("OK")

        # Mês
        conteudo_inapropriado_click_mes = driver.find_element(By.ID, "month_toggle")
        conteudo_inapropriado_click_mes.click()
        ActionChains(driver).send_keys(Keys.ARROW_UP).send_keys(Keys.ENTER).perform()
        WebDriverWait(driver, 2)
        print("OK")
        
        # Ano
        conteudo_inapropriado_click_ano = driver.find_element(By.ID, "year_toggle")
        conteudo_inapropriado_click_ano.click()
        ActionChains(driver).send_keys(Keys.ARROW_UP).send_keys(Keys.ENTER).perform()
        WebDriverWait(driver, 2)
        print("OK")

        # Continuar
        conteudo_inapropriado_click_ano = driver.find_element(By.ID, "btn_age_continue")
        conteudo_inapropriado_click_ano.click()
        print("OK")

    except NoSuchElementException:
        print("OK")


    # ---- Nomes ----

    try:
        conteiner_titulo_jogo = driver.find_element(By.CLASS_NAME, "css-1mzagbj") 
        titulo_jogo = conteiner_titulo_jogo.text
        dic_nomes_jogos.append(titulo_jogo)

    except NoSuchElementException:
        print("Err Nome não encontrado!!!", dic_link_jogos[contador])
        dic_nomes_jogos.append("Err")

    # ---- Descrição ----
    try:
        conteiner_descricao_jogo = driver.find_element(By.CLASS_NAME, "css-1myreog")
        descricao_jogo = conteiner_descricao_jogo.text
        dic_descricao_jogo.append(descricao_jogo)

    except NoSuchElementException:
        print("Err Descrição não encontrado!!!", dic_link_jogos[contador])
        dic_descricao_jogo.append("Err")

    # ---- Preço ----

    try:
        elementos = driver.find_elements(By.CLASS_NAME, "css-15fg505")

        for elemento in elementos:

            texto_elemento = elemento.text
            
            if "R$" in texto_elemento:
                padrao_preco = r'R\$\s*\d[\d\.,]*'

                precos_encontrados = re.findall(padrao_preco, texto_elemento)
                dic_preco_jogo.append(precos_encontrados)

            elif "Gratuito" in texto_elemento:
                dic_preco_jogo.append(texto_elemento)

    except NoSuchElementException:
        dic_preco_jogo.append("Err")

    # ---- Fim da promoção ----

    try:
        conteiner_fim_promoção_jogo = driver.find_element(By.CLASS_NAME, "css-iqno47")
        fim_promoção_jogo = conteiner_fim_promoção_jogo.text
        dic_fim_promoção_jogo.append(fim_promoção_jogo)

    except NoSuchElementException:
        print("Err Fim da promoção não encontrado!!!", dic_link_jogos[contador])
        dic_fim_promoção_jogo.append("Err")

    # ---- Capa ----

    try:
        conteiner_capa_jogo = driver.find_element(By.CLASS_NAME, "css-7i770w")
        capa_jogo = conteiner_capa_jogo.get_attribute("src")
        dic_capa_jogos.append(capa_jogo)

    except NoSuchElementException:
        print("Err Capa não encontrado!!!", dic_link_jogos[contador])
        dic_capa_jogos.append("Err")

    # ---- Gêneros ----

    try:
        genero_li = driver.find_elements(By.XPATH, "//ul[@class='css-vs1xw0']/li")

        genero_texto_li = []

        for elemento_li in genero_li[:3]:
            texto_li = elemento_li.text
            
            genero_texto_li.append(texto_li)

        genero_texto_completo = ", ".join(genero_texto_li)

        dic_genero_jogos.append(genero_texto_completo)

    except NoSuchElementException:
        print("Err Gêneros não encontrado!!!", dic_link_jogos[contador])
        dic_genero_jogos.append("Err")

    # ---- return ---- 
    driver.quit()
    return contador, dic_nomes_jogos, dic_descricao_jogo, dic_preco_jogo, dic_fim_promoção_jogo, dic_capa_jogos, dic_genero_jogos

def metacritic(dic_nota_metacritic, dic_nomes_jogos, contador_metacritic):

    if dic_nomes_jogos[contador_metacritic] != "Err":

        # ---- Metacritic link ----

        metacritic_link = "https://www.metacritic.com/game"

        metacritic_nome_formatado = "-".join(dic_nomes_jogos[contador_metacritic].lower().split())

        metacritic_link_formatado = metacritic_link + "/" + metacritic_nome_formatado

        driver = webdriver.Edge(service=service, options=options)
        driver.get(metacritic_link_formatado)

    elif dic_nomes_jogos[contador_metacritic] == "Err":
        print("Dicionario de nomes = Err")
        dic_nota_metacritic.append("Err")
        return contador_metacritic, dic_nota_metacritic
    # ---- Cookies bar ----

    try:
        recusar_cookies = driver.find_element(By.ID, "onetrust-reject-all-handler")
        recusar_cookies.click()
        print("NO COOKIES!!!")

    except NoSuchElementException:
        print("Não ofereceu cookies?")
        


    #---- Metacritic notas ----

    notas_provisorias = []
    metacritic_score_conteiner = driver.find_element(By.CLASS_NAME, "c-productHero_score-container")
    metacritic_score_divs = metacritic_score_conteiner.find_elements(By.XPATH, ".//div[contains(@class, 'c-productScoreInfo_scoreNumber')]")

    for div in metacritic_score_divs:
        span = div.find_element(By.TAG_NAME, "span")
        notas_provisorias.append(span.text)

    dic_nota_metacritic.append(notas_provisorias)

    # ---- return ---- 
    driver.quit()
    return contador_metacritic, dic_nota_metacritic

def embed():
    
    return
# ---- Regras ----

# ---- Pegar links ----
driver.get(dic_link_jogos[contador])

pegar_links()

driver.quit()

# ---- Dados de cada jogo ----

for elemento in dic_link_jogos[1:]:
    contador += 1
    pagina_jogo(contador)

driver.quit()

# ---- Metacritic ----

for elemento in dic_nomes_jogos:
    metacritic(dic_nota_metacritic, dic_nomes_jogos, contador_metacritic)
    contador_metacritic += 1

# ---- quit ----
driver.quit()




print(dic_link_jogos,'\n',
dic_nomes_jogos,'\n',
dic_descricao_jogo,'\n',
dic_preco_jogo,'\n',
dic_fim_promoção_jogo,'\n',
dic_capa_jogos,'\n',
dic_genero_jogos,'\n',
dic_nota_metacritic
)


# --- CONFIGURAÇÃO INICIAL E IMPORTS ---
import torch  # Biblioteca base para Deep Learning
import torch.nn as nn  # Módulo para construir Redes Neurais
import torch.optim as optim  # Módulo para otimizadores (Adam)
from torch.utils.data import Dataset, DataLoader, random_split  # Gerenciamento e divisão de dados
import numpy as np  # Biblioteca para operações numéricas (usada para seeds)
import os  # Módulo para verificar se o modelo já está salvo (persistência)
from sklearn.metrics import accuracy_score, f1_score

# Definindo uma semente (seed) para garantir a reprodutibilidade dos resultados
torch.manual_seed(42)
np.random.seed(42)
MODEL_PATH = 'compatibility_model_final_weights.pth'  # Caminho para salvar os pesos treinados

# --- DEFINIÇÃO DAS DIMENSÕES (FEATURES) ---

# 13 características do perfil do adotante
# [0] Tipo Imóvel, [1] Área Externa, [2] Imóvel Telado?, [3] Qtd Moradores, [4] Há Crianças?, [5] Há Idosos?,
# [6] Outros Animais, [7] Experiência, [8] Tempo Disp, [9] Tempo Fora, [10] Aceita Esp?, [11] Gastos Mensais, [12] Exp Prévia Esp?
ADOPTER_FEATURES = 13

# 7 características do perfil do pet
# [0] Espécie, [1] Porte, [2] Cuidados Especiais/Crônica, [3] Trauma/Comportamental, [4] Sociável Crianças,
# [5] Sociável Animais, [6] Necessita Tutor Experiente?
PET_FEATURES = 7


# ==============================================================================
# 1. IMPLEMENTAÇÃO DO DATASET (CRIAÇÃO DOS DADOS DE TREINAMENTO)
# ==============================================================================
class PetCompatibilityDataset(Dataset):
    """
    Dataset fictício adaptado para 20 características (13 Adotante + 7 Pet).
    Gera amostras com regras de compatibilidade e ruído (20%).
    80% para treinamento e 20% para validação
    """

    def __init__(self, num_samples):
        self.adopter_features = ADOPTER_FEATURES
        self.pet_features = PET_FEATURES

        # Gerando 10.000 dados fictícios aleatórios (Escala 0 a 5)
        self.adopters = torch.randint(0, 6, (num_samples, self.adopter_features)).float()
        self.pets = torch.randint(0, 6, (num_samples, self.pet_features)).float()

        # --- REGRAS DE COMPATIBILIDADE ---

        # 1. Regra de Recurso/Cuidado: (Adotante com alta Experiência [índice 7 >= 3] E Pet com Trauma [índice 3 == 1])
        care_match = ((self.adopters[:, 7] >= 3) & (self.pets[:, 3] == 1)).float()

        # 2. Regra de Ambiente: (Adotante possui Casa/Chácara [índice 0 >= 3] E Pet é de Porte Grande [índice 1 == 5])
        # Ambiente adequado para um pet com alta demanda de espaço.
        env_match = ((self.adopters[:, 0] >= 3) & (self.pets[:, 1] == 5)).float()

        # 3. Aceita Animais com necessidades especiais?
        # Verifica se o pet precisa de cuidados esp e se o adotante aceita esp
        special_care_match = ((self.adopters[:, 10] == 1) & (self.pets[:, 2] == 1)).float()

        # 4. Sociável com Crianças?
        # Verifica a sociabilidade do pet com Crianças
        has_kids = (self.adopters[:, 4] == 1)
        pet_social_kids = (self.pets[:, 4] == 1)
        # Se o adotante não possuir crianças o lado direito é ignorado
        # Se o adotante possuir crianças o programa verifica o lado direito
        kids_match_safe = (~has_kids | (has_kids & pet_social_kids))

        # 5. Sociável com Animais?
        # Verifica se o pet é sociável com Animais
        has_pets = (self.adopters[:, 6] == 1)
        pet_social_pets = (self.pets[:, 5] == 1)
        # Se o adotante não possuir pets o lado direito é ignorado
        # Se o adotante possuir pets o programa verifica o lado direito
        pets_match_safe = (~has_pets | (has_pets & pet_social_pets))

        # A Regra de Segurança Social é atendida SOMENTE SE AMBOS os matches forem True
        social_safety_match = (kids_match_safe & pets_match_safe).float()

        # 6. Necessita tutor experiente?
        exp_care_match = ((self.adopters[:, 7] >= 3) & (self.pets[:, 6] == 1)).float()

        # Pesos das Regras de Compatibilidade
        # Cada Regra de Compatibilidade Possuí valor == 1 que será multiplicada pelo seu peso
        final_score = (
            (care_match * 3)+           # Regra 1 (Recurso/Cuidado)
            (env_match * 1)+            # Regra 2 (Ambiente)
            (social_safety_match * 7)+  # Regra 3 (Segurança Social)
            (special_care_match * 3)+   # Regra 4 (Aceita Esp)
            (exp_care_match * 9)        # Regra 5 (Tutor Exp)
        )

        # O rótulo é TRUE se o score for maior ou igual a 3.
        # Isso garante que a compatibilidade só seja True se as regras de segurança/cuidado de alto peso forem atendidas.
        self.labels = (final_score >= 5).float().unsqueeze(1)

        # --- ADIÇÃO DE RUÍDO (20% para forçar a generalização) ---
        noise = (torch.rand(num_samples, 1) > 0.8).float()  # 20% dos valores aleatórios serão 1
        self.labels = torch.abs(self.labels - noise)  # Aplica a inversão de rótulo onde o ruído for 1

    def __len__(self):
        # Retorna o número total de amostras (essencial para o DataLoader)
        return len(self.adopters)

    def __getitem__(self, idx):
        # Retorna o perfil do adotante, do pet e o rótulo de compatibilidade para um dado índice
        adopter_profile = self.adopters[idx]
        pet_profile = self.pets[idx]
        compatibility_label = self.labels[idx]

        return adopter_profile, pet_profile, compatibility_label

    # ==============================================================================


# 2. IMPLEMENTAÇÃO DO MODELO DE COMPATIBILIDADE
# ==============================================================================
class CompatibilityModel(nn.Module):
    """
    Rede Neural com uma camada oculta.
    """

    def __init__(self, adopter_features, pet_features):
        super(CompatibilityModel, self).__init__()

        # Dimensão de entrada: 13 features do adotante + 7 do pet = 20
        combined_features = adopter_features + pet_features

        self.network = nn.Sequential(
            nn.Linear(combined_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # Sáida entre 0 e 1 para o resultado da compatibilidade
        )

    def forward(self, adopter_profile, pet_profile):
        # Concatenação de Vetores
        combined_profile = torch.cat((adopter_profile, pet_profile), dim=1)
        compatibility_score = self.network(combined_profile)
        return compatibility_score


# ==============================================================================
# 3. TREINAMENTO, VALIDAÇÃO E TESTES
# ==============================================================================

# --- Teste de Validação ---
def validate_model(model, val_loader, criterion):
    model.eval()  # Coloca o modelo em modo de avaliação (desliga o aprendizado)
    total_loss = 0.0
    with torch.no_grad():  # Desativa o cálculo de gradientes (otimização)
        for adopter, pet, label in val_loader:
            outputs = model(adopter, pet)
            loss = criterion(outputs, label)
            total_loss += loss.item() * adopter.size(0)

    avg_loss = total_loss / len(val_loader.dataset)  # Perda Média = Erro Total / Número de Amostras
    model.train()  # Volta o modelo para o modo de treinamento
    return avg_loss


# --- Rotina Principal de Treinamento (Early Stopping Manual) ---
def train_model_with_validation(model, train_loader, val_loader, epochs):
    criterion = nn.BCELoss()  # Função de Perda para classificação binária
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Otimizador Adam

    print(f"Iniciando o treinamento do modelo com {ADOPTER_FEATURES + PET_FEATURES} features ({epochs} épocas)")

    for epoch in range(epochs):
        current_train_loss = 0.0

        for adopter, pet, label in train_loader:
            optimizer.zero_grad()  # Zera os gradientes da iteração anterior
            outputs = model(adopter, pet)
            loss = criterion(outputs, label)
            loss.backward()  # Backpropagation: calcula o gradiente (ajuste)
            optimizer.step()  # Adam: aplica o ajuste aos pesos
            current_train_loss = loss.item()

        # Teste de Generalização
        val_loss = validate_model(model, val_loader, criterion)

        # LOGS DE PROGRESSO: AGORA A CADA 5 ÉPOCAS
        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | Treino Perda: {current_train_loss:.4f} | Validação Perda: {val_loss:.4f}")
    print("Treinamento concluído!")

    # Cálculo De Métricas
    preds, true_labels = [], []
    model.eval()
    with torch.no_grad():
        for adopter, pet, label in val_loader:
            output = model(adopter, pet)
            preds.extend((output > 0.5).float().numpy())
            true_labels.extend(label.numpy())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    print(f"Acurácia: {acc:.3f} | F1-Score: {f1:.3f}")


# ==============================================================================
# 4. PREDIÇÃO DA COMPATIBILIDADE
# ==============================================================================

def predict_and_rank_pets(model, adopter_profile, pet_list):
    """
    Calcula a compatibilidade, aplica uma camada de salvaguarda de segurança
    e ordena os pets seguros.
    """
    def is_safe_match(adopter_profile, pet_profile):
        """
        Verifica de forma programática se a regra de segurança social é atendida.
        Retorna True se o match for seguro, False caso contrário.
        """
        # Recria a lógica da regra de segurança
        has_kids = (adopter_profile[4] == 1)
        pet_social_kids = (pet_profile[4] == 1)
        kids_match_safe = (not has_kids) or (has_kids and pet_social_kids)

        has_pets = (adopter_profile[6] == 1)
        pet_social_pets = (pet_profile[5] == 1)
        pets_match_safe = (not has_pets) or (has_pets and pet_social_pets)

        return kids_match_safe and pets_match_safe

    adopter_tensor = torch.tensor(adopter_profile).float().unsqueeze(0)
    safe_results = []
    unsafe_results = []

    print("\n--- Calculando compatibilidade para o Adotante ---")
    print(f"Perfil do Adotante (13 Features): {adopter_profile}")

    model.eval()
    with torch.no_grad():
        for i, pet in enumerate(pet_list):
            # Passo 1: O modelo de IA calcula a pontuação probabilística
            pet_tensor = torch.tensor(pet).float().unsqueeze(0)
            score = model(adopter_tensor, pet_tensor).item()
            result_data = {'pet_id': i + 1, 'profile': pet, 'score': score}

            # Passo 2: CAMADA DE SALVAGUARDA (Regra de Negócio)
            # Verificamos se o pet passa na regra de segurança crítica
            if is_safe_match(adopter_profile, pet):
                safe_results.append(result_data)
            else:
                unsafe_results.append(result_data)

    # Ordena apenas a lista de resultados seguros
    safe_results_sorted = sorted(safe_results, key=lambda x: x['score'], reverse=True)

    print("\n--- Ranking de Pets SEGUROS por Compatibilidade ---")
    if not safe_results_sorted:
        print("  Nenhum pet seguro encontrado para este perfil de adotante.")
    else:
        for rank, res in enumerate(safe_results_sorted):
            print(f"  Posição {rank + 1}: Pet {res['pet_id']} (Perfil: {res['profile']})")
            print(f"    Pontuação da IA: {res['score']:.4f} (Status: SEGURO)")

    if unsafe_results:
        print("\n--- Pets Filtrados por Violação da Regra de Segurança ---")
        for res in unsafe_results:
            print(f"  - Pet {res['pet_id']} (Perfil: {res['profile']})")
            print(f"    Pontuação da IA: {res['score']:.4f} (Status: INSEGURO/FILTRADO)")

    '''
    Return com os resultados dos pets compatíveis ranqueados e não compatíveis    
    '''
    return {
        'ranked_safe_pets': safe_results_sorted,
        'filtered_unsafe_pets': unsafe_results,
    }

# ==============================================================================
# 5. ANÁLISE DE ERRO
# ==============================================================================
def analyze_safety_rule_violations(model, val_loader):
    """
    Analisa o conjunto de validação para identificar a frequência com que o modelo
    recomenda um pet (score > 0.5) que viola a regra de segurança social.
    """
    model.eval()  # Coloca o modelo em modo de avaliação

    safety_violations_by_model = 0
    total_safety_rule_failures = 0

    print("\n--- Iniciando Análise de Erro Focada na Regra de Segurança ---")

    with torch.no_grad():
        for adopter, pet, _ in val_loader:
            # 1. Recalcular a regra de segurança para este lote de dados
            has_kids = (adopter[:, 4] == 1)
            pet_social_kids = (pet[:, 4] == 1)
            kids_match_safe = (~has_kids | (has_kids & pet_social_kids))

            has_pets = (adopter[:, 6] == 1)
            pet_social_pets = (pet[:, 5] == 1)
            pets_match_safe = (~has_pets | (has_pets & pet_social_pets))

            # A regra de segurança falha se qualquer uma das condições não for atendida
            social_safety_rule_passed = (kids_match_safe & pets_match_safe)

            # 2. Obter a predição do modelo
            outputs = model(adopter, pet).squeeze()  # .squeeze() remove a dimensão extra
            predictions = (outputs > 0.5)

            # 3. Identificar o cenário de violação
            # Queremos os casos onde a REGRA FALHOU (foi False), mas o MODELO RECOMENDOU (foi True)
            violation_scenario = (~social_safety_rule_passed & predictions)

            # 4. Contabilizar os totais
            safety_violations_by_model += torch.sum(violation_scenario).item()
            total_safety_rule_failures += torch.sum(~social_safety_rule_passed).item()

    if total_safety_rule_failures > 0:
        violation_rate = (safety_violations_by_model / total_safety_rule_failures) * 100
        print(f"Análise Concluída:")
        print(
            f" - Total de Casos com Falha na Regra de Segurança no Dataset de Validação: {total_safety_rule_failures}")
        print(f" - O Modelo recomendou um pet inseguro em: {safety_violations_by_model} casos")
        print(f" - Taxa de Violação de Segurança pelo Modelo: {violation_rate:.2f}%")
        print("   (Isso representa a porcentagem de pets inseguros que o modelo INCORRETAMENTE recomendou)")
    else:
        print("Análise Concluída: Nenhuma falha na regra de segurança foi encontrada no dataset de validação.")

    model.train()  # Retorna o modelo ao modo de treino por boas práticas

# --- PERFIS DE TESTE (13 FEATURES ADOTANTE / 7 FEATURES PET) ---
# Adotante: [Tipo Imóvel (3.0=Casa), Area Ext (1.0=Sim), Telado (1.0=Sim), Moradores (5.0), Crianças (0.0=Não), Idosos (0.0=Não), Outros Animais (1.0=Sim), Experiência (1.0=Sim), Tempo Disp (5.0=Muito), Tempo Fora (1.0=Pouco), Aceita Esp (1.0=Sim), Gastos (3.0), Exp Previa Esp (1.0=Sim)]
new_adopter_profile = [3.0, 1.0, 1.0, 5.0, 0.0, 0.0, 1.0, 1.0, 5.0, 1.0, 1.0, 3.0, 1.0]

# Formato: [Espécie, Porte, Cuidados Especiais, Trauma, Sociável Crianças, Sociável Animais, Tutor Experiente]
potential_pets = [
    [1.0, 5.0, 0.0, 0.0, 1.0, 1.0, 1.0],  # Pet 1
    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # Pet 2
    [1.0, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0],  # Pet 3
    [1.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # Pet 4
    [1.0, 3.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # Pet 5
    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],  # Pet 6
    [1.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Pet 7
    [1.0, 3.0, 0.0, 0.0, 1.0, 1.0, 1.0],  # Pet 8
    [0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # Pet 9
    [1.0, 3.0, 0.0, 1.0, 1.0, 1.0, 1.0],  # Pet 10
    [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],  # Pet 11
    [1.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Pet 12
    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],  # Pet 13
    [0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Pet 14
    [1.0, 5.0, 0.0, 0.0, 1.0, 1.0, 1.0]   # Pet 15
]


# --- INICIALIZAÇÃO DO MODELO ---

# Valores do DataSet com 50000 amostras
full_dataset = PetCompatibilityDataset(num_samples=50000)
# Ruído 20% (80% para treinamento e 20% para validação)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Organização dos datasets em lotes de 32 cada
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Cria uma instância em branco
model = CompatibilityModel(ADOPTER_FEATURES, PET_FEATURES)


# Lógica de persistência: Carrega ou Treina
if os.path.exists(MODEL_PATH):
    print(f"\nCarregando modelo pré-treinado de: {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Modelo carregado com sucesso. Pulando treinamento.")
else:
    print("\n--- PESOS NÃO ENCONTRADOS ---")
    train_model_with_validation(model, train_dataloader, val_dataloader, epochs=35)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Pesos do modelo salvos em: {MODEL_PATH}")

# Executa a Análise de Erro Focada no dataset de validação
analyze_safety_rule_violations(model, val_dataloader)

# Executa o ranqueamento com o modelo carregado ou recém-treinado
predict_and_rank_pets(model, new_adopter_profile, potential_pets)

