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