"""
Trabalho Prático de PAI — Classificação da resposta ao tratamento do câncer de mama (pCR)
Dataset: BreastDCEDL, subconjunto I-SPY1 MinCrop Version 4
Módulo: ResNet50 

Integrante: João Pedro Torres - 857622

Seções implementadas:
    1. Configuração
    2. Dados: leitura, validação, normalização conjunta e seleção de cortes
    3. Aumento de dados: espelhamento e rotações (somente treino)
    4. Entrada e modelo: normalização ImageNet e ResNet50 adaptada
    5. Extração antecipada de características (backbone congelado)
       5.1 Variante com recorte da região tumoral (experimento complementar; usa a máscara)
    6. Treino da cabeça classificadora
    7. Avaliação por paciente
    8. Grad-CAM
"""
from __future__ import annotations

import copy
import io
import json
import os
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


# =============================================================================
# 1. CONFIGURAÇÃO
# =============================================================================

SEMENTE = 42

# Ordem das fases (e dos canais da rede) em todo o código:
# canal 0 = pré-contraste, canal 1 = pós-contraste precoce, canal 2 = pós-contraste tardio
NOMES_FASES = ("pre", "post_early", "post_late")

# Coluna "test" do CSV, segundo a Tabela 4 do artigo do BreastDCEDL.
# Atenção: 1 é TESTE e 2 é VALIDAÇÃO (ordem não intuitiva).
NOMES_PARTICAO = {0: "treino", 1: "teste", 2: "validacao"}

# Experimento A: uma fase replicada nos 3 canais. Experimento B: as 3 fases como 3 canais.
EXPERIMENTOS = ("A", "B")
FASE_EXPERIMENTO_A = "post_early"

# Todos os volumes do MinCrop estão no formato (Z, 256, 256): o eixo 0 é o eixo dos cortes.
# (Verificado nos cabeçalhos dos 688 arquivos.)
EIXO_Z = 0
TAMANHO_PLANO = (256, 256)

# Percentis da normalização conjunta (ver normalizar_conjuntamente).
# O superior é P99.9, e não P99: o tumor ocupa ~1-2% do volume e é a região mais clara
# no pós-contraste, então o P99 caía dentro do tumor e saturava a maior parte dele.
# Comparação em treino/validação (teste excluído), fase pós precoce, mediana entre pacientes:
#   P99   -> 71% do tumor recortado (treino) | 72,9% do realce preservado
#   P99.9 -> 10% do tumor recortado (treino) | 98,1% do realce preservado
PERCENTIL_INFERIOR = 1.0
PERCENTIL_SUPERIOR = 99.9

# Padrões dos nomes de arquivo, ex.: ISPY1_1001_spy1_vis1_dce_aqc_0.nii.gz
# O número após "aqc_" é o índice da aquisição no exame original (não é 0, 1, 2 para todos).
PADRAO_ARQUIVO_FASE = re.compile(r"^(?P<pid>ISPY1_\d+)_spy1_vis1_dce_aqc_(?P<indice>\d+)\.nii\.gz$")
SUFIXO_MASCARA = "_spy1_vis1_mask.nii.gz"


@dataclass
class CaminhosDataset:
    """Caminhos das pastas de imagens, máscaras e do CSV de metadados."""

    pasta_dce: str
    pasta_mascara: str
    caminho_csv: str

    @classmethod
    def a_partir_da_raiz(cls, pasta_raiz: str, caminho_csv: str) -> "CaminhosDataset":
        """Monta os caminhos a partir da pasta que contém as subpastas dce/ e mask/."""
        return cls(os.path.join(pasta_raiz, "dce"), os.path.join(pasta_raiz, "mask"), caminho_csv)

    def validar(self) -> None:
        for pasta in (self.pasta_dce, self.pasta_mascara):
            if not os.path.isdir(pasta):
                raise FileNotFoundError(f"Pasta não encontrada: {pasta}")
        if not os.path.isfile(self.caminho_csv):
            raise FileNotFoundError(f"CSV não encontrado: {self.caminho_csv}")


def definir_semente(semente: int = SEMENTE) -> None:
    """Fixa as sementes aleatórias para tornar os experimentos reproduzíveis."""
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)
    # Força o cuDNN (biblioteca da GPU) a usar algoritmos determinísticos: a mesma
    # entrada passa a gerar sempre a mesma saída, ao custo de um pouco de velocidade.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 2. DADOS
# =============================================================================

class AvisoDivergenciaDados(UserWarning):
    """Aviso de divergência entre o CSV e os arquivos. O CSV original nunca é alterado."""


# -----------------------------------------------------------------------------
# 2.1 Metadados e localização dos arquivos
# -----------------------------------------------------------------------------

def ler_metadados(caminho_csv: str) -> pd.DataFrame:
    """Lê o CSV, valida rótulos e partições e adiciona a coluna legível 'particao'."""
    # O CSV usa '\r' (padrão antigo do Mac) como fim de linha; normalizamos para '\n'.
    with open(caminho_csv, "r", newline="") as arquivo:
        texto = arquivo.read().replace("\r\n", "\n").replace("\r", "\n")
    metadados = pd.read_csv(io.StringIO(texto))

    if metadados["pid"].duplicated().any():
        raise ValueError("Há pacientes repetidos no CSV.")
    if metadados["pCR"].isna().any():
        raise ValueError("Há pacientes sem rótulo pCR no CSV.")

    metadados["pCR"] = metadados["pCR"].astype(int)
    metadados["particao"] = metadados["test"].astype(int).map(NOMES_PARTICAO)
    if metadados["particao"].isna().any():
        raise ValueError("Valor inesperado na coluna 'test' do CSV.")
    return metadados


def listar_indices_aquisicao(pasta_dce: str) -> dict[str, list[int]]:
    """Mapeia cada paciente para a lista ordenada de índices de aquisição presentes em dce/."""
    indices_por_paciente: dict[str, list[int]] = {}
    for nome_arquivo in os.listdir(pasta_dce):
        correspondencia = PADRAO_ARQUIVO_FASE.match(nome_arquivo)
        if correspondencia is None:
            warnings.warn(f"Arquivo fora do padrão ignorado em dce/: {nome_arquivo}", AvisoDivergenciaDados)
            continue
        indices_por_paciente.setdefault(correspondencia["pid"], []).append(int(correspondencia["indice"]))
    return {pid: sorted(indices) for pid, indices in indices_por_paciente.items()}


def resolver_indices_fases(pid: str, indices_disponiveis: list[int], linha_csv: pd.Series) -> dict[str, int]:
    """
    Decide qual arquivo corresponde a cada fase.

    Regra adotada: os 3 arquivos existentes do paciente, ordenados pelo índice de
    aquisição, são (pré, pós precoce, pós tardio). Em 171 dos 172 pacientes essa regra
    coincide com as colunas pre/post_early/post_late do CSV.

    Exceção conhecida: ISPY1_1139 — o CSV indica (0, 2, 2), mas os arquivos são (0, 1, 2).
    Seguir o CSV duplicaria a fase tardia e ignoraria o arquivo _1. Nesse caso usamos a
    ordenação dos arquivos e emitimos um aviso, sem modificar o CSV.
    """
    if len(indices_disponiveis) != len(NOMES_FASES):
        raise ValueError(f"{pid}: esperados 3 arquivos de fase, encontrados índices {indices_disponiveis}")

    indices_arquivos = sorted(indices_disponiveis)
    indices_csv = [int(linha_csv[nome_fase]) for nome_fase in NOMES_FASES]
    if indices_arquivos != indices_csv:
        warnings.warn(
            f"{pid}: índices do CSV {indices_csv} diferem dos arquivos {indices_arquivos}; "
            f"usando a ordenação dos arquivos.",
            AvisoDivergenciaDados,
        )
    return dict(zip(NOMES_FASES, indices_arquivos))


def caminho_fase(caminhos: CaminhosDataset, pid: str, indice_aquisicao: int) -> str:
    return os.path.join(caminhos.pasta_dce, f"{pid}_spy1_vis1_dce_aqc_{indice_aquisicao}.nii.gz")


def caminho_mascara(caminhos: CaminhosDataset, pid: str) -> str:
    return os.path.join(caminhos.pasta_mascara, f"{pid}{SUFIXO_MASCARA}")


# -----------------------------------------------------------------------------
# 2.2 Leitura e validação dos volumes
# -----------------------------------------------------------------------------

def ler_volume(caminho: str) -> np.ndarray:
    """
    Lê um volume NIfTI como float32 no formato (Z, 256, 256).

    Os arquivos estão em float64; float32 ocupa metade da memória e tem precisão
    mais que suficiente para intensidades de MRI.
    """
    volume = nib.load(caminho).get_fdata(dtype=np.float32)
    if volume.ndim != 3 or volume.shape[1:] != TAMANHO_PLANO:
        raise ValueError(f"Forma inesperada {volume.shape} em {caminho}")
    if not np.isfinite(volume).all():
        raise ValueError(f"Valores NaN ou infinitos em {caminho}")
    return volume


def carregar_paciente(caminhos: CaminhosDataset, linha_csv: pd.Series, indices_disponiveis: list[int]) -> dict:
    """
    Carrega as 3 fases e a máscara de um paciente, com validações.

    Retorna um dicionário com:
        volumes  -> float32 (3, Z, 256, 256), na ordem de NOMES_FASES
        mascara  -> uint8   (Z, 256, 256), 0 = fundo, 1 = tumor
    """
    pid = linha_csv["pid"]
    indices_fases = resolver_indices_fases(pid, indices_disponiveis, linha_csv)

    lista_volumes = [ler_volume(caminho_fase(caminhos, pid, indices_fases[nome])) for nome in NOMES_FASES]
    formas = {volume.shape for volume in lista_volumes}
    if len(formas) != 1:
        raise ValueError(f"{pid}: as 3 fases têm formas diferentes: {formas}")
    volumes = np.stack(lista_volumes)

    mascara = ler_volume(caminho_mascara(caminhos, pid))
    if mascara.shape != volumes.shape[1:]:
        raise ValueError(f"{pid}: máscara {mascara.shape} incompatível com as fases {volumes.shape[1:]}")
    valores_mascara = np.unique(mascara)
    if not np.isin(valores_mascara, (0.0, 1.0)).all():
        raise ValueError(f"{pid}: máscara com valores diferentes de 0 e 1: {valores_mascara}")

    return {
        "pid": pid,
        "pCR": int(linha_csv["pCR"]),
        "particao": linha_csv["particao"],
        "indices_fases": indices_fases,
        "volumes": volumes,
        "mascara": mascara.astype(np.uint8),
    }


# -----------------------------------------------------------------------------
# 2.3 Normalização conjunta das fases
# -----------------------------------------------------------------------------

def normalizar_conjuntamente(
    volumes: np.ndarray,
    percentil_inferior: float = PERCENTIL_INFERIOR,
    percentil_superior: float = PERCENTIL_SUPERIOR,
) -> tuple[np.ndarray, dict]:
    """
    Normaliza as 3 fases de UM paciente com uma referência comum.

    1. Os limites são P1 e P99.9, calculados sobre os voxels das 3 fases juntas.
    2. Valores fora dos limites são recortados (clip).
    3. Todas as fases são reescaladas para [0, 1] com os MESMOS limites.

    Como a mesma transformação linear é aplicada às 3 fases, um tumor que fica mais
    claro após o contraste continua mais claro depois da normalização: o realce,
    que é a informação temporal investigada no trabalho, é preservado.
    Normalizar por paciente remove diferenças de escala entre aparelhos e centros.

    Por que P99.9 (escolhido só com treino e validação, comparando P99, P99.5, P99.9,
    P99.99 e normalização independente por fase):
    - P99 e P99.5 saturavam a maior parte do tumor nas fases pós-contraste;
    - P99.99 quase não satura, mas o limite passa a depender de pouquíssimos voxels
      (vasos, artefatos), tornando a escala instável entre pacientes;
    - a normalização independente por fase continuava saturando o tumor e distorcia
      a relação entre as fases com um fator diferente para cada paciente.
    Limitação: em ~10% dos pacientes, mais de 40% do tumor ainda satura no pós-contraste.
    """
    limite_inferior, limite_superior = np.percentile(volumes, [percentil_inferior, percentil_superior])
    if limite_superior <= limite_inferior:
        raise ValueError(f"Limites de normalização inválidos: {limite_inferior}, {limite_superior}")

    normalizados = (np.clip(volumes, limite_inferior, limite_superior) - limite_inferior) / (
        limite_superior - limite_inferior
    )
    limites = {"limite_inferior": float(limite_inferior), "limite_superior": float(limite_superior)}
    return normalizados.astype(np.float32), limites


def estatisticas_intensidade(volumes: np.ndarray, mascara: np.ndarray | None = None) -> pd.DataFrame:
    """
    Tabela de intensidades por fase (mín., percentis, máx., média).

    Se a máscara for informada, inclui a intensidade média dentro do tumor. Comparar
    essa coluna antes e depois da normalização mostra se o realce foi preservado.
    """
    linhas = []
    for nome_fase, volume in zip(NOMES_FASES, volumes):
        p1, p50, p99 = np.percentile(volume, [1, 50, 99])
        linha = {
            "min": float(volume.min()),
            "p1": float(p1),
            "mediana": float(p50),
            "p99": float(p99),
            "max": float(volume.max()),
            "media": float(volume.mean()),
        }
        if mascara is not None:
            linha["media_tumor"] = float(volume[mascara > 0].mean())
        linhas.append(pd.Series(linha, name=nome_fase))
    return pd.DataFrame(linhas)


def carregar_paciente_normalizado(
    caminhos: CaminhosDataset, linha_csv: pd.Series, indices_disponiveis: list[int]
) -> dict:
    """Carrega o paciente e adiciona 'normalizados' (3, Z, 256, 256) e os limites usados."""
    paciente = carregar_paciente(caminhos, linha_csv, indices_disponiveis)
    paciente["normalizados"], paciente["limites_normalizacao"] = normalizar_conjuntamente(paciente["volumes"])
    return paciente


# -----------------------------------------------------------------------------
# 2.4 Estratégias de seleção de cortes
# -----------------------------------------------------------------------------
# As redes 2D recebem imagens, não volumes. Os cortes são devolvidos no formato
# (N, 3, 256, 256): N cortes, 3 canais (fases), altura e largura — a ordem
# "canais primeiro" usada pelo PyTorch.

def area_tumoral_por_corte(mascara: np.ndarray) -> np.ndarray:
    """Número de pixels de tumor em cada corte z."""
    return mascara.sum(axis=(1, 2))


def selecionar_corte_maior_area(mascara: np.ndarray) -> int:
    """Índice z do corte com maior área tumoral na máscara."""
    areas = area_tumoral_por_corte(mascara)
    if areas.max() == 0:
        raise ValueError("Máscara sem nenhum pixel de tumor.")
    return int(areas.argmax())


def cortes_estrategia_maior_area(normalizados: np.ndarray, mascara: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ESTRATÉGIA 1 — EXPERIMENTAL, COM INFORMAÇÃO DE SEGMENTAÇÃO.

    Usa a máscara verdadeira (anotação de especialista) para escolher o corte de maior
    área tumoral. A máscara serve SOMENTE para escolher o corte; ela não entra na rede.
    Ressalva: em uma predição real essa segmentação normalmente não estaria disponível,
    portanto esta estratégia não é independente da máscara.

    Retorna (cortes (1, 3, 256, 256), índices z usados).
    """
    indice_z = selecionar_corte_maior_area(mascara)
    return normalizados[:, indice_z][np.newaxis], np.array([indice_z])


def cortes_estrategia_todos(normalizados: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ESTRATÉGIA 2 — TODOS OS CORTES, SEM USAR A MÁSCARA.

    Devolve todos os Z cortes do volume; nenhum é escolhido ou descartado pela máscara.
    As probabilidades previstas para cada corte serão agregadas pela média, no nível
    do paciente, na seção de avaliação.

    Retorna (cortes (Z, 3, 256, 256), índices z usados).
    """
    cortes = np.moveaxis(normalizados, EIXO_Z + 1, 0)  # (3, Z, H, W) -> (Z, 3, H, W)
    return np.ascontiguousarray(cortes), np.arange(cortes.shape[0])


def montar_canais(cortes: np.ndarray, experimento: str, fase_unica: str = FASE_EXPERIMENTO_A) -> np.ndarray:
    """
    Monta os 3 canais de entrada da ResNet50 a partir de cortes (N, 3, H, W).

    Experimento A: a fase escolhida é replicada nos 3 canais (a ResNet pré-treinada
                   espera 3 canais; replicar evita alterar a primeira camada).
    Experimento B: as 3 fases já são os 3 canais (pré, pós precoce, pós tardio).
    """
    if experimento == "B":
        return cortes
    if experimento == "A":
        canal = cortes[:, NOMES_FASES.index(fase_unica)]
        return np.repeat(canal[:, np.newaxis], 3, axis=1)
    raise ValueError(f"Experimento desconhecido: {experimento} (use um de {EXPERIMENTOS})")


# -----------------------------------------------------------------------------
# 2.5 Auditoria do dataset e verificação visual
# -----------------------------------------------------------------------------

def auditar_dataset(caminhos: CaminhosDataset, metadados: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica todos os pacientes: arquivos de fase, formas, divergências com o CSV e
    distribuição do tumor ao longo de z. Lê só os cabeçalhos das fases e as máscaras.

    As colunas mask_start/mask_end do CSV descrevem o exame ORIGINAL (antes do recorte
    em z do MinCrop). Elas aparecem aqui apenas para documentação ('margem_z') e não
    são usadas para indexar cortes em nenhum outro lugar do código.
    """
    indices_por_paciente = listar_indices_aquisicao(caminhos.pasta_dce)
    pids_csv, pids_arquivos = set(metadados["pid"]), set(indices_por_paciente)
    if pids_csv != pids_arquivos:
        raise ValueError(
            f"Pacientes só no CSV: {sorted(pids_csv - pids_arquivos)}; "
            f"só nos arquivos: {sorted(pids_arquivos - pids_csv)}"
        )

    registros = []
    for _, linha in metadados.iterrows():
        pid = linha["pid"]
        indices_fases = resolver_indices_fases(pid, indices_por_paciente[pid], linha)
        formas = {nib.load(caminho_fase(caminhos, pid, indice)).shape for indice in indices_fases.values()}
        mascara = ler_volume(caminho_mascara(caminhos, pid))
        if len(formas) != 1 or mascara.shape not in formas:
            raise ValueError(f"{pid}: formas incompatíveis entre fases e máscara: {formas}, {mascara.shape}")

        areas = area_tumoral_por_corte(mascara)
        cortes_com_tumor = np.nonzero(areas)[0]
        profundidade = mascara.shape[EIXO_Z]
        indices_arquivos = tuple(indices_fases.values())
        indices_csv = tuple(int(linha[nome]) for nome in NOMES_FASES)
        registros.append({
            "pid": pid,
            "particao": linha["particao"],
            "pCR": linha["pCR"],
            "indices_arquivos": indices_arquivos,
            "indices_csv": indices_csv,
            "diverge_csv": indices_arquivos != indices_csv,
            "profundidade_z": profundidade,
            "cortes_com_tumor": len(cortes_com_tumor),
            "corte_maior_area": int(areas.argmax()),
            "tumor_toca_borda_z": bool(areas[0] > 0 or areas[-1] > 0),
            "margem_z": profundidade - int(linha["mask_end"] - linha["mask_start"]),
        })

    auditoria = pd.DataFrame(registros)

    print(f"Pacientes: {len(auditoria)} | cortes no total: {auditoria['profundidade_z'].sum()}")
    print("\nCortes por partição:")
    print(auditoria.groupby("particao")["profundidade_z"].agg(pacientes="count", cortes="sum", z_min="min", z_max="max"))
    print("\nDivergências CSV x arquivos:", auditoria.loc[auditoria["diverge_csv"], "pid"].tolist())
    print("Pacientes sem tumor na máscara:", auditoria.loc[auditoria["cortes_com_tumor"] == 0, "pid"].tolist())
    print("Tumor tocando o 1º ou o último corte:", auditoria.loc[auditoria["tumor_toca_borda_z"], "pid"].tolist())
    print("\nMargem em z (profundidade real − faixa do tumor no CSV original):")
    print(auditoria["margem_z"].value_counts().sort_index())
    return auditoria


def visualizar_paciente(paciente: dict, indice_z: int | None = None) -> None:
    """
    Mostra as 3 fases normalizadas no mesmo corte, a máscara sobreposta e a área
    tumoral por corte. As fases usam a mesma escala (0 a 1), então o realce do
    contraste aparece como aumento de brilho entre as fases.
    """
    normalizados, mascara = paciente["normalizados"], paciente["mascara"]
    areas = area_tumoral_por_corte(mascara)
    if indice_z is None:
        indice_z = int(areas.argmax())

    figura, paineis = plt.subplots(1, 5, figsize=(22, 4.5))
    for posicao, nome_fase in enumerate(NOMES_FASES):
        paineis[posicao].imshow(normalizados[posicao, indice_z], cmap="gray", vmin=0, vmax=1)
        paineis[posicao].set_title(f"{nome_fase} (aqc_{paciente['indices_fases'][nome_fase]})")
    paineis[3].imshow(normalizados[1, indice_z], cmap="gray", vmin=0, vmax=1)
    paineis[3].contour(mascara[indice_z] > 0, colors="red", linewidths=1)
    paineis[3].set_title("post_early + máscara")
    for painel in paineis[:4]:
        painel.axis("off")

    paineis[4].bar(np.arange(len(areas)), areas, color="gray")
    paineis[4].axvline(indice_z, color="red", linestyle="--", label=f"z = {indice_z}")
    paineis[4].set_xlabel("corte z")
    paineis[4].set_ylabel("pixels de tumor")
    paineis[4].legend()

    figura.suptitle(f"{paciente['pid']} — pCR={paciente['pCR']} — {paciente['particao']} — corte z={indice_z}")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 3. AUMENTO DE DADOS
# =============================================================================
# O enunciado especifica rotações de -20° a +20° em passos de 10°, com e sem
# espelhamento: 5 ângulos x 2 = 10 variações por imagem (a original é uma delas).
# Aplicado SOMENTE ao conjunto de treino; validação e teste usam apenas a imagem
# original, para que a avaliação meça pacientes reais e não cópias transformadas.

ANGULOS_ROTACAO = (-20, -10, 0, 10, 20)
ESPELHAMENTOS = (False, True)

# Lista fixa (determinística) das 10 variações: (ângulo em graus, espelhar?)
VARIACOES_AUMENTO = tuple((angulo, espelhar) for espelhar in ESPELHAMENTOS for angulo in ANGULOS_ROTACAO)
NOMES_VARIACOES = tuple(
    f"rot{angulo:+d}" + ("_espelho" if espelhar else "") for angulo, espelhar in VARIACOES_AUMENTO
)


def aplicar_variacao(imagem: np.ndarray, angulo: int, espelhar: bool, ordem_interpolacao: int = 1) -> np.ndarray:
    """
    Aplica espelhamento horizontal (opcional) e depois rotação a um corte (C, H, W).

    Os C canais são transformados JUNTOS, em uma única operação: no Experimento B
    isso garante que pré, pós precoce e pós tardio continuem alinhados pixel a pixel.

    ordem_interpolacao: 1 = bilinear (imagens); 0 = vizinho mais próximo (máscaras,
    para continuar binária). Os cantos que ficam sem imagem após a rotação recebem 0,
    o mesmo valor do fundo após a normalização. 'reshape=False' mantém 256 x 256.
    """
    resultado = imagem[..., ::-1] if espelhar else imagem   # inverte o eixo da largura (esquerda <-> direita)
    if angulo != 0:
        eixos_plano = (imagem.ndim - 2, imagem.ndim - 1)    # rotação no plano (H, W), igual para todos os canais
        resultado = ndimage.rotate(
            resultado, angulo, axes=eixos_plano, reshape=False,
            order=ordem_interpolacao, mode="constant", cval=0.0,
        )
        # A interpolação é uma média ponderada dos vizinhos e não ultrapassa os extremos
        # originais; o recorte só remove ruído de arredondamento (ex.: 1.0000001).
        resultado = np.clip(resultado, min(imagem.min(), 0.0), imagem.max())
    return np.ascontiguousarray(resultado, dtype=imagem.dtype)


def gerar_variacoes(corte: np.ndarray, ordem_interpolacao: int = 1) -> np.ndarray:
    """Gera as 10 variações de um corte (C, H, W) -> (10, C, H, W), na ordem de NOMES_VARIACOES."""
    return np.stack([
        aplicar_variacao(corte, angulo, espelhar, ordem_interpolacao)
        for angulo, espelhar in VARIACOES_AUMENTO
    ])


# =============================================================================
# 4. ENTRADA E MODELO (RESNET50)
# =============================================================================

# Média e desvio-padrão por canal das imagens do ImageNet. Os pesos pré-treinados
# foram aprendidos com entradas nessa escala, então aplicamos a mesma padronização
# depois da nossa normalização para [0, 1].
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)

NUM_CARACTERISTICAS = 2048   # tamanho do vetor que sai do backbone da ResNet50
TAXA_DROPOUT = 0.5           # fração de características zeradas a cada passo de treino


def obter_dispositivo() -> torch.device:
    """GPU (cuda) se disponível; caso contrário, CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preparar_lote(cortes: np.ndarray) -> torch.Tensor:
    """
    Converte cortes (N, 3, H, W) em [0, 1] no tensor de entrada da ResNet50.

    Os cortes já devem estar com os canais montados (ver montar_canais). O tamanho
    256 x 256 é mantido: a ResNet termina em uma média global (adaptive average
    pooling), que aceita qualquer tamanho, então não é preciso redimensionar.
    """
    if cortes.ndim != 4 or cortes.shape[1] != 3:
        raise ValueError(f"Esperado (N, 3, H, W), recebido {cortes.shape}")
    if cortes.min() < 0 or cortes.max() > 1:
        raise ValueError(f"Cortes fora de [0, 1]: mín={cortes.min()}, máx={cortes.max()}")

    lote = torch.from_numpy(np.ascontiguousarray(cortes, dtype=np.float32))
    media = torch.tensor(MEDIA_IMAGENET).view(1, 3, 1, 1)
    desvio = torch.tensor(DESVIO_IMAGENET).view(1, 3, 1, 1)
    return (lote - media) / desvio


def criar_cabeca() -> nn.Module:
    """
    Cabeça classificadora: Dropout + Linear(2048 -> 1).

    Usada tanto na ResNet50 completa (como 'fc') quanto no treino sobre as
    características extraídas, garantindo que os pesos treinados se encaixem
    exatamente no modelo completo.
    """
    return nn.Sequential(nn.Dropout(TAXA_DROPOUT), nn.Linear(NUM_CARACTERISTICAS, 1))


def criar_resnet50(pesos_imagenet: bool = True) -> nn.Module:
    """
    ResNet50 adaptada para classificação binária de pCR.

    - Entrada: mantém 3 canais (a 1ª camada e seus pesos pré-treinados são aproveitados).
      Exp. A: uma fase replicada nos 3 canais. Exp. B: pré, pós precoce e pós tardio.
    - Saída: a camada 'fc' original (2048 -> 1000 classes do ImageNet) é substituída
      por Dropout + Linear(2048 -> 1). A saída é um logit; sigmoid(logit) = P(pCR).
    - Congelamento: somente a nova 'fc' é treinável (exigência do enunciado:
      retreinar apenas a parte completamente conectada).
    """
    pesos = ResNet50_Weights.IMAGENET1K_V2 if pesos_imagenet else None
    modelo = resnet50(weights=pesos)
    modelo.fc = criar_cabeca()
    for nome, parametro in modelo.named_parameters():
        parametro.requires_grad = nome.startswith("fc.")
    return modelo


def contar_parametros(modelo: nn.Module) -> tuple[int, int]:
    """(parâmetros treináveis, parâmetros totais)."""
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    return treinaveis, total


@torch.no_grad()
def extrair_caracteristicas_lote(modelo: nn.Module, lote: torch.Tensor) -> torch.Tensor:
    """
    Passa um lote pelo backbone congelado e devolve os vetores (N, 2048), sem a 'fc'.

    O modelo é colocado em modo de avaliação (eval): congelar os pesos não basta,
    pois em modo de treino as camadas de BatchNorm continuariam atualizando suas
    estatísticas. Em eval, a mesma imagem gera sempre o mesmo vetor — premissa da
    extração antecipada de características.
    """
    modelo.eval()
    x = modelo.conv1(lote)      # convolução 7x7, 64 filtros, passo 2   -> 128 x 128
    x = modelo.bn1(x)
    x = modelo.relu(x)
    x = modelo.maxpool(x)       # max pooling, passo 2                  -> 64 x 64
    x = modelo.layer1(x)        # 3 blocos residuais,  256 canais       -> 64 x 64
    x = modelo.layer2(x)        # 4 blocos residuais,  512 canais       -> 32 x 32
    x = modelo.layer3(x)        # 6 blocos residuais, 1024 canais       -> 16 x 16
    x = modelo.layer4(x)        # 3 blocos residuais, 2048 canais       ->  8 x 8
    x = modelo.avgpool(x)       # média global de cada canal            ->  1 x 1
    return torch.flatten(x, 1)  # (N, 2048)


# =============================================================================
# 5. EXTRAÇÃO ANTECIPADA DE CARACTERÍSTICAS
# =============================================================================
# Com o backbone congelado, a mesma imagem gera sempre o mesmo vetor de 2048
# características (verificado). Por isso cada imagem passa pela rede UMA vez, os
# vetores são salvos, e a cabeça (Dropout + Linear) é treinada sobre eles.
# Isso equivale a treinar a ResNet50 inteira com o backbone congelado.

PASTA_RESULTADOS = "/content/resultados"
PARTICOES = ("treino", "validacao", "teste")

# Configurações de entrada: nome -> (experimento, fase usada no Exp. A).
# Todas são extraídas na mesma passagem pelos dados.
CONFIGURACOES_ENTRADA = {
    "B_tres_fases": ("B", None),
    "A_pre": ("A", "pre"),
    "A_post_early": ("A", "post_early"),
    "A_post_late": ("A", "post_late"),
}

_NOME_VARIACAO = dict(zip(VARIACOES_AUMENTO, NOMES_VARIACOES))


def _iterar_imagens_paciente(paciente: dict, aumentar: bool):
    """
    Gera (imagem (3, H, W), registro) para todos os cortes do paciente (Estratégia 2).

    No treino, cada corte gera as 10 variações; fora dele, só a imagem original.
    O registro marca o corte de maior área tumoral ('maior_area'), o que permite
    obter a Estratégia 1 depois, apenas filtrando as linhas — a máscara é usada
    só para essa marcação, nunca como entrada da rede.
    """
    cortes, indices_z = cortes_estrategia_todos(paciente["normalizados"])
    areas = area_tumoral_por_corte(paciente["mascara"])
    z_maior_area = selecionar_corte_maior_area(paciente["mascara"])
    variacoes = VARIACOES_AUMENTO if aumentar else ((0, False),)
    for z in indices_z:
        for angulo, espelhar in variacoes:
            registro = {
                "pid": paciente["pid"],
                "particao": paciente["particao"],
                "pCR": paciente["pCR"],
                "z": int(z),
                "variacao": _NOME_VARIACAO[(angulo, espelhar)],
                "original": angulo == 0 and not espelhar,
                "area_tumor": int(areas[z]),          # área no corte original
                "maior_area": bool(z == z_maior_area),
            }
            yield aplicar_variacao(cortes[z], angulo, espelhar), registro


# -----------------------------------------------------------------------------
# 5.1 Variante com recorte da região tumoral — EXPERIMENTO COMPLEMENTAR, USA A MÁSCARA
# -----------------------------------------------------------------------------
# Motivação (Células D e E): com a imagem inteira, a média global da ResNet é
# dominada por mama, fundo e características do aparelho; o tumor ocupa ~1-2% dos
# pixels. Aqui um quadrado ao redor do tumor é recortado e ampliado para 224 x 224
# (tamanho usado no treino do ImageNet), para que o tumor ocupe boa parte da entrada.
# A ResNet50 continua congelada: muda apenas a imagem mostrada a ela.
#
# A caixa é calculada sobre a PROJEÇÃO da máscara 3D (união de todos os cortes), então
# é a mesma região para todos os cortes do paciente. Só os cortes com tumor são usados:
# Estratégia 1 = corte de maior área; Estratégia 2 = média dos cortes com tumor.
# No aumento de dados, a projeção da máscara recebe a mesma rotação/espelhamento da
# imagem, e a caixa é recalculada, acompanhando o tumor.

TAMANHO_RECORTE = 224        # lado da imagem entregue à ResNet (em pixels)
MARGEM_RECORTE = 0.25        # margem em cada lado, como fração da extensão do tumor
LADO_MINIMO_RECORTE = 32     # evita ampliar demais tumores muito pequenos


def caixa_tumoral(
    mascara_2d: np.ndarray, margem: float = MARGEM_RECORTE, lado_minimo: int = LADO_MINIMO_RECORTE
) -> tuple[int, int, int]:
    """
    Quadrado (y0, x0, lado) ao redor do tumor em uma máscara 2D (H, W).

    O lado é a maior extensão do tumor (altura ou largura) mais a margem dos dois
    lados, com mínimo 'lado_minimo'. Se o quadrado passar da borda da imagem, ele é
    deslocado para dentro (não encolhido), mantendo o mesmo tamanho.
    """
    linhas = np.flatnonzero(mascara_2d.any(axis=1))
    colunas = np.flatnonzero(mascara_2d.any(axis=0))
    if len(linhas) == 0:
        raise ValueError("Máscara 2D sem tumor: não há caixa para recortar.")
    altura, largura = mascara_2d.shape
    extensao = max(linhas[-1] - linhas[0] + 1, colunas[-1] - colunas[0] + 1)
    lado = min(int(np.ceil(max(extensao * (1 + 2 * margem), lado_minimo))), altura, largura)
    centro_y = (linhas[0] + linhas[-1] + 1) / 2
    centro_x = (colunas[0] + colunas[-1] + 1) / 2
    y0 = min(max(int(round(centro_y - lado / 2)), 0), altura - lado)
    x0 = min(max(int(round(centro_x - lado / 2)), 0), largura - lado)
    return y0, x0, lado


def recortar_regiao(imagem: np.ndarray, caixa: tuple[int, int, int], tamanho: int = TAMANHO_RECORTE) -> np.ndarray:
    """Recorta a caixa de uma imagem (C, H, W) e amplia para (C, tamanho, tamanho) com interpolação bilinear."""
    y0, x0, lado = caixa
    recorte = imagem[:, y0:y0 + lado, x0:x0 + lado]
    fator = tamanho / lado
    ampliado = ndimage.zoom(recorte, (1, fator, fator), order=1)
    if ampliado.shape[1:] != (tamanho, tamanho):
        raise ValueError(f"Recorte com tamanho inesperado: {ampliado.shape}")
    return np.clip(ampliado, 0.0, 1.0).astype(np.float32)


def _iterar_imagens_paciente_recorte(paciente: dict, aumentar: bool):
    """
    Igual a _iterar_imagens_paciente, mas: (1) só cortes com tumor; (2) cada imagem é
    o recorte da região tumoral ampliado para TAMANHO_RECORTE. Registro ganha 'lado_caixa'.
    """
    mascara = paciente["mascara"] > 0
    areas = area_tumoral_por_corte(mascara)
    z_maior_area = selecionar_corte_maior_area(mascara)
    cortes, _ = cortes_estrategia_todos(paciente["normalizados"])
    projecao = mascara.any(axis=EIXO_Z).astype(np.float32)[np.newaxis]   # (1, H, W)
    variacoes = VARIACOES_AUMENTO if aumentar else ((0, False),)

    # A caixa depende só da variação (não do corte): calcula uma vez por variação.
    caixas = {}
    for angulo, espelhar in variacoes:
        projecao_variada = aplicar_variacao(projecao, angulo, espelhar, ordem_interpolacao=0)[0] > 0.5
        caixas[(angulo, espelhar)] = caixa_tumoral(projecao_variada)

    for z in np.flatnonzero(areas > 0):
        for angulo, espelhar in variacoes:
            caixa = caixas[(angulo, espelhar)]
            registro = {
                "pid": paciente["pid"],
                "particao": paciente["particao"],
                "pCR": paciente["pCR"],
                "z": int(z),
                "variacao": _NOME_VARIACAO[(angulo, espelhar)],
                "original": angulo == 0 and not espelhar,
                "area_tumor": int(areas[z]),
                "maior_area": bool(z == z_maior_area),
                "lado_caixa": int(caixa[2]),
            }
            yield recortar_regiao(aplicar_variacao(cortes[z], angulo, espelhar), caixa), registro


def extrair_caracteristicas_particao(
    modelo: nn.Module,
    caminhos: CaminhosDataset,
    metadados: pd.DataFrame,
    indices_por_paciente: dict[str, list[int]],
    particao: str,
    configuracoes: dict = CONFIGURACOES_ENTRADA,
    tamanho_lote: int = 64,
    dispositivo: torch.device | None = None,
    recorte_tumoral: bool = False,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, float]:
    """
    Extrai as características de TODOS os cortes dos pacientes de UMA partição.

    - Os pacientes vêm exclusivamente da coluna de partição do CSV.
    - O aumento de dados é decidido aqui (somente se particao == "treino"), não
      por parâmetro, para que validação e teste nunca recebam imagens transformadas.
    - Um paciente por vez é carregado; as imagens são agrupadas em lotes de
      'tamanho_lote' e cada lote passa pela rede uma vez por configuração.

    - recorte_tumoral=True usa a variante da seção 5.1 (recorte da região do tumor,
      só cortes com tumor). O padrão (False) mantém o comportamento original.

    Retorna ({nome_configuracao: matriz (N, 2048)}, tabela de registros (N linhas),
    duração em segundos). A linha i dos registros descreve a linha i de cada matriz.
    """
    if particao not in PARTICOES:
        raise ValueError(f"Partição desconhecida: {particao}")
    aumentar = particao == "treino"
    dispositivo = dispositivo or obter_dispositivo()
    modelo = modelo.to(dispositivo)
    pacientes = metadados[metadados["particao"] == particao]

    iterar_imagens = _iterar_imagens_paciente_recorte if recorte_tumoral else _iterar_imagens_paciente
    blocos = {nome: [] for nome in configuracoes}
    registros, imagens_pendentes, registros_pendentes = [], [], []

    def processar_pendentes() -> None:
        imagens = np.stack(imagens_pendentes)
        for nome, (experimento, fase) in configuracoes.items():
            entrada = montar_canais(imagens, experimento, fase or FASE_EXPERIMENTO_A)
            lote = preparar_lote(entrada).to(dispositivo)
            blocos[nome].append(extrair_caracteristicas_lote(modelo, lote).cpu().numpy())
        registros.extend(registros_pendentes)
        imagens_pendentes.clear()
        registros_pendentes.clear()

    inicio = time.time()
    for numero, (_, linha) in enumerate(pacientes.iterrows(), start=1):
        paciente = carregar_paciente_normalizado(caminhos, linha, indices_por_paciente[linha["pid"]])
        for imagem, registro in iterar_imagens(paciente, aumentar):
            imagens_pendentes.append(imagem)
            registros_pendentes.append(registro)
            if len(imagens_pendentes) == tamanho_lote:
                processar_pendentes()
        del paciente
        if numero % 20 == 0 or numero == len(pacientes):
            print(f"  [{particao}] {numero}/{len(pacientes)} pacientes ({time.time() - inicio:.0f} s)")
    if imagens_pendentes:
        processar_pendentes()
    duracao = time.time() - inicio

    caracteristicas = {nome: np.concatenate(partes) for nome, partes in blocos.items()}
    tabela_registros = pd.DataFrame(registros)

    # Verificações de consistência da partição extraída
    if set(tabela_registros["pid"]) != set(pacientes["pid"]):
        raise ValueError(f"{particao}: pacientes extraídos diferem dos pacientes da partição")
    if (tabela_registros["particao"] != particao).any():
        raise ValueError(f"{particao}: há registros de outra partição")
    if not aumentar and not tabela_registros["original"].all():
        raise ValueError(f"{particao}: imagens aumentadas fora do treino")
    for nome, matriz in caracteristicas.items():
        if matriz.shape != (len(tabela_registros), NUM_CARACTERISTICAS) or not np.isfinite(matriz).all():
            raise ValueError(f"{particao}/{nome}: matriz inválida {matriz.shape}")
    return caracteristicas, tabela_registros, duracao


def _pasta_caracteristicas(pasta: str) -> str:
    return os.path.join(pasta, "caracteristicas")


def salvar_caracteristicas(
    caracteristicas: dict[str, np.ndarray], registros: pd.DataFrame, particao: str, pasta: str = PASTA_RESULTADOS
) -> None:
    """Salva uma matriz .npy por configuração e um CSV de registros por partição."""
    destino = _pasta_caracteristicas(pasta)
    os.makedirs(destino, exist_ok=True)
    registros.to_csv(os.path.join(destino, f"registros_{particao}.csv"), index=False)
    for nome, matriz in caracteristicas.items():
        np.save(os.path.join(destino, f"{nome}_{particao}.npy"), matriz.astype(np.float32))


def carregar_caracteristicas(
    nome_configuracao: str, particao: str, pasta: str = PASTA_RESULTADOS, liberar_teste: bool = False
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Carrega (matriz (N, 2048), registros) de uma configuração e partição.

    Proteção contra vazamento: o conjunto de teste só é carregado com
    liberar_teste=True, o que deve acontecer apenas na avaliação final — nunca
    durante o treino ou a escolha de hiperparâmetros.
    """
    if particao == "teste" and not liberar_teste:
        raise PermissionError("Conjunto de teste bloqueado: use liberar_teste=True apenas na avaliação final.")
    origem = _pasta_caracteristicas(pasta)
    matriz = np.load(os.path.join(origem, f"{nome_configuracao}_{particao}.npy"))
    registros = pd.read_csv(os.path.join(origem, f"registros_{particao}.csv"))
    if len(matriz) != len(registros):
        raise ValueError(f"{nome_configuracao}/{particao}: {len(matriz)} vetores para {len(registros)} registros")
    return matriz, registros


# =============================================================================
# 6. TREINO DA CABEÇA CLASSIFICADORA
# =============================================================================
# Treina somente Dropout + Linear(2048 -> 1) sobre as características salvas.
# Estratégia 1: só o corte de maior área tumoral de cada paciente (e suas variações).
# Estratégia 2: todos os cortes; a previsão do paciente é a média dos cortes.

ESTRATEGIAS = {1: "maior_area", 2: "todos_cortes_media"}


@dataclass
class HiperparametrosCabeca:
    taxa_aprendizado: float = 1e-3   # passo do Adam
    decaimento_pesos: float = 1e-4   # weight decay: penaliza pesos grandes (regularização)
    tamanho_lote: int = 256          # vetores por atualização dos pesos
    max_epocas: int = 200            # limite de segurança; o early stopping costuma parar antes
    paciencia: int = 20              # épocas sem melhora na validação antes de parar


def selecionar_linhas_estrategia(registros: pd.DataFrame, estrategia: int) -> np.ndarray:
    """Máscara booleana das linhas usadas pela estratégia (1 = maior área, 2 = todos os cortes)."""
    if estrategia == 1:
        return registros["maior_area"].to_numpy(dtype=bool)
    if estrategia == 2:
        return np.ones(len(registros), dtype=bool)
    raise ValueError(f"Estratégia desconhecida: {estrategia} (use {list(ESTRATEGIAS)})")


def pesos_por_paciente(registros: pd.DataFrame) -> np.ndarray:
    """
    Peso de cada linha = 1 / nº de linhas do mesmo paciente.

    Assim cada paciente soma 1 na loss, independentemente de quantos cortes
    (10 a 72) ou variações ele tenha. Sem isso, pacientes com tumores maiores
    dominariam o treino.
    """
    linhas_do_paciente = registros.groupby("pid")["pid"].transform("size").to_numpy()
    return (1.0 / linhas_do_paciente).astype(np.float32)


def calcular_pos_weight(registros_treino: pd.DataFrame) -> float:
    """pos_weight = nº de pacientes negativos / nº de positivos, SOMENTE do treino."""
    rotulos = registros_treino.groupby("pid")["pCR"].first()
    positivos = int(rotulos.sum())
    return (len(rotulos) - positivos) / positivos


def _tensores(caracteristicas: np.ndarray, registros: pd.DataFrame, dispositivo: torch.device):
    x = torch.from_numpy(np.ascontiguousarray(caracteristicas, dtype=np.float32)).to(dispositivo)
    y = torch.tensor(registros["pCR"].to_numpy(), dtype=torch.float32, device=dispositivo)
    pesos = torch.from_numpy(pesos_por_paciente(registros)).to(dispositivo)
    return x, y, pesos


def treinar_cabeca(
    caracteristicas_treino: np.ndarray,
    registros_treino: pd.DataFrame,
    caracteristicas_validacao: np.ndarray,
    registros_validacao: pd.DataFrame,
    hiperparametros: HiperparametrosCabeca = HiperparametrosCabeca(),
    dispositivo: torch.device | None = None,
) -> tuple[nn.Module, pd.DataFrame, dict]:
    """
    Treina a cabeça com Adam e BCEWithLogitsLoss (pos_weight do treino, pesos por
    paciente) e early stopping pela loss de validação. Devolve a cabeça com os pesos
    da MELHOR época, o histórico por época e um resumo do treino.

    Somente treino e validação entram aqui; o teste não é recebido por esta função.
    """
    if (registros_treino["particao"] != "treino").any() or (registros_validacao["particao"] != "validacao").any():
        raise ValueError("treinar_cabeca aceita somente dados de treino e de validação.")
    if set(registros_treino["pid"]) & set(registros_validacao["pid"]):
        raise ValueError("Há pacientes presentes no treino e na validação.")

    dispositivo = dispositivo or obter_dispositivo()
    definir_semente()
    cabeca = criar_cabeca().to(dispositivo)
    pos_weight = calcular_pos_weight(registros_treino)
    # reduction="none": a loss de cada linha é ponderada pelo peso do seu paciente
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dispositivo), reduction="none")
    otimizador = torch.optim.Adam(
        cabeca.parameters(), lr=hiperparametros.taxa_aprendizado, weight_decay=hiperparametros.decaimento_pesos
    )

    x_treino, y_treino, pesos_treino = _tensores(caracteristicas_treino, registros_treino, dispositivo)
    x_val, y_val, pesos_val = _tensores(caracteristicas_validacao, registros_validacao, dispositivo)
    gerador = torch.Generator().manual_seed(SEMENTE)   # embaralhamento reproduzível

    historico, melhor = [], {"perda": float("inf"), "epoca": 0, "estado": None}
    inicio = time.time()
    for epoca in range(1, hiperparametros.max_epocas + 1):
        # ---- Treino: dropout ativo, pesos atualizados lote a lote ----
        cabeca.train()
        ordem = torch.randperm(len(x_treino), generator=gerador).to(dispositivo)
        soma_perdas, soma_pesos = 0.0, 0.0
        for inicio_lote in range(0, len(ordem), hiperparametros.tamanho_lote):
            indices = ordem[inicio_lote:inicio_lote + hiperparametros.tamanho_lote]
            perdas = criterio(cabeca(x_treino[indices]).squeeze(1), y_treino[indices])
            perda = (perdas * pesos_treino[indices]).sum() / pesos_treino[indices].sum()
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            soma_perdas += (perdas.detach() * pesos_treino[indices]).sum().item()
            soma_pesos += pesos_treino[indices].sum().item()

        # ---- Validação: sem dropout, sem atualizar pesos ----
        cabeca.eval()
        with torch.no_grad():
            perdas_val = criterio(cabeca(x_val).squeeze(1), y_val)
            perda_val = ((perdas_val * pesos_val).sum() / pesos_val.sum()).item()

        historico.append({"epoca": epoca, "perda_treino": soma_perdas / soma_pesos, "perda_validacao": perda_val})
        if perda_val < melhor["perda"]:
            melhor = {"perda": perda_val, "epoca": epoca, "estado": copy.deepcopy(cabeca.state_dict())}
        elif epoca - melhor["epoca"] >= hiperparametros.paciencia:
            break
    duracao = time.time() - inicio

    cabeca.load_state_dict(melhor["estado"])
    cabeca.eval()
    resumo = {
        "hiperparametros": asdict(hiperparametros),
        "taxa_dropout": TAXA_DROPOUT,
        "pos_weight": pos_weight,
        "melhor_epoca": melhor["epoca"],
        "melhor_perda_validacao": melhor["perda"],
        "epocas_executadas": len(historico),
        "segundos_treino": round(duracao, 2),
        "linhas_treino": len(registros_treino),
        "linhas_validacao": len(registros_validacao),
        "pacientes_treino": registros_treino["pid"].nunique(),
        "pacientes_validacao": registros_validacao["pid"].nunique(),
    }
    return cabeca, pd.DataFrame(historico), resumo


# =============================================================================
# 7. AVALIAÇÃO POR PACIENTE
# =============================================================================

@torch.no_grad()
def prever_probabilidades(
    cabeca: nn.Module, caracteristicas: np.ndarray, dispositivo: torch.device | None = None, tamanho_lote: int = 4096
) -> np.ndarray:
    """P(pCR) de cada linha (corte), aplicando a sigmoide ao logit da cabeça."""
    dispositivo = dispositivo or obter_dispositivo()
    cabeca = cabeca.to(dispositivo).eval()
    partes = []
    for inicio in range(0, len(caracteristicas), tamanho_lote):
        x = torch.from_numpy(np.ascontiguousarray(caracteristicas[inicio:inicio + tamanho_lote], dtype=np.float32))
        partes.append(torch.sigmoid(cabeca(x.to(dispositivo))).squeeze(1).cpu().numpy())
    return np.concatenate(partes)


def agregar_por_paciente(registros: pd.DataFrame, probabilidades: np.ndarray, estrategia: int) -> pd.DataFrame:
    """
    Uma previsão por paciente, usando somente imagens ORIGINAIS (sem aumento).

    Estratégia 1: probabilidade do corte de maior área tumoral.
    Estratégia 2: média das probabilidades de todos os cortes do paciente.
    """
    tabela = registros.assign(probabilidade=probabilidades)
    tabela = tabela[tabela["original"] & selecionar_linhas_estrategia(tabela, estrategia)]
    return (
        tabela.groupby("pid")
        .agg(particao=("particao", "first"), pCR=("pCR", "first"),
             probabilidade=("probabilidade", "mean"), cortes_usados=("z", "size"))
        .reset_index()
    )


def calcular_metricas(previsoes: pd.DataFrame, limiar: float = 0.5) -> dict:
    """
    Métricas no nível do paciente. Classe positiva = pCR (1).

    sensibilidade = VP / (VP + FN)  -> fração dos pCR reconhecidos
    especificidade = VN / (VN + FP) -> fração dos não-pCR reconhecidos
    precisão = VP / (VP + FP)       -> fração das previsões "pCR" que estavam certas
    F1 = 2 VP / (2 VP + FP + FN)    -> média harmônica de precisão e sensibilidade
    AUC: probabilidade de um pCR receber nota maior que um não-pCR (independe do limiar)
    """
    reais = previsoes["pCR"].to_numpy().astype(int)
    probabilidades = previsoes["probabilidade"].to_numpy()
    previstos = (probabilidades >= limiar).astype(int)
    vp = int(((previstos == 1) & (reais == 1)).sum())
    vn = int(((previstos == 0) & (reais == 0)).sum())
    fp = int(((previstos == 1) & (reais == 0)).sum())
    fn = int(((previstos == 0) & (reais == 1)).sum())

    def razao(numerador: float, denominador: float) -> float:
        return numerador / denominador if denominador > 0 else float("nan")

    return {
        "pacientes": len(reais),
        "limiar": limiar,
        "acuracia": razao(vp + vn, len(reais)),
        "precisao": razao(vp, vp + fp),
        "sensibilidade": razao(vp, vp + fn),
        "especificidade": razao(vn, vn + fp),
        "f1": razao(2 * vp, 2 * vp + fp + fn),
        "auc": float(roc_auc_score(reais, probabilidades)) if len(set(reais)) == 2 else float("nan"),
        "VP": vp, "VN": vn, "FP": fp, "FN": fn,
    }


def salvar_experimento(
    nome: str,
    cabeca: nn.Module,
    historico: pd.DataFrame,
    resumo_treino: dict,
    previsoes: dict[str, pd.DataFrame],
    metricas: dict[str, dict],
    pasta: str = PASTA_RESULTADOS,
) -> str:
    """Salva pesos da cabeça (melhor época), histórico, previsões e métricas em <pasta>/modelos/<nome>/."""
    destino = os.path.join(pasta, "modelos", nome)
    os.makedirs(destino, exist_ok=True)
    torch.save({"estado_cabeca": cabeca.state_dict(), "resumo_treino": resumo_treino},
               os.path.join(destino, "cabeca.pt"))
    historico.to_csv(os.path.join(destino, "historico.csv"), index=False)
    for particao, tabela in previsoes.items():
        tabela.to_csv(os.path.join(destino, f"previsoes_{particao}.csv"), index=False)
    with open(os.path.join(destino, "metricas.json"), "w") as arquivo:
        json.dump({"resumo_treino": resumo_treino, "metricas": metricas}, arquivo, indent=2)
    return destino


# =============================================================================
# 8. GRAD-CAM
# =============================================================================
# Mostra QUAIS regiões da imagem mais empurraram a saída da rede para "pCR".
# 1. Guarda a saída da layer4 (2048 mapas de 8 x 8; 7 x 7 no recorte 224 x 224).
# 2. Calcula o gradiente do logit em relação a esses mapas.
# 3. Peso de cada mapa = média do seu gradiente; mapa final = ReLU(soma ponderada).
# 4. Amplia o mapa para o tamanho da imagem e normaliza para [0, 1].
# Observação: como a cabeça é linear sobre a média global, o gradiente de cada mapa é
# constante (= peso da fc / nº de posições). Neste modelo o Grad-CAM coincide com o
# CAM clássico: soma dos mapas da layer4 ponderada pelos pesos da cabeça treinada.
# A ResNet continua congelada: o gradiente é só calculado, nenhum peso é alterado.

def montar_modelo_completo(caminho_cabeca: str, dispositivo: torch.device | None = None) -> nn.Module:
    """ResNet50 completa (backbone ImageNet congelado + cabeça treinada salva em cabeca.pt), em modo eval."""
    dispositivo = dispositivo or obter_dispositivo()
    modelo = criar_resnet50()
    salvo = torch.load(caminho_cabeca, map_location="cpu")
    modelo.fc.load_state_dict(salvo["estado_cabeca"])
    return modelo.to(dispositivo).eval()


def gerar_gradcam(
    modelo: nn.Module, imagem: np.ndarray, classe_alvo: int = 1, dispositivo: torch.device | None = None
) -> tuple[np.ndarray, float]:
    """
    Grad-CAM de UMA imagem (3, H, W) em [0, 1], com os canais já montados.

    classe_alvo=1: regiões que aumentam P(pCR); classe_alvo=0: regiões que a diminuem.
    Retorna (mapa (H, W) em [0, 1], P(pCR) prevista).
    """
    dispositivo = dispositivo or next(modelo.parameters()).device
    modelo.eval()
    capturado = {}

    def gancho(_modulo, _entrada, saida):
        saida.retain_grad()            # guarda o gradiente desse tensor intermediário
        capturado["ativacoes"] = saida

    alca = modelo.layer4.register_forward_hook(gancho)
    try:
        # Os pesos estão congelados; pedir gradiente na ENTRADA faz o PyTorch montar o grafo
        # até a layer4, sem treinar nada.
        lote = preparar_lote(imagem[np.newaxis]).to(dispositivo).requires_grad_(True)
        logit = modelo(lote).squeeze()
        modelo.zero_grad(set_to_none=True)
        (logit if classe_alvo == 1 else -logit).backward()
    finally:
        alca.remove()

    ativacoes = capturado["ativacoes"][0].detach()              # (2048, h, w)
    gradientes = capturado["ativacoes"].grad[0]                 # mesmo formato
    pesos = gradientes.mean(dim=(1, 2))                         # importância de cada mapa
    mapa = torch.relu((pesos[:, None, None] * ativacoes).sum(dim=0))
    mapa = torch.nn.functional.interpolate(
        mapa[None, None], size=imagem.shape[1:], mode="bilinear", align_corners=False
    )[0, 0]
    maximo = mapa.max()
    mapa = mapa / maximo if maximo > 0 else mapa
    return mapa.detach().cpu().numpy(), float(torch.sigmoid(logit).item())


def sobrepor_gradcam(eixo, canal: np.ndarray, mapa: np.ndarray, mascara: np.ndarray | None = None, titulo: str = "") -> None:
    """Desenha um canal em cinza, o Grad-CAM em cores por cima e (opcional) o contorno do tumor em branco."""
    eixo.imshow(canal, cmap="gray", vmin=0, vmax=1)
    eixo.imshow(mapa, cmap="jet", alpha=0.4, vmin=0, vmax=1)
    if mascara is not None and mascara.any():
        eixo.contour(mascara, levels=[0.5], colors="white", linewidths=0.8)
    eixo.set_title(titulo, fontsize=9)
    eixo.axis("off")
