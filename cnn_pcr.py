import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset


PASTA_DADOS = Path("base/BreastDCEDL_ISPY1_min_crop/BreastDCEDL_ISPY1_min_crop")
CSV = Path("base/BreastDCEDL_ISPY1_min_crop/BreastDCEDL_SPY1_min_crop_metadata.csv")
PASTA_CACHE = Path("cache")
PASTA_RESULTADOS = Path("resultados")

PARTICOES = {0: "treino", 1: "teste", 2: "validacao"}
TAMANHO = 128

ROTULO = "final"
USAR_TESTE = False
SEMENTE = 42

ANGULOS_GRAUS = (-20, -10, 0, 10, 20)

CONJUNTOS_FASES = [("pre",), ("pre", "post_early", "post_late")]

def carregar_metadados():
    df = pd.read_csv(CSV)
    df["particao"] = df["test"].map(PARTICOES)
    df["pCR"] = df["pCR"].astype(int)
    return df


def caminho_fase(linha, fase):
    numero = int(linha[fase])
    caminho = PASTA_DADOS / "dce" / f"{linha['pid']}_spy1_vis1_dce_aqc_{numero}.nii.gz"
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    return caminho


def caminho_mascara(pid):
    achados = list((PASTA_DADOS / "mask").glob(f"{pid}_*.nii.gz"))
    if len(achados) != 1:
        raise FileNotFoundError(f"{pid}: encontrado {len(achados)}")
    return achados[0]


def carregar_volume(caminho):
    return nib.load(caminho).get_fdata(dtype=np.float32)


def carregar_paciente(linha, fases=("pre",)):
    canais = [carregar_volume(caminho_fase(linha, fase)) for fase in fases]
    volume = np.stack(canais)
    mascara = carregar_volume(caminho_mascara(linha["pid"])) > 0
    if mascara.shape != volume.shape[1:]:
        raise ValueError(f"{linha['pid']}: máscara {mascara.shape} diferente do volume {volume.shape[1:]}")
    return volume, mascara


def selecionar_corte(mascara, estrategia):
    if estrategia == "central":
        return mascara.shape[0] // 2
    if estrategia == "maior_area":
        return int(mascara.sum(axis=(1, 2)).argmax())
    raise ValueError(f"Estratégia desconhecida: {estrategia}")


def redimensionar(corte, tamanho=TAMANHO):
    x = torch.from_numpy(np.ascontiguousarray(corte))
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(tamanho, tamanho), mode="area")
    return x.squeeze(0).numpy()


def normalizar(volume, percentil=99, maximo=3.0):
    referencia = float(np.percentile(volume[0], percentil))
    if referencia <= 0:
        referencia = float(volume[0].max()) or 1.0
    return np.clip(volume / referencia, 0.0, maximo).astype(np.float32)


def preparar_particao(df, particao, estrategia, fases=("pre",)):
    imagens, rotulos, pids, cortes = [], [], [], []
    linhas = df[df["particao"] == particao]

    for i, (_, linha) in enumerate(linhas.iterrows(), start=1):
        volume, mascara = carregar_paciente(linha, fases)


        volume = normalizar(volume)

        corte_selecioonado = selecionar_corte(mascara, estrategia)

        imagens.append(redimensionar(volume[:, corte_selecioonado]))
        rotulos.append(linha["pCR"])
        pids.append(linha["pid"])
        cortes.append(corte_selecioonado)

        print(f"{particao}: {i}/{len(linhas)} pacientes", end="\r")
    print()

    return {
        "imagens": np.stack(imagens).astype(np.float32),
        "rotulos": np.array(rotulos, dtype=np.float32),
        "pids": np.array(pids),
        "cortes": np.array(cortes),
    }


def salvar(dados, nome):
    PASTA_CACHE.mkdir(exist_ok=True)
    np.savez_compressed(PASTA_CACHE / f"{nome}.npz", **dados)


def carregar(nome):
    arquivo = np.load(PASTA_CACHE / f"{nome}.npz")
    return {chave: arquivo[chave] for chave in arquivo.files}


def gerar_caches():
    df = carregar_metadados()
    for fases in CONJUNTOS_FASES:
        sufixo = "" if fases == ("pre",) else "_" + "-".join(fases)
        for estrategia in ["central", "maior_area"]:
            for particao in ["treino", "validacao", "teste"]:
                dados = preparar_particao(df, particao, estrategia, fases)
                salvar(dados, f"{particao}_{estrategia}{sufixo}")
                print(estrategia, particao, fases, dados["imagens"].shape)


@dataclass
class Variacao:
    imagem: np.ndarray
    mascara: np.ndarray | None
    angulo: int
    espelhada: bool


def _girar(imagem, angulo, ordem):

    if angulo == 0:
        return np.array(imagem, copy=True)
    return ndimage.rotate(
        imagem,
        angle=float(angulo),
        reshape=False,
        order=ordem,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def aumentar_corte(imagem, mascara=None) -> list[Variacao]:

    imagem = np.asarray(imagem)
    if imagem.ndim != 2:
        raise ValueError("aumentar_corte recebe o corte 2D, nao o volume 3D.")
    if mascara is not None and np.asarray(mascara).shape != imagem.shape:
        raise ValueError("imagem e mascara precisam ter o mesmo formato.")

    variacoes = []
    for angulo in ANGULOS_GRAUS:
        for espelhada in (False, True):
            base = np.ascontiguousarray(imagem, dtype=np.float32)
            base_mascara = None if mascara is None else np.ascontiguousarray(mascara)
            if espelhada:
                base = np.ascontiguousarray(np.fliplr(base))
                if base_mascara is not None:
                    base_mascara = np.ascontiguousarray(np.fliplr(base_mascara))
            girada = _girar(base, angulo, ordem=1).astype(np.float32, copy=False)
            mascara_girada = None
            if base_mascara is not None:
                mascara_girada = _girar(base_mascara, angulo, ordem=0)
                mascara_girada = (mascara_girada > 0.5).astype(np.uint8)
            variacoes.append(Variacao(girada, mascara_girada, int(angulo), espelhada))
    return variacoes


def aumento_adaptado(imagem, variacao):
    angulo = ANGULOS_GRAUS[variacao // 2]
    espelhada = variacao % 2 == 1

    canais = []
    for canal in imagem.numpy():
        base = np.ascontiguousarray(canal, dtype=np.float32)
        if espelhada:
            base = np.ascontiguousarray(np.fliplr(base))
        canais.append(_girar(base, angulo, ordem=1).astype(np.float32, copy=False))
    return torch.from_numpy(np.stack(canais))


class DatasetPCR(Dataset):

    def __init__(self, dados, aumento=None, n_variacoes=1):
        self.imagens = torch.from_numpy(dados["imagens"])
        self.rotulos = torch.from_numpy(dados["rotulos"])
        self.aumento = aumento
        self.n_variacoes = n_variacoes

    def __len__(self):
        return len(self.rotulos) * self.n_variacoes

    def __getitem__(self, i):
        indice, variacao = divmod(i, self.n_variacoes)
        imagem = self.imagens[indice]
        if self.aumento is not None:
            imagem = self.aumento(imagem, variacao)
        return imagem, self.rotulos[indice], indice


def criar_loaders(estrategia, fases=("pre",), lote=16, aumento=None, n_variacoes=1):
    sufixo = "" if tuple(fases) == ("pre",) else "_" + "-".join(fases)
    treino = DatasetPCR(carregar(f"treino_{estrategia}{sufixo}"), aumento, n_variacoes)
    validacao = DatasetPCR(carregar(f"validacao_{estrategia}{sufixo}"))
    teste = DatasetPCR(carregar(f"teste_{estrategia}{sufixo}"))

    gerador = torch.Generator().manual_seed(SEMENTE)
    loader_treino = DataLoader(treino, batch_size=lote, shuffle=True,
                               drop_last=True, generator=gerador)
    loader_validacao = DataLoader(validacao, batch_size=lote, shuffle=False)
    loader_teste = DataLoader(teste, batch_size=lote, shuffle=False)
    return loader_treino, loader_validacao, loader_teste


def bloco(canais_entrada, canais_saida, com_pool):
    camadas = [
        nn.Conv2d(canais_entrada, canais_saida, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(canais_saida),
        nn.ReLU(),
    ]
    if com_pool:
        camadas.append(nn.MaxPool2d(kernel_size=2))
    return nn.Sequential(*camadas)


FILTROS_BASE = (16, 32, 64, 128, 128)
FILTROS_ESTREITA = (8, 16, 32, 64, 64)
FILTROS_MUITO_ESTREITA = (4, 8, 16, 32, 32)


class CNNpCR(nn.Module):
    def __init__(self, canais_entrada=1, dropout=0.5, filtros=FILTROS_BASE):
        super().__init__()
        f1, f2, f3, f4, f5 = filtros
        self.bloco1 = bloco(canais_entrada, f1, com_pool=True)
        self.bloco2 = bloco(f1, f2, com_pool=True)
        self.bloco3 = bloco(f2, f3, com_pool=True)
        self.bloco4 = bloco(f3, f4, com_pool=True)
        self.bloco5 = bloco(f4, f5, com_pool=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.densa = nn.Linear(f5, 1)

    def forward(self, x):
        x = self.bloco1(x)
        x = self.bloco2(x)
        x = self.bloco3(x)
        x = self.bloco4(x)
        x = self.bloco5(x)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.densa(x)
        return x.squeeze(1)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


def fixar_semente():
    random.seed(SEMENTE)
    np.random.seed(SEMENTE)
    torch.manual_seed(SEMENTE)
    torch.cuda.manual_seed_all(SEMENTE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def escolher_dispositivo():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ram_atual_mb():
    return psutil.Process().memory_info().rss / 2**20


class MedidorRecursos:

    def __init__(self, dispositivo, intervalo=0.05):
        self.dispositivo = torch.device(dispositivo)
        self.usa_cuda = self.dispositivo.type == "cuda"
        self.intervalo = intervalo

    def _amostrar_ram(self):
        while not self._parar.wait(self.intervalo):
            self.ram_pico_mb = max(self.ram_pico_mb, ram_atual_mb())

    def __enter__(self):
        if self.usa_cuda:
            torch.cuda.synchronize(self.dispositivo)
            torch.cuda.reset_peak_memory_stats(self.dispositivo)
        self.ram_inicial_mb = ram_atual_mb()
        self.ram_pico_mb = self.ram_inicial_mb
        self._parar = threading.Event()
        self._thread = threading.Thread(target=self._amostrar_ram, daemon=True)
        self._thread.start()
        self._inicio = time.perf_counter()
        return self

    def __exit__(self, *excecao):
        if self.usa_cuda:
            torch.cuda.synchronize(self.dispositivo)
        self.tempo_s = time.perf_counter() - self._inicio
        self._parar.set()
        self._thread.join()
        self.ram_pico_mb = max(self.ram_pico_mb, ram_atual_mb())
        if self.usa_cuda:
            self.vram_pico_mb = torch.cuda.max_memory_allocated(self.dispositivo) / 2**20
            self.vram_reservada_mb = torch.cuda.max_memory_reserved(self.dispositivo) / 2**20
        else:
            self.vram_pico_mb = self.vram_reservada_mb = float("nan")
        return False

    def resumo(self, fase):
        return {
            f"{fase}_tempo_s": round(self.tempo_s, 2),
            f"{fase}_ram_pico_mb": round(self.ram_pico_mb, 1),
            f"{fase}_ram_aumento_mb": round(self.ram_pico_mb - self.ram_inicial_mb, 1),
            f"{fase}_vram_pico_mb": round(self.vram_pico_mb, 1),
            f"{fase}_vram_reservada_mb": round(self.vram_reservada_mb, 1),
        }

    def mostrar(self, fase):
        print(f"[{fase}] tempo {self.tempo_s:.1f}s | RAM pico {self.ram_pico_mb:.0f} MB "
              f"(+{self.ram_pico_mb - self.ram_inicial_mb:.0f} MB) | "
              f"VRAM pico {self.vram_pico_mb:.0f} MB (reservada {self.vram_reservada_mb:.0f} MB)")


def calcular_pos_weight(dataset):
    positivos = float(dataset.rotulos.sum())
    negativos = len(dataset.rotulos) - positivos
    return negativos / positivos


def treinar_uma_epoca(modelo, loader, criterio, otimizador, dispositivo):
    modelo.train()
    soma_perdas, n_imagens = 0.0, 0

    for imagens, rotulos, _ in loader:
        imagens = imagens.to(dispositivo)
        rotulos = rotulos.to(dispositivo)

        otimizador.zero_grad()
        logits = modelo(imagens)
        perda = criterio(logits, rotulos)
        perda.backward()
        otimizador.step()

        soma_perdas += perda.item() * len(rotulos)
        n_imagens += len(rotulos)

    return soma_perdas / n_imagens


@torch.no_grad()
def avaliar(modelo, loader, criterio, dispositivo):
    modelo.eval()
    soma_perdas, n_imagens = 0.0, 0
    todas_probs, todos_rotulos = [], []

    for imagens, rotulos, _ in loader:
        imagens = imagens.to(dispositivo)
        rotulos = rotulos.to(dispositivo)
        logits = modelo(imagens)
        soma_perdas += criterio(logits, rotulos).item() * len(rotulos)
        n_imagens += len(rotulos)
        todas_probs.append(torch.sigmoid(logits).cpu())
        todos_rotulos.append(rotulos.cpu())

    probabilidades = torch.cat(todas_probs).numpy()
    rotulos = torch.cat(todos_rotulos).numpy()
    auc = roc_auc_score(rotulos, probabilidades)
    return soma_perdas / n_imagens, auc, probabilidades, rotulos


def treinar(modelo, loader_treino, loader_val, criterio, otimizador, dispositivo,
            arquivo_modelo, epocas_max=150, paciencia=20, paciencia_taxa=7):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode="min", factor=0.1, patience=paciencia_taxa)
    historico = []
    melhor_perda, melhor_epoca = float("inf"), 0

    for epoca in range(1, epocas_max + 1):
        inicio = time.time()
        perda_treino = treinar_uma_epoca(modelo, loader_treino, criterio, otimizador, dispositivo)
        perda_val, auc_val, _, _ = avaliar(modelo, loader_val, criterio, dispositivo)
        scheduler.step(perda_val)
        taxa = otimizador.param_groups[0]["lr"]

        historico.append({"epoca": epoca, "perda_treino": perda_treino,
                          "perda_val": perda_val, "auc_val": auc_val, "taxa": taxa})

        marcador = ""
        if perda_val < melhor_perda:
            melhor_perda, melhor_epoca = perda_val, epoca
            torch.save(modelo.state_dict(), arquivo_modelo)
            marcador = "  <- melhor"

        print(f"época {epoca:3d} | perda treino {perda_treino:.4f} | perda val {perda_val:.4f} | "
              f"AUC val {auc_val:.3f} | taxa {taxa:.0e} | {time.time() - inicio:.1f}s{marcador}")

        if epoca - melhor_epoca >= paciencia:
            print(f"Early stopping: {paciencia} épocas sem melhora na perda de validação.")
            break

    modelo.load_state_dict(torch.load(arquivo_modelo, map_location=dispositivo))
    return pd.DataFrame(historico), melhor_epoca


def plotar_curvas(historico, melhor_epoca, arquivo, mostrar=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(historico["epoca"], historico["perda_treino"], color="#2a78d6", lw=2, label="Treino")
    ax1.plot(historico["epoca"], historico["perda_val"], color="#eb6834", lw=2, label="Validação")
    ax1.set(title="Perda por época", xlabel="Época", ylabel="Perda (BCE com peso)")
    ax1.legend(frameon=False)
    ax2.plot(historico["epoca"], historico["auc_val"], color="#eb6834", lw=2)
    ax2.set(title="AUC de validação", xlabel="Época", ylabel="AUC")
    for ax in (ax1, ax2):
        ax.axvline(melhor_epoca, color="#8a8a8a", ls="--", lw=1)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=120)
    if mostrar:
        plt.show()
    plt.close(fig)


def metricas(rotulos, probabilidades, limiar=0.5):
    rotulos = np.asarray(rotulos).astype(int)
    previsoes = (np.asarray(probabilidades) >= limiar).astype(int)

    vp = int(((previsoes == 1) & (rotulos == 1)).sum())
    vn = int(((previsoes == 0) & (rotulos == 0)).sum())
    fp = int(((previsoes == 1) & (rotulos == 0)).sum())
    fn = int(((previsoes == 0) & (rotulos == 1)).sum())

    sensibilidade = vp / (vp + fn)
    especificidade = vn / (vn + fp)
    precisao = vp / (vp + fp) if vp + fp > 0 else 0.0
    f1 = (2 * precisao * sensibilidade / (precisao + sensibilidade)
          if precisao + sensibilidade > 0 else 0.0)
    return {
        "limiar": limiar,
        "auc": roc_auc_score(rotulos, probabilidades),
        "acuracia": (vp + vn) / len(rotulos),
        "sensibilidade": sensibilidade,
        "especificidade": especificidade,
        "precisao": precisao,
        "f1": f1,
        "acuracia_balanceada": (sensibilidade + especificidade) / 2,
        "VP": vp, "FN": fn, "VN": vn, "FP": fp,
    }


def limiar_youden(rotulos, probabilidades):
    taxa_fp, taxa_vp, limiares = roc_curve(rotulos, probabilidades)
    melhor = np.argmax(taxa_vp - taxa_fp)
    return float(min(limiares[melhor], 1.0))


def mostrar(nome, m):
    print(f"{nome}")
    print(f"  AUC {m['auc']:.3f} | acurácia {m['acuracia']:.3f} | "
          f"sensibilidade {m['sensibilidade']:.3f} | especificidade {m['especificidade']:.3f} | "
          f"precisão {m['precisao']:.3f} | F1 {m['f1']:.3f} | "
          f"acurácia balanceada {m['acuracia_balanceada']:.3f} | limiar {m['limiar']:.3f}")
    print(f"  VP {m['VP']}  FN {m['FN']}  VN {m['VN']}  FP {m['FP']}")


def plotar_roc(rotulos, probabilidades, limiar, arquivo, titulo):
    taxa_fp, taxa_vp, _ = roc_curve(rotulos, probabilidades)
    m = metricas(rotulos, probabilidades, limiar)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(taxa_fp, taxa_vp, color="#eb6834", lw=2,
            label=f"Modelo (AUC = {m['auc']:.3f})".replace(".", ","))
    ax.plot([0, 1], [0, 1], color="#8a8a8a", ls="--", lw=1, label="Chute (AUC = 0,5)")
    ax.scatter([1 - m["especificidade"]], [m["sensibilidade"]], color="#2a78d6", s=60,
               edgecolor="white", linewidth=2, zorder=3,
               label=f"Limiar de Youden da validação ({limiar:.2f})".replace(".", ","))
    ax.set(xlabel="1 − especificidade", ylabel="Sensibilidade", title=titulo,
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def avaliacao_final(modelo, loader_val, loader_teste, criterio, dispositivo):
    _, _, probs_val, rotulos_val = avaliar(modelo, loader_val, criterio, dispositivo)
    limiar = limiar_youden(rotulos_val, probs_val)
    _, _, probs_teste, rotulos_teste = avaliar(modelo, loader_teste, criterio, dispositivo)
    return {
        "limiar_05": metricas(rotulos_teste, probs_teste, 0.5),
        "limiar_youden": metricas(rotulos_teste, probs_teste, limiar),
        "probabilidades": probs_teste,
        "rotulos": rotulos_teste,
        "limiar": limiar,
    }


def escolher_melhor_configuracao(tabela):
    linha = tabela.loc[tabela["perda_val"].idxmin()]
    return linha["estrategia"], tuple(linha["fases"].split("-")), linha


def carregar_modelo_salvo(linha, fases, dispositivo):
    filtros = tuple(int(f) for f in str(linha["filtros"]).split("-"))
    modelo = CNNpCR(canais_entrada=len(fases), dropout=linha["dropout"], filtros=filtros)
    modelo = modelo.to(dispositivo)
    modelo.load_state_dict(torch.load(PASTA_RESULTADOS / f"{linha['nome']}.pt",
                                      map_location=dispositivo))
    modelo.eval()
    return modelo


def grad_cam(modelo, imagem, dispositivo, limiar=0.5):
    modelo.eval()
    x = imagem.unsqueeze(0).to(dispositivo)
    guardado = {}

    def guardar_ativacao(modulo, entrada, saida):
        saida.retain_grad()
        guardado["ativacao"] = saida

    gancho = modelo.bloco5.register_forward_hook(guardar_ativacao)
    with torch.enable_grad():
        logit = modelo(x)[0]
    gancho.remove()

    probabilidade = torch.sigmoid(logit).item()
    alvo = logit if probabilidade >= limiar else -logit
    modelo.zero_grad()
    alvo.backward()

    ativacao = guardado["ativacao"]
    gradiente = ativacao.grad
    pesos = gradiente.mean(dim=(2, 3), keepdim=True)
    mapa = F.relu((pesos * ativacao).sum(dim=1, keepdim=True))
    mapa = F.interpolate(mapa, size=x.shape[-2:], mode="bilinear", align_corners=False)
    mapa = mapa[0, 0].detach().cpu().numpy()
    if mapa.max() > 0:
        mapa = mapa / mapa.max()
    return mapa, probabilidade


def mascara_do_corte(pid, corte):
    mascara = carregar_volume(caminho_mascara(pid))[int(corte)] > 0
    return redimensionar(mascara[None].astype(np.float32))[0] > 0.5


def escolher_exemplos(rotulos, probabilidades, limiar):
    rotulos = np.asarray(rotulos).astype(int)
    previsoes = (np.asarray(probabilidades) >= limiar).astype(int)
    casos = {
        "VP": (previsoes == 1) & (rotulos == 1),
        "VN": (previsoes == 0) & (rotulos == 0),
        "FP": (previsoes == 1) & (rotulos == 0),
        "FN": (previsoes == 0) & (rotulos == 1),
    }
    exemplos = []
    for tipo, filtro in casos.items():
        indices = np.flatnonzero(filtro)
        if len(indices) > 0:
            confianca = np.abs(probabilidades[indices] - limiar)
            exemplos.append((tipo, int(indices[np.argmax(confianca)])))
    return exemplos


def plotar_grad_cam(modelo, dados_teste, exemplos, limiar, dispositivo, arquivo, titulo):
    fig, eixos = plt.subplots(len(exemplos), 2, figsize=(7, 3.4 * len(exemplos)), squeeze=False)
    nomes_casos = {"VP": "verdadeiro positivo", "VN": "verdadeiro negativo",
                   "FP": "falso positivo", "FN": "falso negativo"}
    for (tipo, i), (ax_img, ax_cam) in zip(exemplos, eixos):
        imagem = torch.from_numpy(dados_teste["imagens"][i])
        mapa, probabilidade = grad_cam(modelo, imagem, dispositivo, limiar)
        mascara = mascara_do_corte(dados_teste["pids"][i], dados_teste["cortes"][i])
        corte = dados_teste["imagens"][i][0]

        ax_img.imshow(corte, cmap="gray")
        ax_img.contour(mascara, levels=[0.5], colors="#2a78d6", linewidths=1)
        ax_img.set_title(f"{tipo} ({nomes_casos[tipo]})\n{dados_teste['pids'][i]}  |  "
                         f"P(pCR) = {probabilidade:.2f}", fontsize=9)
        ax_cam.imshow(corte, cmap="gray")
        ax_cam.imshow(mapa, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        ax_cam.contour(mascara, levels=[0.5], colors="white", linewidths=1)
        ax_cam.set_title("Grad-CAM (classe prevista)", fontsize=9)
        for ax in (ax_img, ax_cam):
            ax.axis("off")
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def analisar_melhor_configuracao(tabela):
    estrategia, fases, linha = escolher_melhor_configuracao(tabela)
    texto_fases = "-".join(fases)
    print(f"\nMelhor configuração pela validação: {estrategia}, {texto_fases}")

    dispositivo = escolher_dispositivo()
    sufixo = "" if fases == ("pre",) else "_" + "-".join(fases)
    dados_teste = carregar(f"teste_{estrategia}{sufixo}")
    loader_treino, loader_val, loader_teste = criar_loaders(estrategia, fases=fases)
    pos_weight = calcular_pos_weight(loader_treino.dataset)
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dispositivo))

    modelo = carregar_modelo_salvo(linha, fases, dispositivo)
    resultado = avaliacao_final(modelo, loader_val, loader_teste, criterio, dispositivo)
    rotulos, probabilidades, limiar = (resultado["rotulos"], resultado["probabilidades"],
                                       resultado["limiar"])

    base = f"{estrategia}_{texto_fases}_{ROTULO}"
    pd.DataFrame({
        "pid": dados_teste["pids"],
        "corte": dados_teste["cortes"],
        "pCR": rotulos.astype(int),
        "probabilidade": probabilidades.round(4),
        "limiar_youden": round(limiar, 4),
        "previsao": (probabilidades >= limiar).astype(int),
    }).to_csv(PASTA_RESULTADOS / f"previsoes_teste_{base}.csv", index=False)
    plotar_roc(rotulos, probabilidades, limiar, PASTA_RESULTADOS / f"roc_teste_{base}.png",
               f"Curva ROC no teste: {estrategia}, {texto_fases}")

    exemplos = escolher_exemplos(rotulos, probabilidades, limiar)
    plotar_grad_cam(modelo, dados_teste, exemplos, limiar, dispositivo,
                    PASTA_RESULTADOS / f"gradcam_teste_{base}.png",
                    f"Grad-CAM no teste: {estrategia}, {texto_fases} "
                    f"(limiar {limiar:.2f})".replace(".", ","))
    print(f"Salvos em {PASTA_RESULTADOS}/: previsoes_teste_{base}.csv, roc_teste_{base}.png, "
          f"gradcam_teste_{base}.png")


def nome_experimento(estrategia, fases=("pre",), lr=1e-4, wd=1e-3, dropout=0.5,
                     filtros=FILTROS_BASE):
    texto_fases = "-".join(fases)
    texto_filtros = "-".join(str(f) for f in filtros)
    return (f"cnn_{estrategia}_{texto_fases}_f{texto_filtros}_lr{lr:g}_wd{wd:g}"
            f"_do{dropout:g}_s{SEMENTE}_{ROTULO}")


def metricas_de_teste(resultado):
    linha = {}
    for tipo_limiar in ["limiar_05", "limiar_youden"]:
        for metrica in ["auc", "acuracia", "sensibilidade", "especificidade", "precisao", "f1",
                        "acuracia_balanceada", "limiar", "VP", "FN", "VN", "FP"]:
            linha[f"teste_{tipo_limiar}_{metrica}"] = resultado[tipo_limiar][metrica]
    return linha


def rodar_experimento(estrategia, fases=("pre",), lr=1e-4, wd=1e-3, dropout=0.5,
                      filtros=FILTROS_BASE, usar_teste=False):
    texto_fases = "-".join(fases)
    texto_filtros = "-".join(str(f) for f in filtros)
    nome = nome_experimento(estrategia, fases, lr, wd, dropout, filtros)
    print(f"\n=== {nome} ===")

    fixar_semente()
    dispositivo = escolher_dispositivo()
    loader_treino, loader_val, loader_teste = criar_loaders(
        estrategia, fases=fases, aumento=aumento_adaptado, n_variacoes=10)

    modelo = CNNpCR(canais_entrada=len(fases), dropout=dropout, filtros=filtros).to(dispositivo)
    print(f"Parâmetros: {contar_parametros(modelo):,}")
    pos_weight = calcular_pos_weight(loader_treino.dataset)
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dispositivo))
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=wd)

    with MedidorRecursos(dispositivo) as medidor_treino:
        historico, melhor_epoca = treinar(modelo, loader_treino, loader_val, criterio, otimizador,
                                          dispositivo, arquivo_modelo=PASTA_RESULTADOS / f"{nome}.pt")
    medidor_treino.mostrar("treino")
    historico.to_csv(PASTA_RESULTADOS / f"{nome}_historico.csv", index=False)
    plotar_curvas(historico, melhor_epoca, PASTA_RESULTADOS / f"{nome}_curvas.png", mostrar=False)

    melhor = historico[historico["epoca"] == melhor_epoca].iloc[0]
    linha = {"nome": nome, "estrategia": estrategia, "fases": texto_fases, "filtros": texto_filtros,
             "parametros": contar_parametros(modelo), "lr": lr, "wd": wd,
             "dropout": dropout, "semente": SEMENTE, "melhor_epoca": melhor_epoca,
             "perda_val": melhor["perda_val"], "auc_val": melhor["auc_val"],
             **medidor_treino.resumo("treino")}

    if usar_teste:
        with MedidorRecursos(dispositivo) as medidor_teste:
            resultado = avaliacao_final(modelo, loader_val, loader_teste, criterio, dispositivo)
        medidor_teste.mostrar("teste")
        linha.update(metricas_de_teste(resultado))
        linha.update(medidor_teste.resumo("teste"))
    return linha


def avaliar_experimento_no_teste(estrategia, fases=("pre",), lr=1e-4, wd=1e-3, dropout=0.5,
                                 filtros=FILTROS_BASE):
    nome = nome_experimento(estrategia, fases, lr, wd, dropout, filtros)
    dispositivo = escolher_dispositivo()
    loader_treino, loader_val, loader_teste = criar_loaders(estrategia, fases=fases)

    modelo = CNNpCR(canais_entrada=len(fases), dropout=dropout, filtros=filtros).to(dispositivo)
    modelo.load_state_dict(torch.load(PASTA_RESULTADOS / f"{nome}.pt", map_location=dispositivo))
    pos_weight = calcular_pos_weight(loader_treino.dataset)
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dispositivo))

    with MedidorRecursos(dispositivo) as medidor_teste:
        resultado = avaliacao_final(modelo, loader_val, loader_teste, criterio, dispositivo)
    print(nome)
    medidor_teste.mostrar("teste")
    return {"nome": nome, **metricas_de_teste(resultado), **medidor_teste.resumo("teste")}


def main():
    configuracoes = []
    for estrategia in ["central", "maior_area"]:
        for fases in CONJUNTOS_FASES:
            configuracoes.append({"estrategia": estrategia, "fases": fases,
                                  "filtros": FILTROS_ESTREITA})

    faltando = []
    for config in configuracoes:
        sufixo = "" if config["fases"] == ("pre",) else "_" + "-".join(config["fases"])
        for particao in ["treino", "validacao", "teste"]:
            arquivo = PASTA_CACHE / f"{particao}_{config['estrategia']}{sufixo}.npz"
            if not arquivo.exists():
                faltando.append(arquivo.name)
    if "--gerar-cache" in sys.argv or faltando:
        if faltando:
            print(f"Faltam {len(faltando)} arquivos de cache, por exemplo {faltando[0]}. Gerando o cache.")
        gerar_caches()

    PASTA_RESULTADOS.mkdir(exist_ok=True)

    arquivo_tabela = PASTA_RESULTADOS / f"tabela_experimentos_{ROTULO}.csv"
    anteriores = pd.read_csv(arquivo_tabela) if arquivo_tabela.exists() else pd.DataFrame()

    if "--avaliar-teste" in sys.argv:
        linhas = [avaliar_experimento_no_teste(**config) for config in configuracoes]
        teste = pd.DataFrame(linhas)
        colunas_antigas = [c for c in anteriores.columns if c.startswith("teste_")]
        tabela = anteriores.drop(columns=colunas_antigas).merge(teste, on="nome", how="inner")
        tabela.to_csv(arquivo_tabela, index=False)

        colunas = ["teste_limiar_05_auc", "teste_limiar_youden_acuracia",
                   "teste_limiar_youden_sensibilidade", "teste_limiar_youden_especificidade",
                   "teste_limiar_youden_precisao", "teste_limiar_youden_f1",
                   "teste_limiar_youden_acuracia_balanceada"]
        resumo = tabela.set_index(["estrategia", "fases"])[colunas].round(3)
        resumo.to_csv(PASTA_RESULTADOS / f"resumo_teste_{ROTULO}.csv")
        print(f"\nTeste: métricas por configuração (semente {SEMENTE})")
        print(resumo.to_string())

        colunas_recursos = ["teste_tempo_s", "teste_ram_pico_mb", "teste_vram_pico_mb"]
        print("\nTeste: tempo de execução e memória por configuração")
        print(tabela.set_index(["estrategia", "fases"])[colunas_recursos].round(1).to_string())

        analisar_melhor_configuracao(tabela)
        return

    resultados = []
    for config in configuracoes:
        resultados.append(rodar_experimento(**config, usar_teste=USAR_TESTE))
        tabela = pd.concat([anteriores, pd.DataFrame(resultados)], ignore_index=True)
        tabela = tabela.drop_duplicates(subset="nome", keep="last")
        tabela.to_csv(arquivo_tabela, index=False)

    atuais = pd.DataFrame(resultados)
    print(f"\nResultados (semente {SEMENTE}):")
    print(atuais[["estrategia", "fases", "filtros", "parametros", "melhor_epoca", "perda_val", "auc_val"]].round(3).to_string())
    colunas_recursos = ["treino_tempo_s", "treino_ram_pico_mb", "treino_vram_pico_mb"]
    print("\nTreino: tempo de execução e memória por configuração:")
    print(atuais.set_index(["estrategia", "fases"])[colunas_recursos].round(1).to_string())


if __name__ == "__main__":
    main()
