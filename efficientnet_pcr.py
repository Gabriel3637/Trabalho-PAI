"""Terceira rede do trabalho de PAI: EfficientNet-B0 para pCR no I-SPY1.

Execute ao lado de trabalho_pai.py. Importar o módulo não abre a interface.
Consulte o guia para a preparação da base, os experimentos e a demonstração.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torchvision
from matplotlib.figure import Figure
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

try:
    from trabalho_pai import (
        ANGULOS_GRAUS,
        Paciente,
        VolumeNifti,
        associar_arquivos,
        aumentar_corte,
        carregar_mascara,
        carregar_nifti,
        corte_axial,
        indice_maior_area,
        ler_metadados_ispy1,
        localizar_pastas,
        para_uint8,
    )
except ModuleNotFoundError as erro:
    if erro.name == "trabalho_pai":
        raise ModuleNotFoundError(
            "Coloque efficientnet_pcr.py ao lado do trabalho_pai.py extraído da interface."
        ) from erro
    raise


FASES = ("pre", "precoce", "tardio")
NOMES_FASES = {
    "pre": "Pré-contraste",
    "precoce": "Pós-contraste precoce",
    "tardio": "Pós-contraste tardio",
}
ESTRATEGIAS = ("central", "maior_area")
PARTICOES = {"treino": 0, "validacao": 2, "teste": 1}
PESOS_IMAGENET = EfficientNet_B0_Weights.IMAGENET1K_V1
VERSAO_CHECKPOINT = 1


@dataclass(frozen=True)
class ConfiguracaoPCR:
    """Parâmetros necessários para preparar a entrada e reconstruir o modelo."""

    fases: tuple[str, ...] = ("precoce",)
    estrategia: str = "central"
    tamanho: int = 224
    percentis: tuple[float, float] = (1.0, 99.5)
    media: tuple[float, ...] = (0.485, 0.456, 0.406)
    desvio: tuple[float, ...] = (0.229, 0.224, 0.225)
    limiar: float = 0.5
    radiologico: bool = True

    def __post_init__(self):
        if self.fases not in (("precoce",), FASES):
            raise ValueError("Use apenas precoce ou as fases pré, precoce e tardia nessa ordem.")
        if self.estrategia not in ESTRATEGIAS or self.tamanho < 32:
            raise ValueError("Use uma estratégia conhecida e tamanho de pelo menos 32 pixels.")
        baixo, alto = self.percentis
        if not 0 <= baixo < alto <= 100:
            raise ValueError("Os percentis devem ser crescentes e estar entre 0 e 100.")
        if not 0 < self.limiar < 1:
            raise ValueError("O limiar deve estar entre 0 e 1, excluindo os extremos.")


@dataclass(frozen=True)
class ExamePCR:
    """Exame para a demonstração, sem necessidade de rótulo ou CSV."""

    pid: str
    arquivos: dict[str, Path]
    mascara: Path | None = None


@dataclass
class CortesPCR:
    """Planos no formato e na orientação de exibição da interface."""

    z: int
    imagens: dict[str, np.ndarray]
    mascara: np.ndarray | None


@dataclass(frozen=True)
class ReferenciaAmostra:
    """Identificação mantida junto ao tensor, fora da entrada da rede."""

    pid: str
    pcr: int
    z: int
    aquisicoes: tuple[int, ...]


@dataclass
class ResultadoPCR:
    """Predição e mapas em coordenadas dos cortes devolvidos."""

    probabilidade: float
    classe_prevista: int
    classe_gradcam: int
    cortes: CortesPCR
    gradcam: dict[str, np.ndarray]


# 1. Metadados e volumes


def ler_csv(caminho: Path) -> dict[str, Paciente]:
    """Reusa a leitura da interface e exige um registro rotulado por PID."""
    inteiros = ("pCR", "test", "n_z", "pre", "post_early", "post_late")
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        faltantes = {"pid", "dataset", *inteiros} - set(leitor.fieldnames or ())
        if faltantes:
            raise ValueError(f"Colunas ausentes em {caminho.name}: {sorted(faltantes)}.")
        for linha in leitor:
            if (linha["dataset"] or "").strip().lower() != "spy1":
                continue
            pid = linha["pid"]
            for coluna in inteiros:
                try:
                    valor = float(linha[coluna])
                except (TypeError, ValueError) as erro:
                    raise ValueError(f"{pid}: valor ausente ou inválido na coluna {coluna}.") from erro
                if not valor.is_integer() or valor < 0:
                    raise ValueError(f"{pid}: a coluna {coluna} requer um inteiro não negativo.")
                if coluna == "test" and valor not in PARTICOES.values():
                    raise ValueError(f"{pid}: código test desconhecido ({valor}).")
                if coluna == "pCR" and valor not in (0, 1):
                    raise ValueError(f"{pid}: pCR deve ser 0 ou 1.")
    pacientes = {}
    for paciente in ler_metadados_ispy1(caminho):
        pid = paciente.pid
        if not pid or pid in pacientes:
            raise ValueError(f"PID ausente ou repetido no CSV: {pid!r}.")
        pacientes[pid] = paciente
    if not pacientes:
        raise ValueError(f"Nenhum paciente I-SPY1 com pCR em {caminho}.")
    return pacientes


def carregar_coorte(
    raiz: Path, csv_principal: Path, csv_fases: Path | None = None,
) -> list[Paciente]:
    """Preserva rótulos e partições do CSV principal e confere os arquivos."""
    pacientes = ler_csv(csv_principal)
    if csv_fases is not None:
        referencia = ler_csv(csv_fases)
        for pid, paciente in pacientes.items():
            ref = referencia.get(pid)
            if ref is None or (ref.pcr, ref.particao) != (paciente.pcr, paciente.particao):
                raise ValueError(f"{pid}: CSV de referência sem o mesmo rótulo e partição.")
            if paciente.aquisicoes != ref.aquisicoes:
                print(
                    f"{pid}: fases do CSV principal {paciente.aquisicoes} "
                    f"corrigidas pela referência MinCrop para {ref.aquisicoes}.",
                    file=sys.stderr,
                )
                paciente.aquisicoes = ref.aquisicoes.copy()

    pasta_dce, pasta_mask = localizar_pastas(raiz)
    associar_arquivos(pacientes.values(), pasta_dce, pasta_mask)
    problemas = []
    for paciente in pacientes.values():
        pid = paciente.pid
        indices = tuple(paciente.aquisicoes[fase] for fase in FASES)
        if not 0 <= indices[0] < indices[1] < indices[2]:
            problemas.append(
                f"{pid}: fases repetidas ou fora de ordem {indices}. "
                "Confira pre/post_early/post_late ou indique --csv-fases."
            )
        for fase, indice in zip(FASES, indices):
            nome = f"{pid}_spy1_vis1_dce_aqc_{indice}"
            problema = conferir_arquivo(pasta_dce, nome, paciente.arquivos[fase])
            if problema:
                problemas.append(f"{pid}, fase {fase}: {problema}")
        nome = f"{pid}_spy1_vis1_mask"
        problema = conferir_arquivo(pasta_mask, nome, paciente.mascara)
        if problema:
            problemas.append(f"{pid}, máscara: {problema}")
    if problemas:
        raise ValueError("Problemas de associação:\n" + "\n".join(problemas))
    return list(pacientes.values())


def conferir_arquivo(pasta: Path | None, nome: str, associado: Path | None) -> str:
    """Valida a associação por nome completo e aponta falta ou ambiguidade."""
    candidatos = []
    if pasta is not None:
        candidatos = [
            pasta / (nome + extensao)
            for extensao in (".nii", ".nii.gz")
            if (pasta / (nome + extensao)).is_file()
        ]
    if len(candidatos) != 1 or candidatos[0] != associado:
        return f"esperado um arquivo {nome}.nii[.gz]; associação encontrada: {associado}."
    return ""


def contagens_classes(pacientes: list[Paciente]) -> dict:
    """Conta pacientes únicos por código oficial e classe."""
    contagens = Counter((paciente.particao, paciente.pcr) for paciente in pacientes)
    resumo = {}
    for particao, codigo in PARTICOES.items():
        ausentes = contagens[particao, 0]
        presentes = contagens[particao, 1]
        resumo[particao] = {
            "codigo": codigo, "pcr_0": ausentes, "pcr_1": presentes,
            "total": ausentes + presentes,
        }
        print(f"{particao:10s} (test={codigo}): pCR 0 = {ausentes}, pCR 1 = {presentes}")
    return resumo


def conferir_grade(referencia: VolumeNifti, volume: VolumeNifti):
    """Exige a mesma grade de voxels para usar o mesmo índice Z."""
    if (
        referencia.dados.shape != volume.dados.shape
        or referencia.eixo_z != volume.eixo_z
        or not np.allclose(referencia.affine, volume.affine, rtol=0, atol=1e-4)
        or not np.allclose(referencia.zooms, volume.zooms, rtol=0, atol=1e-4)
    ):
        raise ValueError(
            f"grade de {volume.caminho.name} diverge de {referencia.caminho.name} "
            f"(formas {volume.dados.shape} e {referencia.dados.shape}); "
            "confira forma, eixo de corte, espaçamento e affine."
        )


def obter_plano(volume: VolumeNifti, z: int, radiologico: bool, mascara: bool = False):
    """Extrai o plano da interface, incluindo o caso MinCrop com affine identidade."""
    plano, orientacao_reconhecida = corte_axial(volume, z, radiologico, mascara)
    # No pacote MinCrop local, (Z, 256, 256) tem affine identidade.
    # A ordem de pixels é a da interface. O cabeçalho não confirma a orientação anatômica.
    mincrop = (
        volume.eixo_z == 0
        and volume.dados.shape[1:] == (256, 256)
        and np.allclose(volume.affine, np.eye(4), rtol=0, atol=1e-4)
    )
    if not orientacao_reconhecida and not mincrop:
        raise ValueError(f"orientação axial não reconhecida em {volume.caminho.name}.")
    if not np.isfinite(plano).all():
        raise ValueError(f"intensidades não finitas no corte de {volume.caminho.name}.")
    return plano


def selecionar_cortes(
    exame: Paciente | ExamePCR, config: ConfiguracaoPCR, mostrar_mascara: bool = False,
) -> CortesPCR:
    """Seleciona um Z comum, lê cada fase necessária e devolve planos alinhados."""
    try:
        for fase in config.fases:
            if not exame.arquivos.get(fase):
                raise ValueError(f"arquivo da fase {fase} ausente.")
        referencia = carregar_nifti(exame.arquivos["precoce"])
        mascara = None
        if config.estrategia == "maior_area" or (mostrar_mascara and exame.mascara):
            if exame.mascara is None:
                raise ValueError("máscara ausente para escolher a maior área.")
            mascara = carregar_mascara(exame.mascara)
            conferir_grade(referencia, mascara)
        if config.estrategia == "maior_area":
            if not mascara.dados.any():
                raise ValueError("máscara tumoral vazia.")
            z = indice_maior_area(mascara)
        else:
            z = referencia.n_cortes // 2

        imagens = {}
        for fase in config.fases:
            volume = referencia if fase == "precoce" else carregar_nifti(exame.arquivos[fase])
            conferir_grade(referencia, volume)
            imagens[fase] = obter_plano(volume, z, config.radiologico)
            del volume
        plano_mascara = None
        if mostrar_mascara and mascara is not None:
            plano_mascara = obter_plano(mascara, z, config.radiologico, mascara=True)
        return CortesPCR(z, imagens, plano_mascara)
    except Exception as erro:
        raise ValueError(f"{exame.pid}: {erro}") from erro


# 2. Pré-processamento e lotes


def normalizar_cortes(cortes: CortesPCR, config: ConfiguracaoPCR) -> dict[str, np.ndarray]:
    """Aplica a todas as fases a mesma faixa robusta, estimada no corte precoce."""
    precoce = cortes.imagens["precoce"]
    baixo, alto = np.percentile(precoce, config.percentis)
    if alto <= baixo:
        baixo, alto = float(precoce.min()), float(precoce.max())
    escala = float(alto - baixo)
    if escala <= 0:
        escala = 1.0
    return {
        fase: np.clip((imagem - float(baixo)) / escala, 0.0, 1.0).astype(np.float32)
        for fase, imagem in cortes.imagens.items()
    }


def preparar_entradas(
    cortes: CortesPCR, config: ConfiguracaoPCR, aumentar: bool,
) -> torch.Tensor:
    """Converte os planos em tensor float32 de forma (V, P, 3, T, T)."""
    normalizadas = normalizar_cortes(cortes, config)
    por_fase = []
    for fase in config.fases:
        imagem = normalizadas[fase]
        if aumentar:
            # Em todas as fases, o mesmo índice equivale ao mesmo espelho e ângulo.
            variacoes = [variacao.imagem for variacao in aumentar_corte(imagem)]
        else:
            variacoes = [imagem]
        por_fase.append(np.stack(variacoes))
    planos = np.stack(por_fase, axis=1)  # V, P, H, W
    n_variacoes, n_fases, altura, largura = planos.shape
    cinza = torch.from_numpy(planos).reshape(n_variacoes * n_fases, 1, altura, largura)
    redimensionado = F.interpolate(
        cinza, (config.tamanho, config.tamanho),
        mode="bilinear", align_corners=False, antialias=True,
    )
    media = redimensionado.new_tensor(config.media).view(1, 3, 1, 1)
    desvio = redimensionado.new_tensor(config.desvio).view(1, 3, 1, 1)
    rgb = (redimensionado.expand(-1, 3, -1, -1) - media) / desvio
    return rgb.reshape(n_variacoes, n_fases, 3, config.tamanho, config.tamanho)


def lotes_preparados(
    pacientes: list[Paciente], config: ConfiguracaoPCR, aumentar: bool, lote: int,
):
    """Gera lotes de imagens e referências, lendo um exame por vez."""
    imagens, referencias = [], []
    for paciente in pacientes:
        cortes = selecionar_cortes(paciente, config)
        entrada = preparar_entradas(cortes, config, aumentar)
        referencia = ReferenciaAmostra(
            paciente.pid, paciente.pcr, cortes.z,
            tuple(paciente.aquisicoes[fase] for fase in config.fases),
        )
        for exemplo in entrada.unbind(0):
            imagens.append(exemplo)
            referencias.append(referencia)
            if len(imagens) == lote:
                yield torch.stack(imagens), referencias
                imagens, referencias = [], []
    if imagens:
        yield torch.stack(imagens), referencias


# 3. Uma EfficientNet-B0 compartilhada


class EfficientNetPCR(nn.Module):
    """Extrator congelado, média espacial e uma camada linear binária."""

    def __init__(self, n_fases: int, imagenet: bool):
        super().__init__()
        base = efficientnet_b0(weights=PESOS_IMAGENET if imagenet else None)
        self.n_fases = n_fases
        self.extrator = base.features.requires_grad_(False)
        self.extrator.eval()
        self.pool = base.avgpool
        self.classificador = nn.Linear(n_fases * base.classifier[1].in_features, 1)

    def train(self, mode: bool = True):
        """Alterna o modo do classificador e mantém BatchNorm do extrator em eval."""
        super().train(mode)
        self.extrator.eval()
        return self

    def mapas_convolucionais(self, imagens: torch.Tensor) -> torch.Tensor:
        """Recebe (B, P, 3, T, T) e aplica um único extrator a B × P imagens."""
        if imagens.ndim != 5 or imagens.shape[1] != self.n_fases or imagens.shape[2] != 3:
            raise ValueError("A entrada deve ter forma (B, P, 3, T, T), conforme o modelo.")
        return self.extrator(imagens.flatten(0, 1))

    def vetores(self, mapas: torch.Tensor) -> torch.Tensor:
        """Aplica média espacial e concatena as P fases por paciente."""
        return self.pool(mapas).reshape(-1, self.n_fases * mapas.shape[1])

    def logits(self, vetores: torch.Tensor) -> torch.Tensor:
        """Devolve um logit por exemplo, com forma (B,)."""
        return self.classificador(vetores).squeeze(-1)

    def forward(self, imagens: torch.Tensor) -> torch.Tensor:
        """Executa a predição conjunta a partir de um lote de imagens."""
        return self.logits(self.vetores(self.mapas_convolucionais(imagens)))


def escolher_dispositivo(nome: str) -> torch.device:
    """Resolve auto para CUDA, MPS ou CPU, nessa ordem."""
    if nome == "auto":
        if torch.cuda.is_available():
            nome = "cuda"
        elif torch.backends.mps.is_available():
            nome = "mps"
        else:
            nome = "cpu"
    if nome == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA indisponível.")
    if nome == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS indisponível.")
    return torch.device(nome)


def sincronizar(dispositivo: torch.device):
    """Aguarda operações do acelerador para medir tempo de parede."""
    if dispositivo.type == "cuda":
        torch.cuda.synchronize(dispositivo)
    elif dispositivo.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def extrair_vetores(modelo, pacientes, config, lote, dispositivo) -> TensorDataset:
    """Extrai em lotes as dez versões e guarda apenas vetores e rótulos na RAM."""
    vetores, rotulos = [], []
    for imagens, referencias in lotes_preparados(pacientes, config, True, lote):
        mapas = modelo.mapas_convolucionais(imagens.to(dispositivo))
        vetores.append(modelo.vetores(mapas).cpu())
        rotulos.extend(ref.pcr for ref in referencias)
    return TensorDataset(torch.cat(vetores), torch.tensor(rotulos, dtype=torch.float32))


def rodar_epoca(modelo, lotes, perda, dispositivo, otimizador=None) -> float:
    """Faz uma passagem pelos vetores e devolve a perda média por exemplo."""
    treino = otimizador is not None
    modelo.train(treino)
    soma, quantidade = 0.0, 0
    with torch.set_grad_enabled(treino):
        for vetores, rotulos in lotes:
            rotulos = rotulos.to(dispositivo)
            valor = perda(modelo.logits(vetores.to(dispositivo)), rotulos)
            if treino:
                otimizador.zero_grad(set_to_none=True)
                valor.backward()
                otimizador.step()
            quantidade += rotulos.numel()
            soma += valor.item() * rotulos.numel()
    media = soma / quantidade
    if not np.isfinite(media):
        raise ValueError("Perda não finita nesta época.")
    return media


def salvar_modelo(caminho, modelo, config, dados_treino):
    """Salva a configuração e todo o state_dict, inclusive o extrator."""
    temporario = caminho.with_suffix(".tmp")
    torch.save({
        "versao": VERSAO_CHECKPOINT,
        "configuracao": asdict(config),
        "treinamento": dados_treino,
        "estado": modelo.state_dict(),
    }, temporario)
    temporario.replace(caminho)


def carregar_modelo(caminho: Path | str, dispositivo: torch.device):
    """Reconstrói o modelo salvo em eval, sem usar ou baixar pesos do ImageNet."""
    salvo = torch.load(caminho, map_location="cpu", weights_only=True)
    if salvo["versao"] != VERSAO_CHECKPOINT:
        raise ValueError(f"Versão de checkpoint não reconhecida: {salvo['versao']}.")
    config = ConfiguracaoPCR(**salvo["configuracao"])
    modelo = EfficientNetPCR(len(config.fases), imagenet=False)
    modelo.load_state_dict(salvo["estado"], strict=True)
    return modelo.to(dispositivo).eval(), config


def escrever_csv(caminho: Path, linhas: list[dict]):
    """Grava linhas com cabeçalho na ordem dos campos do primeiro registro."""
    with caminho.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def grafico_convergencia(historico: list[dict], caminho: Path):
    """Salva as duas curvas de perda por época."""
    figura = Figure(figsize=(7, 4), layout="constrained")
    eixo = figura.subplots()
    epocas = [linha["epoca"] for linha in historico]
    eixo.plot(epocas, [linha["perda_treino"] for linha in historico], "o-", label="Treino")
    eixo.plot(epocas, [linha["perda_validacao"] for linha in historico], "s-", label="Validação")
    eixo.set(xlabel="Época", ylabel="BCE ponderada", title="Convergência da EfficientNet-B0")
    eixo.legend()
    figura.savefig(caminho, dpi=150)


def treinar(args, pacientes: list[Paciente], contagens: dict, dispositivo: torch.device):
    """Treina a cabeça, seleciona pela perda de validação e exporta medições."""
    config = ConfiguracaoPCR(
        fases=FASES if args.fases == 3 else ("precoce",),
        estrategia=args.corte, tamanho=args.tamanho, percentis=tuple(args.percentis),
    )
    treino = [paciente for paciente in pacientes if paciente.particao == "treino"]
    validacao = [paciente for paciente in pacientes if paciente.particao == "validacao"]
    negativos, positivos = (contagens["treino"][chave] for chave in ("pcr_0", "pcr_1"))
    if not negativos or not positivos or not validacao:
        raise ValueError("São necessários treino com as duas classes e validação não vazia.")
    pos_weight = negativos / positivos
    print(f"pos_weight = {negativos}/{positivos} = {pos_weight:.4g}", flush=True)
    args.saida.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.semente)
    modelo = EfficientNetPCR(len(config.fases), imagenet=True).to(dispositivo)
    perda = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=dispositivo),
    )
    otimizador = torch.optim.Adam(modelo.classificador.parameters(), lr=args.taxa)
    dados_treino = {
        "epocas": args.epocas, "lote": args.lote, "taxa": args.taxa,
        "semente": args.semente, "pos_weight": pos_weight, "contagens": contagens,
        "pesos_iniciais": PESOS_IMAGENET.name, "extrator_congelado": True,
        "angulos": list(ANGULOS_GRAUS), "espelhamento_horizontal": [False, True],
        "criterio": "menor_perda_validacao", "dispositivo": str(dispositivo),
        "csv": str(args.csv.resolve()), "csv_fases": str(args.csv_fases or ""),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
    }
    sincronizar(dispositivo)
    inicio = perf_counter()
    print(f"Extraindo vetores de treino e validação em {dispositivo}.", flush=True)
    dados_treino_vetores = extrair_vetores(modelo, treino, config, args.lote, dispositivo)
    dados_val_vetores = extrair_vetores(modelo, validacao, config, args.lote, dispositivo)
    sincronizar(dispositivo)
    tempo_extracao = perf_counter() - inicio
    lotes_treino = DataLoader(
        dados_treino_vetores, batch_size=args.lote, shuffle=True,
        generator=torch.Generator().manual_seed(args.semente),
    )
    lotes_val = DataLoader(dados_val_vetores, batch_size=args.lote)
    historico = []
    melhor_perda = float("inf")
    melhor_epoca = None
    for epoca in range(1, args.epocas + 1):
        inicio_epoca = perf_counter()
        perda_treino = rodar_epoca(modelo, lotes_treino, perda, dispositivo, otimizador)
        perda_validacao = rodar_epoca(modelo, lotes_val, perda, dispositivo)
        sincronizar(dispositivo)
        linha = {
            "epoca": epoca, "perda_treino": perda_treino, "perda_validacao": perda_validacao,
            "tempo_epoca_s": perf_counter() - inicio_epoca,
        }
        historico.append(linha)
        if perda_validacao < melhor_perda:
            melhor_perda, melhor_epoca = perda_validacao, epoca
            salvar_modelo(args.saida / "melhor.pt", modelo, config, dados_treino | linha)
        escrever_csv(args.saida / "historico.csv", historico)
        print(
            f"Época {epoca:02d}  treino {perda_treino:.5f}  "
            f"validação {perda_validacao:.5f}  {linha['tempo_epoca_s']:.2f} s", flush=True,
        )
    tempo_total = perf_counter() - inicio
    resumo = dados_treino | {
        "configuracao": asdict(config), "melhor_epoca": melhor_epoca,
        "melhor_perda_validacao": melhor_perda, "tempo_extracao_s": tempo_extracao,
        "tempo_total_treino_s": tempo_total,
    }
    (args.saida / "resumo_treino.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    grafico_convergencia(historico, args.saida / "convergencia.png")


# 4. Predição, métricas e Grad-CAM


def calcular_metricas(previsoes: list[dict]) -> dict:
    """Deriva a matriz e as cinco métricas com pCR presente como positiva."""
    contagens = Counter(linha["desfecho"] for linha in previsoes)
    vp, vn, fp, fn = (contagens[chave] for chave in ("VP", "VN", "FP", "FN"))

    def razao(numerador, denominador):
        return numerador / denominador if denominador else None

    return {
        "VP": vp, "VN": vn, "FP": fp, "FN": fn,
        "sensibilidade": razao(vp, vp + fn),
        "especificidade": razao(vn, vn + fp),
        "precisao": razao(vp, vp + fp),
        "acuracia": razao(vp + vn, vp + vn + fp + fn),
        "f1": razao(2 * vp, 2 * vp + fp + fn),
    }


@torch.inference_mode(False)
def predizer_com_gradcam(
    exame: Paciente | ExamePCR, modelo: EfficientNetPCR, config: ConfiguracaoPCR,
    alvo: int | None = None,
) -> ResultadoPCR:
    """Retorna probabilidade e um mapa por fase, derivados do mesmo logit."""
    if alvo not in (None, 0, 1):
        raise ValueError("O alvo do Grad-CAM deve ser 0, 1 ou None para a classe prevista.")
    cortes = selecionar_cortes(exame, config, mostrar_mascara=True)
    dispositivo = modelo.classificador.weight.device
    imagens = preparar_entradas(cortes, config, aumentar=False).to(dispositivo)
    modelo.eval()
    with torch.no_grad():
        mapas = modelo.mapas_convolucionais(imagens)
    # Basta derivar o escore em relação aos mapas. Os pesos continuam fixos.
    with torch.enable_grad():
        mapas = mapas.detach().requires_grad_(True)
        logit = modelo.logits(modelo.vetores(mapas))[0]
        probabilidade = logit.detach().sigmoid().item()
        classe_prevista = int(probabilidade >= config.limiar)
        classe_gradcam = classe_prevista if alvo is None else alvo
        escore = logit if classe_gradcam == 1 else -logit
        gradientes, = torch.autograd.grad(escore, mapas)

    with torch.no_grad():
        pesos = gradientes.mean(dim=(-2, -1), keepdim=True)
        mapas_cam = (pesos * mapas).sum(dim=1, keepdim=True).relu()
        forma = cortes.imagens["precoce"].shape
        mapas_cam = F.interpolate(mapas_cam, forma, mode="bilinear", align_corners=False)
        mapas_cam = mapas_cam - mapas_cam.amin(dim=(-2, -1), keepdim=True)
        maximos = mapas_cam.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        mapas_cam = (mapas_cam / maximos).squeeze(1).cpu().numpy()
    return ResultadoPCR(
        probabilidade, classe_prevista, classe_gradcam, cortes,
        dict(zip(config.fases, mapas_cam)),
    )


def demonstrar(
    exame: Paciente | ExamePCR, checkpoint: Path | str, dispositivo: str = "cpu",
    alvo: int | None = None,
) -> ResultadoPCR:
    """Carrega pesos e infere um exame para uso futuro em um botão do Tkinter."""
    modelo, config = carregar_modelo(checkpoint, escolher_dispositivo(dispositivo))
    return predizer_com_gradcam(exame, modelo, config, alvo)


def salvar_gradcam(resultado: ResultadoPCR, titulo: str, caminho: Path):
    """Mostra o mesmo corte com contorno tumoral e com sobreposição Grad-CAM."""
    figura = Figure(figsize=(7, 1.0 + 3.0 * len(resultado.gradcam)), layout="constrained")
    eixos = figura.subplots(len(resultado.gradcam), 2, squeeze=False)
    mascara = resultado.cortes.mascara
    figura.suptitle(
        f"{titulo} · Z={resultado.cortes.z} · P(pCR)={resultado.probabilidade:.3f}\n"
        f"Classe prevista {resultado.classe_prevista} · Grad-CAM classe {resultado.classe_gradcam}",
    )
    for (fase, mapa), (esquerda, direita) in zip(resultado.gradcam.items(), eixos):
        cinza = para_uint8(resultado.cortes.imagens[fase])
        for eixo in (esquerda, direita):
            eixo.imshow(cinza, cmap="gray", vmin=0, vmax=255)
            eixo.set_axis_off()
        direita.imshow(mapa, cmap="inferno", alpha=0.65 * mapa, vmin=0, vmax=1)
        for eixo in (esquerda, direita):
            if mascara is not None and mascara.any() and not mascara.all():
                eixo.contour(mascara, levels=[0.5], colors=["#65e6ff"], linewidths=0.9)
        estado_mascara = ""
        if mascara is not None:
            estado_mascara = " · máscara" if mascara.any() else " · máscara vazia em Z"
        esquerda.set_title(NOMES_FASES[fase] + estado_mascara, fontsize=10)
        direita.set_title("Grad-CAM", fontsize=10)
    figura.savefig(caminho, dpi=150)


def testar(args, pacientes: list[Paciente], dispositivo: torch.device):
    """Avalia o checkpoint fixo com um único corte sem aumento por paciente."""
    modelo, config = carregar_modelo(args.checkpoint, dispositivo)
    pacientes_teste = [paciente for paciente in pacientes if paciente.particao == "teste"]
    if not pacientes_teste:
        raise ValueError("O CSV não tem pacientes com test=1.")
    previsoes = []
    desfechos = {(0, 0): "VN", (0, 1): "FP", (1, 0): "FN", (1, 1): "VP"}
    with torch.no_grad():
        for imagens, referencias in lotes_preparados(pacientes_teste, config, False, args.lote):
            probabilidades = modelo(imagens.to(dispositivo)).sigmoid().cpu().tolist()
            for ref, probabilidade in zip(referencias, probabilidades):
                classe = int(probabilidade >= config.limiar)
                previsoes.append({
                    "pid": ref.pid, "fases": "+".join(config.fases),
                    "aquisicoes": "+".join(map(str, ref.aquisicoes)),
                    "estrategia": config.estrategia, "z": ref.z, "pcr_real": ref.pcr,
                    "probabilidade_pcr": probabilidade, "classe_prevista": classe,
                    "limiar": config.limiar, "desfecho": desfechos[ref.pcr, classe],
                })

    args.saida.mkdir(parents=True, exist_ok=True)
    metricas = calcular_metricas(previsoes)
    escrever_csv(args.saida / "previsoes.csv", previsoes)
    escrever_csv(args.saida / "metricas.csv", [{
        "fases": "+".join(config.fases), "estrategia": config.estrategia, **metricas,
    }])
    escrever_csv(args.saida / "matriz_confusao.csv", [
        {"classe_real": 0, "prevista_0": metricas["VN"], "prevista_1": metricas["FP"]},
        {"classe_real": 1, "prevista_0": metricas["FN"], "prevista_1": metricas["VP"]},
    ])
    print(json.dumps(metricas, indent=2, ensure_ascii=False))

    pasta_cam = args.saida / "gradcam"
    pasta_cam.mkdir(exist_ok=True)
    por_pid = {paciente.pid: paciente for paciente in pacientes_teste}
    exemplos = {}
    for linha in previsoes:
        exemplos.setdefault(linha["desfecho"], linha["pid"])
    for desfecho, pid in exemplos.items():
        resultado = predizer_com_gradcam(por_pid[pid], modelo, config)
        salvar_gradcam(resultado, f"{pid} · {desfecho}", pasta_cam / f"{desfecho}_{pid}.png")


# 5. Comandos


def argumentos():
    """Define os comandos e valida os parâmetros básicos da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    comandos = parser.add_subparsers(dest="comando", required=True)
    for nome in ("auditar", "treinar", "testar", "demo"):
        comando = comandos.add_parser(nome)
        comando.add_argument("--base", type=Path, required=True, help="Pasta extraída do MinCrop.")
        comando.add_argument("--csv", type=Path, required=True, help="CSV com pCR e partições.")
        comando.add_argument("--csv-fases", type=Path, help="CSV MinCrop de referência das fases.")
        if nome != "auditar":
            comando.add_argument("--dispositivo", choices=("auto", "cpu", "cuda", "mps"), default="auto")
            comando.add_argument("--saida", type=Path, required=True, help="Diretório dos resultados.")
        if nome in ("treinar", "testar"):
            comando.add_argument("--lote", type=int, default=8)
        if nome in ("testar", "demo"):
            comando.add_argument("--checkpoint", type=Path, required=True)
        if nome == "treinar":
            comando.add_argument("--fases", type=int, choices=(1, 3), required=True)
            comando.add_argument("--corte", choices=ESTRATEGIAS, required=True)
            comando.add_argument("--tamanho", type=int, default=224)
            comando.add_argument("--percentis", nargs=2, type=float, default=(1.0, 99.5))
            comando.add_argument("--epocas", type=int, default=20)
            comando.add_argument("--taxa", type=float, default=1e-3)
            comando.add_argument("--semente", type=int, default=42)
        if nome == "demo":
            comando.add_argument("--pid", required=True)
            comando.add_argument("--alvo", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.comando in ("treinar", "testar") and args.lote < 1:
        parser.error("--lote deve ser positivo.")
    if args.comando == "treinar" and (
        args.epocas < 1 or not np.isfinite(args.taxa) or args.taxa <= 0
    ):
        parser.error("--epocas e --taxa devem ser positivos e finitos.")
    return args


def main():
    """Executa uma etapa e informa falhas com mensagens legíveis."""
    args = argumentos()
    try:
        pacientes = carregar_coorte(args.base, args.csv, args.csv_fases)
        contagens = contagens_classes(pacientes)
        if args.comando == "auditar":
            config = ConfiguracaoPCR(FASES, "maior_area")
            for paciente in pacientes:
                selecionar_cortes(paciente, config, mostrar_mascara=True)
            print(f"{len(pacientes)} pacientes com fases e máscara conferidas.")
            return
        dispositivo = escolher_dispositivo(args.dispositivo)
        if args.comando == "treinar":
            treinar(args, pacientes, contagens, dispositivo)
        elif args.comando == "testar":
            testar(args, pacientes, dispositivo)
        else:
            exame = next((paciente for paciente in pacientes if paciente.pid == args.pid), None)
            if exame is None:
                raise ValueError(f"PID {args.pid} não encontrado no CSV.")
            resultado = demonstrar(exame, args.checkpoint, str(dispositivo), args.alvo)
            args.saida.mkdir(parents=True, exist_ok=True)
            caminho = args.saida / f"{exame.pid}.png"
            salvar_gradcam(resultado, exame.pid, caminho)
            print(f"{exame.pid}: P(pCR) = {resultado.probabilidade:.6f}, figura em {caminho}")
    except (ValueError, OSError, RuntimeError, KeyError) as erro:
        raise SystemExit(f"Erro: {erro}") from erro


if __name__ == "__main__":
    main()
