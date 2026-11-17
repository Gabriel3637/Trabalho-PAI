# Classificação da resposta ao tratamento do câncer de mama (pCR) — ResNet50

**Trabalho Prático de Processamento e Análise de Imagens — PUC Minas**
**Parte do grupo:** classificador ResNet50
**Responsável:** João Pedro Torres — 8587622
**Dataset:** BreastDCEDL, subconjunto I-SPY1 MinCrop (172 pacientes)
**Código:** `src/resnet50_pcr.py`

> Todos os números deste documento vêm de execuções reais no Google Colab (GPU). Os arquivos de origem estão em `resultados/`. As saídas das células de verificação usadas durante o desenvolvimento não fazem parte deste repositório. Nada foi estimado ou projetado.

---

## 1. Objetivo e requisitos atendidos

O objetivo é estimar, a partir da DCE-MRI pré-tratamento, a probabilidade de uma paciente apresentar **resposta patológica completa (pCR)**.

| Requisito do enunciado | Como foi atendido |
|---|---|
| ResNet50 com pesos do ImageNet | `torchvision.models.resnet50`, pesos `IMAGENET1K_V2` |
| Retreinar somente a parte completamente conectada | Todo o backbone convolucional está **congelado**. Treinam apenas os 2.049 parâmetros da nova cabeça, de um total de ≈ 23,5 milhões. |
| Adaptar entrada e saída | Entrada: 3 canais (as fases ou uma fase replicada). Saída: `fc` de 1000 classes substituída por Dropout + Linear(2048 → 1). A sigmoide do logit é P(pCR). |
| Experimento A (fase isolada) | Três variantes: pré-contraste, pós-contraste precoce e pós-contraste tardio |
| Experimento B (três fases) | As três fases como os três canais da rede |
| ≥ 2 estratégias de cortes | E1: corte de maior área tumoral. E2: todos os cortes, com média das probabilidades. |
| Aumento de dados | Espelhamento e rotações de −20° a +20°, de 10° em 10° (10 variações), **somente no treino** |
| Desbalanceamento | `pos_weight` = 3,0 na loss, mais peso igual por paciente |
| Métricas | Acurácia, precisão, sensibilidade, especificidade, F1 e AUC, por paciente |
| Curvas de loss e tempo | Seções 7.4 e 7.6 |
| Grad-CAM | Seção 7.7 |
| Pesos salvos | `resultados/modelos/final_*/cabeca.pt`. O modelo completo é montado com `montar_modelo_completo`. |

---

## 2. Dados

### 2.1 Partições (oficiais, coluna `test` do CSV)

| Partição | Código em `test` | Pacientes | pCR | não-pCR |
|---|---|---|---|---|
| Treino | 0 | 104 | 26 | 78 |
| Validação | 2 | 33 | 11 | 22 |
| Teste | 1 | 35 | 12 | 23 |
| **Total** | | **172** | **49** | **123** |

- As partições oficiais foram respeitadas. Nenhum paciente aparece em mais de uma partição.
- O código verifica isso automaticamente em todo treino.
- O conjunto de teste ficou **bloqueado por código**: `carregar_caracteristicas` gera erro sem `liberar_teste=True`. Ele foi aberto uma única vez, na avaliação final.

### 2.2 Auditoria dos arquivos

Uma auditoria cobriu os 172 pacientes.

**Formato dos arquivos:**
- Todos os volumes têm formato (Z, 256, 256), com o eixo dos cortes no eixo 0. O número de cortes Z varia por paciente.
- As máscaras são binárias e têm o mesmo formato das fases.

**Fases:**
- Cada paciente tem três arquivos de fase (`aqc_N`). A fase é definida **ordenando os três índices existentes**, e essa regra coincide com o CSV em 171 de 172 pacientes.
- **Exceção ISPY1_1139:** o CSV indica (0, 2, 2), mas os arquivos são (0, 1, 2). O código usa os arquivos, emite um aviso e não altera o CSV.

**Divergências entre o CSV e os arquivos:**
- Nas colunas `n_z`, `n_xy` e `mask_start/end`, o CSV descreve o exame **original**, não o recorte MinCrop. Exemplo: ISPY1_1002 tem 22 cortes no arquivo e `n_z = 60` no CSV.
- Por isso, essas colunas não foram usadas para ler as imagens.

---

## 3. Pré-processamento

### 3.1 Normalização de intensidade: conjunta por paciente, P1–P99,9

**Como funciona:**
1. Para cada paciente, calculam-se P1 e P99,9 sobre os voxels das **três fases juntas**.
2. Valores fora desse intervalo são recortados.
3. As três fases são reescaladas para [0, 1] com os **mesmos** limites.

**Por que normalizar as três fases juntas:**
- A mesma transformação linear vale para as três fases. Assim, o **realce do contraste** (o tumor fica mais claro no pós-contraste) é preservado, e essa é justamente a informação temporal que o Experimento B investiga.
- Normalizar por paciente também reduz diferenças de escala entre aparelhos.

**Por que o percentil superior é P99,9** (escolha feita só com treino e validação; mediana entre pacientes, fase pós-contraste precoce):

| Limite superior | Tumor saturado (treino) | Realce preservado |
|---|---|---|
| P99 | 71 % | 72,9 % |
| **P99,9 (adotado)** | **10 %** | **98,1 %** |

**Alternativas rejeitadas:**
- **P99,99:** o limite passa a depender de pouquíssimos voxels (vasos, artefatos) e fica instável entre pacientes.
- **Normalização independente por fase:** continuava saturando o tumor e distorcia a relação entre as fases com um fator diferente para cada paciente.

**Limitação:** em cerca de 10 % dos pacientes, mais de 40 % do tumor ainda satura no pós-contraste.

### 3.2 Estratégias de combinação de cortes

| Estratégia | Cortes usados | Previsão do paciente | Usa a máscara? |
|---|---|---|---|
| **E1** | O corte de maior área tumoral | A probabilidade desse corte | **Sim**, para escolher o corte. A máscara não entra na rede. |
| **E2** | Todos os Z cortes do volume | A média das probabilidades dos cortes | Não |

A E1 depende de uma segmentação especialista. Numa aplicação real essa segmentação nem sempre estaria disponível, e isso é declarado como limitação.

### 3.3 Aumento de dados

- As variações são 5 ângulos (−20°, −10°, 0°, 10°, 20°) × 2 (com e sem espelhamento horizontal) = **10 variações determinísticas** por imagem, incluindo a original.
- As três fases recebem **a mesma** transformação, em uma única operação, e continuam alinhadas pixel a pixel.
- A interpolação é bilinear, e os cantos vazios recebem 0.
- O aumento é aplicado **somente ao treino**. Validação e teste usam apenas a imagem original.
- **Justificativa**, já que o enunciado menciona "treino/validação":
  - na ResNet, a validação serve para o early stopping e para comparar configurações;
  - cópias giradas não acrescentam pacientes, e a loss de validação já é ponderada por paciente;
  - mantê-la só com imagens originais faz a métrica medir pacientes reais.
- O módulo de aumento do grupo (`trabalho_pai.py`) gera as mesmas 10 variações, na mesma ordem (espelha e depois rotaciona).

---

## 4. Arquitetura

```
Corte (3, 256, 256) em [0, 1]
  → padronização ImageNet (média e desvio por canal)
  → ResNet50 congelada: conv1 → layer1 … layer4 → média global (avgpool)
  → vetor de 2048 características
  → cabeça treinável: Dropout(0,5) → Linear(2048 → 1)
  → logit → sigmoide = P(pCR)
```

**Entrada:**
- Exp. A: a fase escolhida é **replicada** nos três canais, o que permite aproveitar a primeira camada pré-treinada sem alterá-la.
- Exp. B: os canais são pré, pós precoce e pós tardio.
- O tamanho 256 × 256 é mantido, porque a média global da ResNet aceita qualquer resolução.

**Congelamento:**
- Atende à orientação do professor: retreinar somente a parte completamente conectada, sem fine-tuning.
- A rede é usada em modo `eval`, com as estatísticas de BatchNorm fixas.

**Extração antecipada de características:**
- Como o backbone é congelado, cada imagem gera sempre o mesmo vetor. Cada imagem passa pela rede **uma única vez**, os vetores são salvos e a cabeça é treinada sobre eles.
- Verificamos que isso é numericamente equivalente a treinar a ResNet50 completa com o backbone congelado (célula de verificação executada durante o desenvolvimento).
- O ganho de tempo é grande: o treino da cabeça leva segundos em vez de horas.

---

## 5. Treinamento

| Item | Valor | Justificativa |
|---|---|---|
| Função de perda | `BCEWithLogitsLoss` | Classificação binária com saída logit, numericamente estável |
| `pos_weight` | 3,0 = 78 / 26 (pacientes do treino) | Compensa a proporção de 1 pCR para 3 não-pCR |
| Peso por paciente | 1 / (nº de linhas do paciente) | Cada paciente soma 1 na loss. Sem isso, pacientes com mais cortes dominariam o treino. |
| Otimizador | Adam | Padrão robusto para uma camada linear |
| Taxa de aprendizado | **1e-4** (final); 1e-3 (preliminar) | Ver seção 7.1 (V1) |
| Weight decay | **1e-2** (final); 1e-4 (preliminar) | Regularização mais forte diante do overfitting observado |
| Dropout | 0,5 | Regularização da cabeça |
| Lote | 256 vetores | |
| Épocas | Máximo de 200, com early stopping (paciência 20) pela loss de validação | Guarda os pesos da melhor época |
| Semente | 42 (`random`, `numpy`, `torch`, cuDNN determinístico) | Reprodutibilidade |

**Treinos realizados:**
- **Preliminares:** 2 treinos de `B_tres_fases` (E1 e E2) com lr = 1e-3 e wd = 1e-4.
- **Finais:** 8 combinações (4 configurações × 2 estratégias) com **uma única configuração de hiperparâmetros**, fixada antes do treino e sem busca na validação. Com apenas 33 pacientes, escolher o "melhor" valor na validação equivaleria a escolher o mais sortudo.
- **Complementares:** as mesmas 8 combinações com o recorte tumoral (seção 6).

---

## 6. Experimento complementar: recorte da região tumoral

**Motivação:** o diagnóstico da seção 7.2 sugeriu que o tumor, com mediana de cerca de 2 % dos pixels do corte, é diluído na média global da ResNet.

**Como funciona:**
1. Calcula-se um quadrado ao redor do tumor, a partir da projeção da máscara 3D, com margem de 25 % de cada lado e lado mínimo de 32 px.
2. O quadrado é recortado e ampliado para 224 × 224.
3. No aumento de dados, a máscara recebe a mesma transformação da imagem, e o quadrado é recalculado.
4. São usados apenas os cortes que contêm tumor:
   - **E1:** o corte de maior área;
   - **E2:** a média dos cortes com tumor.

A ResNet continua congelada. Essa variante **usa a segmentação também na inferência**, o que é permitido pelo enunciado como opção. Ela foi declarada **complementar antes da avaliação no teste**; a família principal é a imagem inteira.

---

## 7. Resultados

### 7.1 Treinos preliminares (lr = 1e-3, wd = 1e-4, `B_tres_fases`, validação)

| Estratégia | Melhor época | Loss val. | AUC treino | AUC val. | Acurácia val. |
|---|---|---|---|---|---|
| E2 | 1 de 21 | 1,270 | 0,963 | 0,533 | 0,545 |
| E1 | 1 de 21 | 1,177 | 0,741 | 0,467 | 0,576 |

**Referência:** um classificador que sempre responde 0,5 tem loss de validação de **1,155**. Nos dois treinos, a loss de validação subiu desde a primeira época, enquanto o treino continuava melhorando. É **overfitting** imediato.

### 7.2 Diagnóstico das características (sem treinar)

Os diagnósticos usaram só treino e validação.

**V1, tamanho do passo:**
- As características são não negativas (33,8 % são zeros). A soma dos 2048 valores por imagem tem mediana de 176.
- Com lr = 1e-3, um passo do Adam pode mudar o logit em até ≈ 0,18. Com 1e-4, em até ≈ 0,02.
- A cabeça sem treino tem loss de validação de 1,16, o mesmo nível da referência.
- O passo grande acelera o overfitting, mas **não explica sozinho** a falta de generalização.

**V2, componentes principais** (`B_tres_fases`, média por paciente):
- As duas primeiras componentes explicam só 9 % e 7 % da variância.
- Elas correlacionam com o **tamanho do pixel** (ρ de Spearman = 0,51 e −0,59). Nas fases isoladas, a componente mais ligada ao pixel chega a |ρ| = 0,56–0,64.
- No mapa de pacientes, as aquisições com pixel ≈ 0,4 mm (originalmente 512 × 512) formam um grupo separado, enquanto pCR e não-pCR ficam misturados.
- Nenhuma componente separa pCR no treino (AUC entre 0,43 e 0,53).

**V3 e V4, característica por característica:**

| Configuração | Corr. AUC treino × val. | p (permutação) | Maior \|AUC − 0,5\| no treino | Mesmo valor sob acaso (p95) | AUC val. das 20 melhores do treino |
|---|---|---|---|---|---|
| B_tres_fases | −0,001 | 0,52 | 0,19 | 0,26 | 0,56 |
| A_pre | −0,065 | 0,71 | 0,22 | 0,27 | 0,48 |
| A_post_early | −0,013 | 0,49 | 0,20 | 0,26 | 0,52 |
| A_post_late | −0,053 | 0,65 | 0,20 | 0,26 | 0,49 |

**Leitura:**
- As características que separam pCR no treino **não** voltam a separar na validação.
- Nem a melhor característica do treino supera o que se obtém com rótulos embaralhados.

### 7.3 Teste de sinal: regressão logística regularizada

**Procedimento:**
- Padronização + regressão logística com `class_weight="balanced"`.
- C escolhido por validação cruzada **dentro do treino** (5 partes × 3 repetições).
- Comparação com 20 permutações dos rótulos.

**Imagem inteira:**

| Configuração | Estratégia | AUC VC (treino) | AUC acaso (média) | p | AUC validação |
|---|---|---|---|---|---|
| B_tres_fases | E1 | 0,604 | 0,487 | 0,14 | 0,455 |
| B_tres_fases | E2 | 0,509 | 0,509 | 0,48 | 0,517 |
| A_pre | E1 | 0,617 | 0,501 | 0,19 | 0,413 |
| A_pre | E2 | 0,512 | 0,524 | 0,57 | 0,388 |
| A_post_early | E1 | 0,506 | 0,528 | 0,52 | 0,500 |
| A_post_early | E2 | 0,422 | 0,501 | 0,91 | 0,504 |
| A_post_late | E1 | 0,570 | 0,475 | 0,14 | 0,442 |
| A_post_late | E2 | 0,502 | 0,496 | 0,43 | 0,517 |

**Recorte tumoral:**
- A AUC de validação ficou entre 0,376 e 0,566.
- Nenhum p-valor ficou abaixo de 0,05, e nenhuma AUC de validação cruzada superou o p95 do acaso.

Em resumo, nem um classificador linear bem regularizado encontrou sinal que generalize, em nenhuma das duas famílias.

### 7.4 Treinos finais da cabeça (lr = 1e-4, wd = 1e-2), validação (33 pacientes)

**Imagem inteira (família principal):**

| Configuração | Est. | Melhor época / executadas | Loss val. | AUC treino | AUC val. | Acur. | Prec. | Sens. | Espec. | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| B_tres_fases | E1 | 2 / 22 | 1,165 | 0,527 | 0,463 | 0,485 | 0,313 | 0,455 | 0,500 | 0,370 |
| B_tres_fases | E2 | 1 / 21 | 1,172 | 0,822 | 0,492 | 0,606 | 0,417 | 0,455 | 0,682 | 0,435 |
| A_pre | E1 | 1 / 21 | 1,161 | 0,618 | 0,455 | 0,424 | 0,278 | 0,455 | 0,409 | 0,345 |
| A_pre | E2 | 1 / 21 | 1,203 | 0,754 | 0,397 | 0,667 | 0,500 | 0,273 | 0,864 | 0,353 |
| A_post_early | E1 | 1 / 21 | 1,133 | 0,515 | 0,599 | 0,455 | 0,348 | 0,727 | 0,318 | 0,471 |
| A_post_early | E2 | 2 / 22 | 1,173 | 0,825 | 0,517 | 0,606 | 0,400 | 0,364 | 0,727 | 0,381 |
| A_post_late | E1 | 1 / 21 | 1,160 | 0,600 | 0,488 | 0,424 | 0,318 | 0,636 | 0,318 | 0,424 |
| A_post_late | E2 | 1 / 21 | 1,181 | 0,762 | 0,488 | 0,606 | 0,375 | 0,273 | 0,773 | 0,316 |

**Recorte tumoral (complementar):**

| Configuração | Est. | Melhor época / executadas | Loss val. | AUC treino | AUC val. | Acur. | Prec. | Sens. | Espec. | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| B_tres_fases | E1 | 7 / 27 | 1,191 | 0,701 | 0,368 | 0,576 | 0,333 | 0,273 | 0,727 | 0,300 |
| B_tres_fases | E2 | 1 / 21 | 1,165 | 0,778 | 0,467 | 0,424 | 0,250 | 0,364 | 0,455 | 0,296 |
| A_pre | E1 | 2 / 22 | 1,218 | 0,720 | 0,331 | 0,424 | 0,167 | 0,182 | 0,545 | 0,174 |
| A_pre | E2 | 1 / 21 | 1,200 | 0,759 | 0,413 | 0,455 | 0,231 | 0,273 | 0,545 | 0,250 |
| A_post_early | E1 | 3 / 23 | 1,191 | 0,661 | 0,388 | 0,545 | 0,333 | 0,364 | 0,636 | 0,348 |
| A_post_early | E2 | 1 / 21 | 1,177 | 0,755 | 0,467 | 0,485 | 0,286 | 0,364 | 0,545 | 0,320 |
| A_post_late | E1 | 3 / 23 | 1,199 | 0,657 | 0,368 | 0,485 | 0,250 | 0,273 | 0,591 | 0,261 |
| A_post_late | E2 | 1 / 21 | 1,182 | 0,768 | 0,475 | 0,455 | 0,182 | 0,182 | 0,591 | 0,182 |

**Referência na validação:** responder sempre "não-pCR" dá acurácia de 22/33 = 0,667.

**Curvas de convergência:** `resultados/figuras/convergencia_final_inteira.png` e `convergencia_final_recorte.png`.
- **E2:** a loss de treino cai de ≈ 1,02 para ≈ 0,75, enquanto a de validação sobe desde a primeira época. A cabeça memoriza os pacientes de treino: AUC de treino de 0,75–0,83, contra ≈ 0,4–0,5 na validação.
- **E1:** as duas curvas quase não se movem.
- Em 15 dos 16 modelos, a loss de validação ficou acima da referência constante (1,155).

### 7.5 Avaliação final no teste (35 pacientes: 12 pCR e 23 não-pCR; executada uma única vez)

O intervalo de confiança de 95 % da AUC foi obtido por bootstrap de pacientes (2.000 reamostragens).

**Imagem inteira (família principal):**

| Configuração | Est. | AUC | IC 95 % | Acur. | Prec. | Sens. | Espec. | F1 | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_tres_fases | E1 | 0,601 | [0,37; 0,83] | 0,629 | 0,467 | 0,583 | 0,652 | 0,519 | 7 | 15 | 8 | 5 |
| B_tres_fases | E2 | 0,558 | [0,33; 0,78] | 0,543 | 0,375 | 0,500 | 0,565 | 0,429 | 6 | 13 | 10 | 6 |
| A_pre | E1 | 0,525 | [0,29; 0,76] | 0,371 | 0,308 | 0,667 | 0,217 | 0,421 | 8 | 5 | 18 | 4 |
| A_pre | E2 | 0,453 | [0,24; 0,67] | 0,543 | 0,300 | 0,250 | 0,696 | 0,273 | 3 | 16 | 7 | 9 |
| A_post_early | E1 | 0,540 | [0,30; 0,77] | 0,486 | 0,364 | 0,667 | 0,391 | 0,471 | 8 | 9 | 14 | 4 |
| A_post_early | E2 | 0,598 | [0,39; 0,80] | 0,600 | 0,417 | 0,417 | 0,696 | 0,417 | 5 | 16 | 7 | 7 |
| A_post_late | E1 | 0,580 | [0,36; 0,79] | 0,457 | 0,348 | 0,667 | 0,348 | 0,457 | 8 | 8 | 15 | 4 |
| A_post_late | E2 | 0,572 | [0,35; 0,78] | 0,543 | 0,333 | 0,333 | 0,652 | 0,333 | 4 | 15 | 8 | 8 |

**Recorte tumoral (complementar):**

| Configuração | Est. | AUC | IC 95 % | Acur. | Prec. | Sens. | Espec. | F1 | VP | VN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_tres_fases | E1 | 0,641 | [0,42; 0,84] | 0,714 | 0,583 | 0,583 | 0,783 | 0,583 | 7 | 18 | 5 | 5 |
| B_tres_fases | E2 | 0,728 | [0,52; 0,91] | 0,629 | 0,474 | 0,750 | 0,565 | 0,581 | 9 | 13 | 10 | 3 |
| A_pre | E1 | 0,489 | [0,26; 0,72] | 0,514 | 0,381 | 0,667 | 0,435 | 0,485 | 8 | 10 | 13 | 4 |
| A_pre | E2 | 0,587 | [0,36; 0,79] | 0,514 | 0,353 | 0,500 | 0,522 | 0,414 | 6 | 12 | 11 | 6 |
| A_post_early | E1 | 0,576 | [0,34; 0,80] | 0,629 | 0,471 | 0,667 | 0,609 | 0,552 | 8 | 14 | 9 | 4 |
| A_post_early | E2 | 0,656 | [0,45; 0,83] | 0,514 | 0,333 | 0,417 | 0,565 | 0,370 | 5 | 13 | 10 | 7 |
| A_post_late | E1 | 0,529 | [0,29; 0,76] | 0,514 | 0,353 | 0,500 | 0,522 | 0,414 | 6 | 12 | 11 | 6 |
| A_post_late | E2 | 0,645 | [0,44; 0,83] | 0,571 | 0,400 | 0,500 | 0,609 | 0,444 | 6 | 14 | 9 | 6 |

**Referência no teste ("sempre não-pCR"):** acurácia 0,657, sensibilidade 0, especificidade 1, F1 0.

**Resumo:**
- **Imagem inteira:**
  - AUC média de 0,55 (de 0,45 a 0,60), e **os 8 intervalos de confiança incluem 0,5**.
  - Nenhum modelo supera a acurácia do classificador trivial.
- **Recorte:**
  - AUC média de 0,61 (de 0,49 a 0,73). Só B_tres_fases E2 tem limite inferior acima de 0,5, e mesmo assim por pouco (0,52).
  - Esses valores **contradizem a validação**: os mesmos modelos tiveram 0,33–0,48 na validação.
  - Entre 16 modelos avaliados, é esperado que um ou dois pareçam melhores apenas por acaso.
  - Por isso, esse resultado não é tomado como evidência de que o recorte funciona.

### 7.6 Tempos (GPU do Colab)

| Etapa | Tempo |
|---|---|
| Extração das características, imagem inteira (4 configurações de uma vez) | Treino 919 s (≈ 15 min; 29.850 imagens com aumento), validação 30 s, teste 32 s |
| Extração das características, recorte | Treino: 919 s, validacao: 37 s, teste: 40 s|
| Passagem pela ResNet50 | ≈ 3,1 ms por imagem |
| Treino da cabeça, E1 (1.040 vetores) | 0,2–0,8 s |
| Treino da cabeça, E2 (29.850 vetores inteira; 25.510 recorte) | 3,3–4,1 s |

### 7.7 Grad-CAM

- **Modelos analisados:** B_tres_fases E1 das duas famílias, sobre os 33 pacientes da validação.
- **Método:** o modelo completo é montado com o backbone congelado e a cabeça salva, e o mapa sai do gradiente do logit em relação à saída da `layer4`.
- **Caso particular:** como a cabeça é linear sobre a média global, o Grad-CAM coincide aqui com o CAM clássico, isto é, a soma dos mapas da `layer4` ponderada pelos pesos da cabeça.

**Verificação:** a probabilidade do modelo completo coincide com a probabilidade calculada no treino a partir das características salvas. A maior diferença foi de 1,8 × 10⁻⁷. Isso confirma que os pesos salvos reconstroem corretamente o modelo.

**Concentração no tumor** = (fração do mapa dentro do tumor) ÷ (fração da imagem ocupada pelo tumor). Valor 1 significa o mesmo que o acaso; acima de 1, a rede olha mais para o tumor.

| Família | Tumor na imagem (mediana) | Calor no tumor (mediana) | Concentração (mediana) | Pacientes com concentração > 1 |
|---|---|---|---|---|
| Inteira | 1,8 % | 1,9 % | 0,95 | 16 / 33 |
| Recorte | 7,7 % | 2,8 % | 0,49 | 7 / 33 |

**Figura:** `resultados/figuras/gradcam_validacao.png`.
- Nos exemplos, as regiões mais ativas ficam frequentemente na pele, na borda da mama, na parede torácica ou no fundo, e não no tumor.
- As probabilidades previstas ficam entre 0,44 e 0,57 para todos os pacientes. O modelo é praticamente indiferente entre as classes, coerente com o early stopping nas primeiras épocas.

---

## 8. Discussão e análise de erros

1. **Os modelos não apresentaram desempenho acima do acaso.** Nas 8 combinações exigidas (família principal), a AUC de teste ficou entre 0,45 e 0,60, com intervalos que incluem 0,5, e nenhum modelo superou o classificador trivial em acurácia. A validação (0,40–0,60) e os diagnósticos independentes (seções 7.2 e 7.3) mostram o mesmo.
2. **Por quê, segundo as evidências que temos:**
   - **Características genéricas:** as características do ImageNet, com o backbone congelado, variam principalmente com o **protocolo de aquisição** (tamanho do pixel), e não com a resposta ao tratamento (seção 7.2).
   - **Diluição do tumor:** o tumor ocupa cerca de 2 % do corte. O Grad-CAM mostra que a rede não concentra a atenção nele (concentração mediana de 0,95). Recortar a região tumoral não mudou o quadro na validação.
   - **Poucos dados:** são 104 pacientes de treino (26 pCR) para 2048 características. Os padrões que aparecem no treino não se repetem (seção 7.2, V3). A E2 memoriza os pacientes (AUC de treino até 0,83).
3. **Os erros dependem do limiar 0,5:**
   - Com probabilidades tão próximas de 0,5, pequenas diferenças decidem a classe. Por isso alguns modelos tendem a chamar muitos pacientes de pCR (ex.: inteira A_pre E1, sensibilidade 0,67 e especificidade 0,22) e outros, poucos (ex.: inteira A_pre E2, sensibilidade 0,25 e especificidade 0,70).
   - A AUC, que não depende do limiar, mostra que nenhum dos dois padrões reflete separação real.
4. **Exp. A × Exp. B e E1 × E2:** nenhuma configuração foi consistentemente melhor. As diferenças observadas estão dentro da variação esperada para 33 a 35 pacientes.
5. **Recorte:** os resultados de teste mais altos, sem correspondência na validação, são tratados como uma observação a investigar com mais dados, não como resultado.

---

## 9. Limitações e trabalhos futuros

**Limitações:**
- **Restrição metodológica do trabalho:** o backbone está congelado, sem fine-tuning, e as características do ImageNet não foram adaptadas a DCE-MRI.
- **Tamanho das partições:** 104 / 33 / 35 pacientes. Por isso, os intervalos de confiança das métricas são largos (± 0,2 na AUC).
- **Dependência da segmentação especialista:** a E1 e o recorte usam a máscara.
- **Normalização:** em cerca de 10 % dos pacientes o tumor ainda satura parcialmente no pós-contraste.

**Trabalhos futuros:**
- fine-tuning das últimas camadas;
- usar o BreastDCEDL completo, que reúne mais coortes;
- harmonização entre aparelhos;
- agregação 3D em vez de cortes 2D.

---

## 10. Reprodutibilidade

- **Código:** `src/resnet50_pcr.py`, organizado em seções:
  1. configuração;
  2. dados;
  3. aumento;
  4. modelo;
  5. extração (5.1 recorte);
  6. treino;
  7. avaliação;
  8. Grad-CAM.
- **Semente fixa:** 42, com cuDNN determinístico.
- **Pesos:** `resultados/modelos/final_<família>_<configuração>_E<estratégia>/cabeca.pt`. A ResNet50 completa é reconstruída com `montar_modelo_completo(caminho)`, a partir dos pesos ImageNet públicos e da cabeça salva.
- **Arquivos salvos por modelo:** histórico por época, previsões por paciente (treino, validação e teste) e métricas, na mesma pasta.
- **Fora do ZIP de entrega** (limite de 10 MB): o dataset, os pesos e as características extraídas (≈ 1 GB).

---