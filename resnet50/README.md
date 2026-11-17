# ResNet50 para prever pCR — Trabalho Prático de PAI (PUC Minas)

Este repositório contém a parte **ResNet50** do trabalho "Classificação da resposta ao tratamento do câncer de mama usando redes neurais" (Processamento e Análise de Imagens, Prof. Alexei Machado). O dataset é o BreastDCEDL I-SPY1 MinCrop (172 pacientes).

## Estrutura

```
src/resnet50_pcr.py                       código da ResNet50: dados, aumento, modelo, extração, treino, avaliação e Grad-CAM
notebooks/executar_resnet50_pcr.ipynb     notebook do Colab que reproduz o experimento usando o módulo
dados/BreastDCEDL_SPY1_min_crop_metadata.csv   metadados do dataset (rótulos pCR e partições dos 172 pacientes)
documentacao/RELATORIO_RESNET50.md        decisões, métodos e resultados da ResNet50
documentacao/TEXTOS_RELATORIO_RESNET50.md blocos em LaTeX para o relatório SBC do grupo
resultados/resultados_teste_16.csv        métricas dos 16 modelos no teste (+ referência "sempre não pCR")
resultados/resumo_treinos_finais.csv      16 treinos: melhor época, tempo, métricas de treino e validação
resultados/gradcam_validacao.csv          concentração do Grad-CAM no tumor, por paciente da validação
resultados/tempos_extracao.json           tempo de extração das características
resultados/figuras/                       curvas de convergência da loss e Grad-CAM
resultados/modelos/final_*/               16 modelos; em cada pasta: cabeca.pt (pesos treinados),
                                          historico.csv, metricas.json, metricas_teste.json, previsoes_*.csv
```

**Fora do repositório** (ver `.gitignore`), por tamanho ou por exigência do enunciado:
- o dataset (~1,3 GB);
- as características extraídas `.npy` (~2 GB);
- o modelo completo com os pesos do ImageNet (~190 MB).

Tudo isso é **regenerado pelo notebook**.

## Como executar no Google Colab (pelo Drive)

1. No Google Drive, crie a pasta `MyDrive/resnet50_pcr/` e coloque **na raiz dela**:
   - `resnet50_pcr.py` (o arquivo de `src/`);
   - `executar_resnet50_pcr.ipynb` (o arquivo de `notebooks/`);
   - `BreastDCEDL_SPY1_min_crop_metadata.csv` (o arquivo de `dados/`);
   - (opcional) a pasta `resultados/` inteira, para reaproveitar os modelos já treinados e avaliados.
2. No Drive, clique duas vezes no notebook → **Abrir com → Google Colaboratory**.
3. Selecione **Ambiente de execução → Alterar o tipo → GPU (T4)**.
4. Confira a célula 1 (os caminhos já apontam para `MyDrive/resnet50_pcr/`) e rode as células em ordem.
   - **O dataset não precisa ser enviado.** Na primeira sessão, o notebook o baixa do Zenodo e guarda uma cópia em `MyDrive/resnet50_pcr/dados/`.
   - Com os modelos já salvos em `resultados/`, a extração de características (~30 min) é pulada (`GERAR_CARACTERISTICAS = False`).

## Resultado resumido

Com a ResNet50 congelada (pesos ImageNet, treinando só a camada final), nenhum dos 8 modelos exigidos previu pCR acima do acaso. A AUC no teste ficou entre 0,45 e 0,60, e todos os intervalos de confiança incluem 0,5. Os detalhes e a análise das causas estão em `documentacao/RELATORIO_RESNET50.md`.
