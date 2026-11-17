# Textos da ResNet50 para o relatório SBC (LaTeX)

## 2.4 Desbalanceamento (completar os dois TODO)

```latex
O aumento de dados multiplica as duas classes pelo mesmo fator, então a razão entre elas
continua 3,00 no treino e não corrige o desbalanceamento. A estratégia adotada foi o uso de
pesos de classe na função de perda, e foram descartados o aumento proporcional (repetiria as
mesmas geometrias) e a focal loss (introduz o hiperparâmetro $\gamma$ extra).

Na ResNet50, a perda é a entropia cruzada binária (\texttt{BCEWithLogitsLoss}) com
\texttt{pos\_weight} $= n_0/n_1 = 78/26 = 3{,}0$, calculado só com os pacientes de treino: errar
um paciente com pCR custa três vezes mais que errar um sem pCR. Esse valor equivale à razão
$w_1/w_0 = 2{,}0/0{,}667$ da Equação~1. Além disso, cada linha é ponderada por $1/(\text{número
de linhas do paciente})$, para que cada paciente contribua com peso 1, independentemente de
quantos cortes tenha (de 10 a 72).
```

## 2.3 Aumento de dados (frase a acrescentar, para a ResNet)

```latex
Na ResNet50, o aumento foi aplicado somente à partição de treino. A partição de validação,
usada para o \textit{early stopping}, contém apenas as imagens originais, de modo que a perda
de validação meça pacientes reais e não cópias transformadas. As três fases de um corte sofrem
a mesma transformação, em uma única operação, e permanecem alinhadas.
```

## 2.5 Estratégias de combinação dos cortes

```latex
Foram investigadas duas estratégias. Na \textbf{E1}, usa-se apenas o corte de maior área
tumoral de cada paciente, segundo a máscara de segmentação; a máscara serve somente para
escolher o corte e não entra na rede. Na \textbf{E2}, todos os cortes do volume são
classificados e a probabilidade do paciente é a \textbf{média} das probabilidades dos cortes;
essa estratégia não usa a máscara. A avaliação é feita por paciente, e em ambas as estratégias
usam-se apenas as imagens originais, sem aumento. Não foram avaliados o corte central nem o
máximo das previsões.
```

## 2.6.2 ResNet50

```latex
Foi usada a ResNet50 do \texttt{torchvision} com os pesos \texttt{IMAGENET1K\_V2}. Todo o
backbone convolucional permanece \textbf{congelado}, em modo de avaliação (as estatísticas da
\textit{batch normalization} não são atualizadas), conforme a orientação do professor. A camada
completamente conectada original (2048~$\to$~1000 classes) foi substituída por
\textit{Dropout}(0,5) seguido de \texttt{Linear}(2048~$\to$~1). A saída é um \textit{logit};
sua sigmoide é $P(\text{pCR})$. Apenas os 2.049 parâmetros da nova cabeça são treinados, de
cerca de 23,5 milhões.

A primeira camada convolucional manteve 3 canais e seus pesos pré-treinados. As imagens são
normalizadas para $[0,1]$ e depois padronizadas com a média e o desvio por canal do ImageNet. O
tamanho 256$\times$256 é mantido, pois a média global (\textit{adaptive average pooling}) da
rede aceita qualquer resolução.

Como o backbone é congelado, a mesma imagem sempre gera o mesmo vetor de 2048 características.
Cada imagem passa pela rede uma única vez, os vetores são salvos e a cabeça é treinada sobre
eles. Verificou-se numericamente que isso equivale a treinar a ResNet50 completa com o backbone
congelado. O modelo completo (backbone + cabeça treinada) é reconstruído a partir do arquivo
de pesos da cabeça (\texttt{cabeca.pt}) e dos pesos públicos do ImageNet.
```

## 2.7 Uma fase versus três fases

```latex
No \textbf{Experimento A}, uma única fase (pré-contraste, pós-precoce ou pós-tardio) é
replicada nos três canais de entrada, o que permite aproveitar a primeira camada pré-treinada
sem alterá-la. No \textbf{Experimento B}, as três fases formam os três canais: canal 1,
pré-contraste; canal 2, pós-contraste precoce; canal 3, pós-contraste tardio. Cada paciente
tem as três fases normalizadas em conjunto: os percentis 1 e 99,9 são calculados sobre os
voxels das três fases juntas e a mesma transformação linear é aplicada às três, o que preserva
o realce de contraste. O percentil 99,9 foi escolhido, usando só treino e validação, porque o
percentil 99 saturava 71\% do tumor na fase pós-precoce, contra 10\% do percentil 99,9, que
preserva 98\% do realce.
```

## 2.8 Treinamento e protocolo

```latex
A cabeça foi treinada com Adam (taxa de aprendizado $10^{-4}$, \textit{weight decay}
$10^{-2}$), lotes de 256 vetores, até 200 épocas, com \textit{early stopping} de paciência 20
sobre a perda de validação, mantendo os pesos da melhor época. Os hiperparâmetros foram fixados
antes do treino e são iguais para todas as combinações, pois, com 33 pacientes na validação,
escolher o melhor valor nela equivaleria a escolher o mais favorecido pelo acaso. Todas as
sementes foram fixadas em 42 (\texttt{random}, \texttt{numpy}, \texttt{torch}, cuDNN
determinístico). O conjunto de teste foi mantido bloqueado no código e avaliado uma única vez,
depois de todos os modelos estarem salvos. O treinamento foi feito em GPU do Google Colab
T4.
```

## 2.9 Grad-CAM

```latex
O Grad-CAM é calculado sobre a saída da última camada convolucional (\texttt{layer4}, 2048
mapas): o gradiente do \textit{logit} em relação a esses mapas fornece o peso de cada um, e o
mapa final é o ReLU da soma ponderada, ampliado ao tamanho da imagem. Como a cabeça é linear
sobre a média global, o resultado coincide com o CAM clássico. O mapa é sobreposto ao canal
pós-contraste precoce, com o contorno do tumor. Para quantificar a relação com o tumor,
usou-se a \textbf{concentração no tumor}: fração do mapa dentro da máscara dividida pela fração
da imagem ocupada pelo tumor (1 = igual ao acaso).
```

## 2.10 Máscara de segmentação (opcional): foi realizado, manter a seção

```latex
Como experimento complementar, o tumor foi recortado: um quadrado ao redor da projeção da
máscara 3D (margem de 25\% de cada lado, lado mínimo de 32 px) é recortado e ampliado para
224$\times$224. Só os cortes com tumor são usados. A ResNet50 continua congelada; muda apenas a
imagem apresentada. Esse experimento usa a segmentação também na classificação, o que não seria
possível sem uma segmentação disponível. Foi declarado complementar antes da avaliação no teste;
a família principal é a imagem inteira, sem recorte.
```

## Tabela 2: linhas a acrescentar (substitui o `[TODO: framework]`)

```latex
PyTorch / torchvision & \texttt{resnet50}, \texttt{BCEWithLogitsLoss}, \texttt{Adam},
\texttt{Dropout}, \texttt{Linear} & pesos \texttt{IMAGENET1K\_V2}; $\text{lr}=10^{-4}$;
\textit{weight decay} $10^{-2}$; lote 256; até 200 épocas; \textit{dropout} 0,5;
\texttt{pos\_weight}=3,0 & transferência de aprendizado com o backbone congelado; perda
ponderada pelo desbalanceamento \\
scikit-learn & \texttt{roc\_auc\_score} & --- & AUC sem depender do limiar \\
```

## 4.1 Protocolo (completar o TODO)

```latex
Treino em GPU do Google Colab T4, sementes fixas em 42. A avaliação é feita por
paciente, com limiar 0,5 para sensibilidade, especificidade, precisão, acurácia e F1, e
também pela AUC, com intervalo de confiança de 95\% obtido por \textit{bootstrap} de pacientes
(2.000 reamostragens). A partição de teste tem 35 pacientes (12 com pCR). Um classificador que
responde sempre ``sem pCR'' tem acurácia de 0,657, sensibilidade 0 e especificidade 1.
```

## Tabela 3: linhas da ResNet50 (teste, por paciente, Estratégia E2)

Escolhi a E2 para a Tabela 3 porque **não usa a máscara**, o que é um critério independente do resultado. As duas estratégias aparecem na Tabela 4. Se o grupo preferir a E1, a tabela completa está em `resultados/resultados_teste_16.csv`.

```latex
ResNet50 & Pré-contraste  & 0,250 & 0,696 & 0,300 & 0,543 & 0,273 & 15,8$^{a}$ \\
ResNet50 & Pós-precoce    & 0,417 & 0,696 & 0,417 & 0,600 & 0,417 & 15,8$^{a}$ \\
ResNet50 & Pós-tardio     & 0,333 & 0,652 & 0,333 & 0,543 & 0,333 & 15,8$^{a}$ \\
ResNet50 & Três fases     & 0,500 & 0,565 & 0,375 & 0,543 & 0,429 & 15,8$^{a}$ \\
```

Nota de rodapé da tabela:

```latex
$^{a}$ Tempo de extração das características de treino e validação (919 s + 30 s), comum às
quatro entradas, que são extraídas na mesma passagem pela rede; o treino da cabeça leva
3,7 a 4,1 s por modelo.
```

## 4.2 Desempenho

```latex
Na ResNet50 congelada, as AUC no teste ficaram entre 0,45 e 0,60 nas oito combinações
principais, e todos os intervalos de confiança de 95\% incluem 0,5; nenhum modelo superou a
acurácia do classificador trivial (0,657). Na E2, as matrizes de confusão (VP/VN/FP/FN) foram:
três fases, 6/13/10/6; pré-contraste, 3/16/7/9; pós-precoce, 5/16/7/7; pós-tardio, 4/15/8/8.
O experimento complementar com recorte tumoral obteve AUC entre 0,49 e 0,73; o valor mais alto
(três fases, E2, 0,73; IC 0,52--0,91) contradiz a validação do mesmo modelo (AUC 0,47) e,
entre 16 modelos avaliados, é esperado que um ou dois pareçam melhores apenas por acaso, de
modo que não é tomado como evidência de que o recorte funcione.
```

## Tabela 4: estratégias (ResNet50, três fases, validação)

Usei a **validação** porque é a partição onde se compara opções. Preencha o título com "validação, modelo: ResNet50 três fases". As linhas "central" e "máximo" ficam com "--" (não avaliadas).

```latex
Corte central            & --    & --    & --    & --    & --    \\
Maior área tumoral (E1)  & 0,455 & 0,500 & 0,313 & 0,485 & 0,370 \\
Média (todos) (E2)       & 0,455 & 0,682 & 0,417 & 0,606 & 0,435 \\
Máximo (todos os cortes) & --    & --    & --    & --    & --    \\
```

## 4.3 Estratégias de corte

```latex
Na validação, a média de todos os cortes (E2) teve acurácia e especificidade maiores que o corte
de maior área (E1) na entrada de três fases, com sensibilidade igual. No teste, a relação se
inverteu (AUC 0,60 na E1 e 0,56 na E2), o que mostra que, com 33 a 35 pacientes, diferenças desse
tamanho não permitem apontar uma estratégia melhor. Em termos de erro, as E1 das entradas A
tenderam a muitos falsos positivos (18, 14 e 15 FP no teste, nas fases pré, pós-precoce e
pós-tardio) e as E2 a muitos falsos negativos (9, 7 e 8 FN).
```

## Tabela 5: ajuste de hiperparâmetros (ResNet50, três fases, validação)

```latex
ResNet50 & lr $10^{-3}$, wd $10^{-4}$ (E2) & 0,273 & 0,286 & melhor época 1; perda de validação 1,270 \\
ResNet50 & lr $10^{-3}$, wd $10^{-4}$ (E1) & 0,364 & 0,364 & melhor época 1; perda 1,177 \\
ResNet50 & lr $10^{-4}$, wd $10^{-2}$ (E2) & 0,455 & 0,435 & melhor época 1; perda 1,172 \\
ResNet50 & lr $10^{-4}$, wd $10^{-2}$ (E1) & 0,455 & 0,370 & melhor época 2; perda 1,165 \\
```

Observação para a legenda: a perda de validação de um classificador que sempre responde 0,5 é 1,155; todos os valores acima são maiores.

## 4.4 Convergência

```latex
A Figura~3 mostra a perda de treino e de validação dos oito modelos principais da ResNet50. Na
E2, a perda de treino cai de cerca de 1,02 para 0,75 enquanto a de validação sobe desde a
primeira época, com AUC de treino de 0,75 a 0,83 contra 0,40 a 0,52 na validação (a cabeça
memoriza os pacientes de treino). Na E1, as duas curvas quase não se movem. Em 15 dos 16 modelos
treinados, incluindo os do recorte, a perda de validação ficou acima da referência de 1,155, e a
melhor época foi a 1, a 2 ou a 3 em quase todos. Como a perda de treino chega a cair, a rede
tem capacidade para ajustar o treino, mas isso não se transfere aos pacientes novos.
```

## 4.5 Grad-CAM (e legenda honesta da Figura 4)

```latex
Nos 33 pacientes da validação, a concentração mediana do Grad-CAM no tumor foi 0,95 na imagem
inteira (16 pacientes acima de 1) e 0,49 no recorte (7 acima de 1). As probabilidades previstas
ficaram entre 0,44 e 0,57 para todos os pacientes. Nos exemplos visuais, as regiões mais
ativas ficam com frequência na pele, na borda da mama, na parede torácica ou no fundo.
```

Legenda da Figura 4, se usarem a figura que já existe (`resultados/figuras/gradcam_validacao.png`):

```latex
Grad-CAM de quatro pacientes da validação (duas com pCR e duas sem), modelo ResNet50 de três
fases com a E1, imagem inteira (centro) e recorte (direita). O contorno branco é o tumor.
```

**Atenção:** essa figura mostra pacientes por classe real, **não** por VP/VN/FP/FN. Se o template exigir exemplos de acertos e erros, é preciso gerar outra figura a partir das previsões do teste. Posso preparar essa célula.

## 5.1 e 5.3 Hiperparâmetros, overfitting e convergência

```latex
\textbf{Congelamento e cabeça simples.} Com 104 pacientes de treino e backbone fixo, a cabeça
tem apenas 2.049 parâmetros; mesmo assim, em E2 há overfitting forte (seção 4.4).
\textbf{Taxa de aprendizado.} Com $10^{-3}$, um passo do Adam pode mudar o \textit{logit} em até
cerca de 0,18; com $10^{-4}$, em cerca de 0,02. Reduzir a taxa e aumentar o \textit{weight
decay} tornou o treino mais estável, mas não gerou generalização: a causa não é o passo.
\textbf{Underfitting.} Não é a explicação: a perda de treino diminui. \textbf{Velocidade de
convergência.} O \textit{early stopping} interrompeu o treino entre 21 e 27 épocas, com a melhor
época entre a 1 e a 7.
```

## 5.4 Perguntas do enunciado (parte da ResNet50)

```latex
\textbf{1. Três fases versus uma fase?} Na ResNet50, a AUC no teste foi 0,56 para as três fases
e 0,45, 0,60 e 0,57 para pré, pós-precoce e pós-tardio (E2); todos os intervalos incluem 0,5.
Os dados não permitem afirmar que as três fases melhorem a classificação.

\textbf{3. Erros mais frequentes.} Com limiar 0,5, as E1 das fases isoladas tendem a falsos
positivos e as E2 a falsos negativos (seção 4.3). Como as probabilidades ficam em 0,44--0,57,
pequenas diferenças decidem a classe e o tipo de erro predominante depende do limiar.

\textbf{4. Grad-CAM e tumor?} Não de forma consistente: a concentração mediana é 0,95 (acaso =
1). O mapa se distribui por pele, borda da mama e fundo.
```

## 5.5 Limitações

```latex
\begin{itemize}
\item O backbone permanece congelado, por exigência do enunciado, e as características do
ImageNet não foram adaptadas a DCE-MRI. Os diagnósticos indicam que elas refletem sobretudo o
protocolo de aquisição (correlação de 0,51 e $-$0,59 das duas primeiras componentes principais
com o tamanho do pixel) e que as características que separam pCR no treino não se repetem na
validação.
\item As partições têm 104, 33 e 35 pacientes; os intervalos de confiança da AUC são largos
(cerca de $\pm$0,2).
\item A E1 e o recorte dependem da segmentação especialista da máscara.
\end{itemize}
```

## 5.6 Conclusão (parte da ResNet50)

```latex
Com a ResNet50 congelada, nenhuma das oito combinações (quatro entradas $\times$ duas
estratégias) previu pCR de forma distinguível do acaso no teste. A análise das características
e do Grad-CAM sugere que o backbone descreve principalmente o aparelho e a mama inteira, e não
o tumor. Trabalhos futuros incluem o ajuste fino das últimas camadas, o uso do BreastDCEDL
completo, com mais coortes, e a harmonização entre aparelhos.
```