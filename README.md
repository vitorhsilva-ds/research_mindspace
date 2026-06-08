# Arquitetura de Escolha e Nudges Digitais no E-commerce Brasileiro
### Um Estudo de Caso da KaBuM! na Black Friday (2020–2025)

Pipeline de pesquisa para identificação, codificação e análise de componentes do framework **MINDSPACE** nas interfaces da KaBuM! e nas reclamações do Reclame AQUI durante as edições de Black Friday de 2020 a 2025.

---

## Sumário

1. [Visão Geral](#visão-geral)
2. [Arquitetura do Pipeline](#arquitetura-do-pipeline)
3. [Pré-requisitos](#pré-requisitos)
4. [Criação do Ambiente](#criação-do-ambiente)
5. [Configuração](#configuração)
6. [Execução do Pipeline](#execução-do-pipeline)
   - [Etapa 1 — Coleta de interfaces (Wayback Machine)](#etapa-1--coleta-de-interfaces-wayback-machine)
   - [Etapa 2 — Normalização do corpus](#etapa-2--normalização-do-corpus)
   - [Etapa 3 — Análise visual das interfaces (VLM)](#etapa-3--análise-visual-das-interfaces-vlm)
   - [Etapa 4 — Fine-tuning do modelo de texto (opcional)](#etapa-4--fine-tuning-do-modelo-de-texto-opcional)
   - [Etapa 5 — Classificação das reclamações](#etapa-5--classificação-das-reclamações)
   - [Etapa 6 — Construção das tabelas analíticas](#etapa-6--construção-das-tabelas-analíticas)
   - [Etapa 7 — Revisão manual](#etapa-7--revisão-manual)
   - [Etapa 8 — Merge das revisões](#etapa-8--merge-das-revisões)
   - [Etapa 9 — Construção da ABT](#etapa-9--construção-da-abt)
   - [Etapa 10 — Análise e visualizações](#etapa-10--análise-e-visualizações)
7. [Ferramentas de Revisão](#ferramentas-de-revisão)
8. [Estrutura de Diretórios](#estrutura-de-diretórios)
9. [Infraestrutura de Modelos (vLLM)](#infraestrutura-de-modelos-vllm)
10. [Disponibilidade de Dados](#disponibilidade-de-dados)
11. [Framework MINDSPACE](#framework-mindspace)

---

## Visão Geral

Este repositório contém o pipeline de pesquisa desenvolvido para a tese de mestrado sobre **arquitetura de escolha e nudges digitais no e-commerce brasileiro**, com foco no estudo de caso da plataforma KaBuM! nas edições da Black Friday entre 2020 e 2025.

O pipeline combina:

- **Captura histórica** de interfaces via Wayback Machine (Internet Archive)
- **Análise visual multimodal** por modelo de linguagem visual (Qwen3-VL-8B-Instruct)
- **Classificação de reclamações** por modelo de texto (**GPT-OSS-20B**) submetido a fine-tuning supervisionado com LoRA sobre corpus sintético
- **Codificação semiautomatizada** alinhada ao framework MINDSPACE
- **Ferramentas de revisão manual** via interfaces web Flask e FastAPI
- **Análise estatística e visualização** dos indicadores `p_I`, `p_R` e `gap_IR` por componente e por ano

---

## Arquitetura do Pipeline

```
Wayback Machine ──► image_capture/ ──► normalize_corpus.py ──► corpus_interface.jsonl
                                                                         │
                                                              image_processing/
                                                         (Qwen3-VL via vLLM :8001)
                                                                         │
                                                                   vlm_output/
Reclame AQUI ──► [coleta manual MHTML] ──► normalize_corpus.py ──► corpus_ra.jsonl
                                                                         │
                                                              finetuning/classify_corpus.py
                                                           (modelo fine-tuned via vLLM :8000)
                                                                         │
                                                                    ra_output/
                    ┌────────────────────────────────────────────────────┘
                    ▼
              build_ti.py ──► ti_completa.csv / ti_revisao.csv
              build_tr.py ──► tr_completa.csv / tr_revisao.csv
                    │
          [Revisão manual via revisors/]
                    │
              merge_review.py
                    │
              build_abt.py ──► abt.csv
                    │
              analysis/ ──► gráficos e estatísticas
```

---

## Pré-requisitos

### Sistema operacional

Ubuntu 22.04 ou 24.04 (recomendado). Scripts testados em Linux. Compatível com macOS com ajustes menores no Selenium.

### Software base

| Componente | Versão mínima | Observação |
|---|---|---|
| Python | 3.11 | Recomendado 3.11 ou 3.12 |
| pip | 23+ | Atualizar antes da instalação |
| Google Chrome | 120+ | Necessário para o Selenium |
| vLLM | 0.5+ | Servidor de inferência dos modelos |

### Hardware para inferência (VLM e modelo de texto)

| Componente | Requisito |
|---|---|
| GPU NVIDIA | Recomendado RTX 3090 / A100 / A6000 |
| VRAM | ≥ 24 GB (Qwen3-VL-8B-Instruct) |
| RAM | ≥ 32 GB |
| CUDA | 12.1+ |

> **Nota:** A inferência pode ser realizada em servidores remotos. Os scripts se conectam a endpoints OpenAI-compatíveis configuráveis via argumento `--model-api-url`.

---

## Criação do Ambiente

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd tese
```

### 2. Crie o ambiente virtual

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Atualize o pip

```bash
pip install --upgrade pip
```

### 4. Instale as dependências principais

```bash
pip install -r requirements.txt
```

### 5. (Opcional) Instale as dependências de fine-tuning

Necessário apenas se for realizar o fine-tuning do modelo localmente. Requer GPU com CUDA 12.1+.

```bash
pip install -r requirements-finetune.txt
```

> **Atenção:** O Unsloth exige uma variante de instalação específica para o seu ambiente CUDA. Consulte a [documentação oficial](https://docs.unsloth.ai/get-started/installing-+-updating) antes de instalar.

### 6. Verifique a instalação

```bash
python -c "import pandas, numpy, matplotlib, httpx, yaml, PIL, selenium, flask, fastapi; print('OK')"
```

---

## Configuração

### Estrutura de diretórios de trabalho

Por padrão, o pipeline espera o seguinte diretório raiz na home do usuário. Todos os scripts aceitam o argumento `--workspace` para sobrescrever:

```
~/tese/
├── corpus/                  # Saída dos scripts de normalização e construção de tabelas
│   ├── vlm_output/          # Resultados da análise visual (JSON por registro)
│   └── ra_output/           # Resultados da classificação de reclamações (JSON por registro)
├── image_capture/
│   ├── wayback_inventory/   # CSVs de inventário do Wayback Machine
│   └── wayback_captures/    # Screenshots e MHTMLs capturados
├── ra_capture/              # MHTMLs das reclamações coletadas do Reclame AQUI
├── model_artifacts/         # Modelos fine-tuned (após finetune.py)
└── configs/                 # Arquivos YAML de prompt (já incluídos no repositório)
```

Crie os diretórios necessários:

```bash
mkdir -p ~/tese/corpus/vlm_output \
         ~/tese/corpus/ra_output \
         ~/tese/image_capture/wayback_inventory \
         ~/tese/image_capture/wayback_captures \
         ~/tese/ra_capture \
         ~/tese/model_artifacts
```

### Configuração dos prompts MINDSPACE

Os arquivos de configuração dos prompts estão em `configs/` e já estão prontos para uso:

- `configs/mindspace_prompt_config.yaml` — prompts para classificação de reclamações (texto)
- `configs/visual_mindspace_prompt_config.yaml` — prompts para análise visual de interfaces

Edite esses arquivos para ajustar definições operacionais dos componentes MINDSPACE, exemplos de nudges e sludges, ou instruções de sistema conforme necessário.

---

## Execução do Pipeline

### Etapa 1 — Coleta de interfaces (Wayback Machine)

#### 1a. Inventário de URLs disponíveis

Lista as capturas disponíveis no Wayback Machine para as janelas temporais de cada Black Friday (2020–2025):

```bash
cd image_capture

python list_wayback_urls.py \
  --out-dir ~/tese/image_capture/wayback_inventory \
  --years 2020 2021 2022 2023 2024 2025
```

#### 1b. Construção do plano de captura

Seleciona os snapshots mais representativos por ano, tipo de página e janela temporal:

```bash
python build_capture_plan.py \
  --inventory-dir ~/tese/image_capture/wayback_inventory \
  --years 2020 2021 2022 2023 2024 2025
```

#### 1c. Captura dos screenshots

Executa o Selenium para fotografar cada URL do plano. Requer Google Chrome instalado:

```bash
python capture_wayback.py \
  --inventory-dir ~/tese/image_capture/wayback_inventory \
  --out-dir ~/tese/image_capture/wayback_captures
```

> **Nota:** Este passo pode levar horas dependendo do número de capturas. Utilize `--skip-existing` para retomar interrupções.

---

### Etapa 2 — Normalização do corpus

Constrói os arquivos JSONL estruturados a partir das capturas de interface e das reclamações coletadas manualmente:

```bash
cd ..  # raiz do projeto

python normalize_corpus.py \
  --interface-dir ~/tese/image_capture/wayback_captures \
  --complaint-dir ~/tese/ra_capture \
  --out-dir ~/tese/corpus
```

Saídas:
- `~/tese/corpus/corpus_interface.jsonl` — registros das interfaces
- `~/tese/corpus/corpus_ra.jsonl` — registros das reclamações
- `~/tese/corpus/ra_sem_data.jsonl` — reclamações sem data identificada

---

### Etapa 3 — Análise visual das interfaces (VLM)

Envia cada imagem de interface ao modelo visual (Qwen3-VL-8B-Instruct) via servidor vLLM na porta 8001:

```bash
cd image_processing

python visual_interface_extraction.py \
  --corpus ~/tese/corpus/corpus_interface.jsonl \
  --config ../configs/visual_mindspace_prompt_config.yaml \
  --out-dir ~/tese/corpus/vlm_output \
  --model-api-url http://localhost:8001/v1/chat/completions \
  --model-name /caminho/para/Qwen3-VL-8B-Instruct \
  --workers 2
```

Argumentos opcionais:
- `--year 2024` — processa apenas um ano
- `--test <id>` — processa apenas um registro específico
- `--skip-existing` — pula registros já processados

Saída: um arquivo `KBW_<id>.json` por registro em `~/tese/corpus/vlm_output/`.

---

### Etapa 4 — Fine-tuning do modelo de texto (opcional)

Esta etapa é necessária apenas para treinar o modelo classificador de reclamações. Requer GPU.

#### 4a. Geração de dados sintéticos

Gera exemplos sintéticos de nudges e sludges por componente MINDSPACE:

```bash
cd ../finetuning

python generate_nudges.py \
  --config ../configs/mindspace_prompt_config.yaml \
  --model-api-url http://localhost:8000/v1/chat/completions \
  --model-name models/gpt-oss-20b \
  --out-dir ./nudges_raw \
  --examples-per-component 100
```

#### 4b. Classificação do corpus sintético

```bash
python classify_nudges.py \
  --config ../configs/mindspace_prompt_config.yaml \
  --model-api-url http://localhost:8000/v1/chat/completions \
  --model-name models/gpt-oss-20b \
  --in-dir ./nudges_raw
```

#### 4c. Preparação do dataset de fine-tuning

```bash
python prepare_finetune.py \
  --input ./dataset_finetune_full.jsonl \
  --out-dir ./finetune_data
```

#### 4d. Fine-tuning com LoRA

```bash
python finetune.py \
  --epochs 3 \
  --lora-r 8 \
  --lr 2e-4
```

O modelo treinado será salvo em `./model_lora/` e, se bem-sucedido, mesclado em `./model_merged/`.

#### 4e. Validação do modelo

```bash
python validate_model.py \
  --config ../configs/mindspace_prompt_config.yaml \
  --model-api-url http://localhost:8000/v1/chat/completions \
  --model-name model_artifacts/model_merged_mxfp4
```

---

### Etapa 5 — Classificação das reclamações

Classifica cada reclamação do corpus segundo os componentes MINDSPACE:

```bash
cd ../finetuning  # ou da raiz do projeto

python classify_corpus.py \
  --config ../configs/mindspace_prompt_config.yaml \
  --corpus ~/tese/corpus/corpus_ra.jsonl \
  --out-dir ~/tese/corpus/ra_output \
  --model-api-url http://localhost:8000/v1/chat/completions \
  --model-name model_artifacts/model_merged_mxfp4 \
  --skip-existing
```

Saída: um arquivo `RA_<id>.json` por reclamação em `~/tese/corpus/ra_output/`.

---

### Etapa 6 — Construção das tabelas analíticas

#### 6a. Tabela de interfaces (TI)

```bash
cd ..  # raiz do projeto

python build_ti.py \
  --vlm-dir ~/tese/corpus/vlm_output \
  --corpus ~/tese/corpus/corpus_interface.jsonl \
  --out-dir ~/tese/corpus
```

Saídas:
- `ti_completa.csv` — tabela completa de interfaces codificadas
- `ti_revisao.csv` — registros sinalizados para revisão manual

#### 6b. Tabela de reclamações (TR)

```bash
python build_tr.py \
  --ra-dir ~/tese/corpus/ra_output \
  --out-dir ~/tese/corpus
```

Saídas:
- `tr_completa.csv` — tabela completa de reclamações classificadas
- `tr_revisao.csv` — registros sinalizados para revisão manual

---

### Etapa 7 — Revisão manual

Utilize as ferramentas de revisão web para corrigir ou validar registros sinalizados. Veja a seção [Ferramentas de Revisão](#ferramentas-de-revisão) para instruções de uso.

---

### Etapa 8 — Merge das revisões

Incorpora as correções manuais nas tabelas completas:

```bash
python merge_review.py \
  --corpus-dir ~/tese/corpus
```

O script consolida `ti_completa.csv` + `ti_revisao.csv` e `tr_completa.csv` + `tr_revisao.csv`, sobrescrevendo os registros revisados.

---

### Etapa 9 — Construção da ABT

Constrói a **Analytical Base Table** com os indicadores por componente MINDSPACE e por ano:

```bash
python build_abt.py \
  --corpus-dir ~/tese/corpus \
  --out-dir ~/tese/corpus
```

Saídas:
- `abt.csv` — tabela analítica com colunas `comp`, `ano`, `N_I`, `n_I`, `p_I`, `density_media`, `N_R`, `n_R`, `p_R`, `gap_IR`
- `abt_summary.txt` — resumo descritivo dos indicadores

---

### Etapa 10 — Análise e visualizações

Execute os scripts de análise na ordem sugerida abaixo. Todos leem `~/tese/corpus/abt.csv` por padrão e salvam os gráficos em `~/tese/corpus/charts/`.

```bash
cd analysis

# Visão geral do corpus
python corpus_overview.py

# Distribuição por locus de interface
python locus_distribution.py

# Análise de componentes na interface
python interface_component_analysis.py

# Análise de componentes nas reclamações
python complaint_component_analysis.py

# Matriz de co-ocorrência de componentes
python cooccurrence_matrix.py

# Análise temporal do gap_IR
python temporal_gap_analysis.py

# Pattern matching (convergência entre interface e reclamações)
python pattern_matching.py

# Exportação de casos de divergência para análise qualitativa
python export_divergence_cases.py
```

---

## Ferramentas de Revisão

### Revisor de Interfaces (revisor_ti) — porta 5051

Interface web para revisão visual dos registros de interface codificados pelo VLM. Permite editar os componentes MINDSPACE identificados, instâncias e locus, visualizando a imagem correspondente lado a lado.

```bash
cd revisors/revisor_ti

# Configure o diretório do corpus (opcional — padrão: ~/workspace/corpus)
export REVIEW_CORPUS_DIR=~/tese/corpus

python app.py
```

Acesse em: [http://localhost:5051](http://localhost:5051)

### Revisor de Reclamações (revisor_tr) — porta 5050

Interface web para revisão das classificações de reclamações. Exibe o texto original extraído do MHTML e permite corrigir os componentes MINDSPACE e atributos de nudge/sludge.

```bash
cd revisors/revisor_tr

export REVIEW_CORPUS_DIR=~/tese/corpus
export RA_CAPTURE_DIR=~/tese/ra_capture

python app.py
```

Acesse em: [http://localhost:5050](http://localhost:5050)

### Image Slicer — porta 8080

Serviço FastAPI para fatiamento manual de screenshots grandes em slices menores, facilitando a análise visual. Permite recortar regiões de interesse diretamente no navegador.

```bash
cd image_slicer

uvicorn main:app --host 0.0.0.0 --port 8080
```

Acesse em: [http://localhost:8080](http://localhost:8080)

---

## Estrutura de Diretórios

```
.
├── configs/
│   ├── mindspace_prompt_config.yaml         # Prompts para classificação de texto
│   └── visual_mindspace_prompt_config.yaml  # Prompts para análise visual
│
├── image_capture/
│   ├── archive_collection_core.py           # Núcleo de coleta (CDX API + Selenium)
│   ├── list_wayback_urls.py                 # Inventário de URLs no Wayback Machine
│   ├── build_capture_plan.py                # Seleção e planejamento das capturas
│   └── capture_wayback.py                   # Execução das capturas com Selenium
│
├── image_processing/
│   ├── visual_interface_extraction.py       # Pipeline de extração visual (VLM)
│   ├── visual_mindspace_pipeline_core.py    # Núcleo do pipeline visual
│   ├── visual_mindspace_prompt_loader.py    # Carregamento dos prompts visuais
│   ├── visual_model_api_test.py             # Teste de conectividade com a API do VLM
│   └── visual_model_validation.py           # Validação da qualidade do VLM
│
├── image_slicer/
│   ├── main.py                              # Serviço FastAPI de fatiamento de imagens
│   └── static/index.html                   # Interface web do slicer
│
├── finetuning/
│   ├── generate_nudges.py                   # Geração de dados sintéticos de treino
│   ├── classify_nudges.py                   # Classificação do corpus sintético
│   ├── prepare_finetune.py                  # Preparação do dataset train/val
│   ├── finetune.py                          # Fine-tuning LoRA com Unsloth
│   ├── validate_model.py                    # Validação do modelo fine-tuned
│   ├── classify_corpus.py                   # Classificação das reclamações
│   └── prompt_config_loader.py              # Carregamento dos configs YAML
│
├── revisors/
│   ├── revisor_ti/
│   │   ├── app.py                           # API Flask do revisor de interfaces
│   │   └── static/index.html               # Interface web do revisor TI
│   └── revisor_tr/
│       ├── app.py                           # API Flask do revisor de reclamações
│       └── static/index.html               # Interface web do revisor TR
│
├── analysis/
│   ├── visualization_pipeline_core.py       # Serviços compartilhados de visualização
│   ├── corpus_overview.py                   # Visão geral do corpus
│   ├── locus_distribution.py               # Distribuição por locus
│   ├── interface_component_analysis.py      # Análise de componentes na interface
│   ├── complaint_component_analysis.py      # Análise de componentes nas reclamações
│   ├── cooccurrence_matrix.py              # Matriz de co-ocorrência
│   ├── temporal_gap_analysis.py            # Evolução temporal do gap_IR
│   ├── pattern_matching.py                 # Pattern matching (convergência teórica)
│   └── export_divergence_cases.py          # Exportação de casos divergentes
│
├── normalize_corpus.py                      # Normalização e estruturação do corpus
├── build_ti.py                              # Construção da tabela de interfaces
├── build_tr.py                              # Construção da tabela de reclamações
├── merge_review.py                          # Merge das revisões manuais
├── build_abt.py                             # Construção da tabela analítica (ABT)
├── requirements.txt                         # Dependências principais
└── requirements-finetune.txt               # Dependências de fine-tuning (GPU)
```

---

## Infraestrutura de Modelos (vLLM)

O pipeline utiliza dois servidores vLLM independentes com interfaces OpenAI-compatíveis:

| Servidor | Porta padrão | Modelo | Caminho padrão | Uso |
|---|---|---|---|---|
| VLM (Visual) | `8001` | `Qwen3-VL-8B-Instruct` | `~models/Qwen3-VL-8B-Instruct` | Análise visual das interfaces |
| Texto (fine-tuned) | `8000` | `GPT-OSS-20B` (fine-tuned, mxfp4) | `~tese/model_artifacts/model_merged_mxfp4` | Classificação das reclamações |

### Iniciar o servidor VLM (Qwen3-VL)

```bash
vllm serve ~models/Qwen3-VL-8B-Instruct \
  --port 8001 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

### Iniciar o servidor de texto (GPT-OSS-20B fine-tuned)

```bash
vllm serve ~tese/model_artifacts/model_merged_mxfp4 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

> **Dica:** É possível usar qualquer endpoint compatível com a API OpenAI, inclusive serviços em nuvem como Azure OpenAI ou Together AI. Basta ajustar `--model-api-url`, `--model-api-key` e `--model-name` nos scripts.

---

## Disponibilidade de Dados

**Os dados coletados nesta pesquisa não são e não serão disponibilizados publicamente**, independentemente do formato (raw, processado ou derivado). Essa decisão decorre de limitações legais objetivas que se aplicam a ambas as fontes utilizadas.

### Wayback Machine (Internet Archive)

As capturas de tela das interfaces da KaBuM! foram obtidas a partir de snapshots arquivados pelo Internet Archive. O conteúdo arquivado (layouts, imagens, textos de interface) permanece sob os direitos autorais originais da KaBuM!/Magazine Luiza (Lei nº 9.610/1998 — Lei de Direitos Autorais). Os [Termos de Uso do Internet Archive](https://archive.org/about/terms) autorizam o acesso e o uso das coleções para fins de pesquisa e bolsa acadêmica não comercial, mas **não autorizam a redistribuição ou reprodução do conteúdo arquivado** sem permissão escrita dos detentores dos direitos.

### Reclame AQUI

As reclamações coletadas são publicamente acessíveis na plataforma, o que viabiliza seu uso para fins de pesquisa. No entanto, a publicidade dos dados não equivale à ausência de direitos sobre eles, o conteúdo textual das reclamações é de autoria dos consumidores e está protegido pela Lei de Direitos Autorais, a base de dados estruturada é de propriedade do Reclame AQUI, protegida enquanto obra intelectual (Lei nº 9.610/1998, art. 7º, XIII). A LGPD (Lei nº 13.709/2018) autoriza o tratamento de dados pessoais para fins de pesquisa científica com anonimização (art. 7º, IV), mas essa autorização cobre o uso analítico e não a redistribuição dos dados em sua forma original.

### Dados sintéticos e reprodutibilidade

O corpus de treinamento do modelo classificador (**GPT-OSS-20B** fine-tuned) foi gerado **exclusivamente a partir de dados sintéticos** produzidos pelo script `finetuning/generate_nudges.py`, sem nenhuma utilização ou menção de conteúdo proveniente das plataformas KaBuM!, Wayback Machine ou Reclame AQUI. Os pesos do modelo são, portanto, livres de quaisquer restrições derivadas das fontes de dados primárias.

A reprodutibilidade da pesquisa é assegurada pelos **procedimentos documentados** neste repositório e pelas **referências às fontes originais**, permitindo que outros pesquisadores coletem dados equivalentes de forma independente, seguindo o mesmo protocolo:

- Interfaces históricas: acessíveis via [https://web.archive.org](https://web.archive.org) com o mesmo plano de captura (`image_capture/build_capture_plan.py`)
- Reclamações: acessíveis via [https://www.reclameaqui.com.br](https://www.reclameaqui.com.br) seguindo as janelas temporais definidas em `normalize_corpus.py`

---

## Framework MINDSPACE

O framework MINDSPACE (Dolan et al., 2010) organiza os mecanismos de influência comportamental em nove componentes:

| Código | Componente | Descrição |
|---|---|---|
| **M** | Messenger | Somos influenciados por quem comunica a informação |
| **I** | Incentives | Respondemos a incentivos e desincentivos de forma previsível |
| **N** | Norms | Somos influenciados pelo que outros fazem |
| **D** | Defaults | Seguimos as opções pré-definidas |
| **S** | Salience | Nossa atenção é atraída pelo que é novo e relevante |
| **P** | Priming | Somos influenciados por pistas subconscientes |
| **A** | Affect | Reações emocionais moldam nossas ações |
| **C** | Commitments | Tendemos a honrar compromissos públicos e recíprocos |
| **E** | Ego | Agimos de maneira consistente com nossa autoimagem |

O indicador central da pesquisa, `gap_IR`, é calculado como:

```
gap_IR = p_I − p_R
```

Onde `p_I` é a proporção de uso do componente nas interfaces e `p_R` é a proporção de verbalização nas reclamações dos consumidores para o mesmo ano e componente.

---

## Referências

- Dolan, P. et al. (2010). MINDSPACE: Influencing behaviour through public policy. Cabinet Office / Institute for Government.
- Thaler, R. H.; Sunstein, C. R. (2021). *Nudge: A nudge final*. Objetiva.
- Weinmann, M.; Schneider, C.; vom Brocke, J. (2016). Digital nudging. *Business & Information Systems Engineering*, 58(6), 433–436.
- Yin, R. K. (2003). *Case Study Research: Design and Methods*. SAGE.
