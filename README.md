# CGRAG: Continual Learning with Query-Conditioned Edges for Multi-Hop QA

Multi-hop 질문 답변을 위한 지속 학습(Continual Learning) 기반 검색 시스템입니다. Query Decomposition과 Knowledge Graph 기반 검색을 결합하여, 이전 검색 경험(QCEdge)을 활용해 점진적으로 검색 성능을 개선합니다.

## 🎯 핵심 아이디어

### Query-Conditioned Edge (QCEdge)
- **QCEdge**는 STPPR(Source-Target Personalized PageRank)를 통해 추출된 query와 연관된 중요한 knowledge graph edge입니다
- QCEdge 값 = PPR forward score × RBS backward flow
- 각 검색 단계에서 추출된 QCEdge를 다음 단계의 그래프 강화에 활용하여 점진적으로 성능을 개선합니다

### 4단계 파이프라인

```
Step 250 (초기 검색)
    ↓
    - Atomic Bridge Question 추출
    - Context-aware Query Decomposition
    - Multi-hop 검색 수행
    - STPPR → QCEdge 추출
    ↓
Step 500 (QCEdge 강화)
    ↓
    - Step 1의 QCEdge로 그래프 엣지 가중치 강화
    - 강화된 그래프로 재검색
    - 새로운 QCEdge 추출
    - Intersection QCEdge 계산 (Step 1과 Step 2의 교집합)
    ↓
Step 750 (Intersection + Extra QCEdge 강화)
    ↓
    - Extra QCEdge 선택 (percentile 기준)
    - Intersection + Extra QCEdge로 그래프 강화 (theta_mult=15, wub=6)
    - 강화된 그래프로 재검색
    ↓
Step 1000 (강한 강화)
    ↓
    - 더 강한 파라미터로 그래프 강화 (theta_mult=30, wub=20)
    - 최종 검색 수행
```

## 📋 주요 구성 요소

### 1. Query Decomposition
- **Atomic Bridge Question**: Multi-hop 질문을 단일 factual question으로 분해
- **Context-aware Decomposition**: 이전 검색 결과를 컨텍스트로 활용한 질문 분해

### 2. STPPR (Source-Target Personalized PageRank)
- 검색된 passage에서 시작하여 query와 관련된 중요한 노드/엣지를 찾는 알고리즘
- Forward PPR과 Backward RBS(Reset-based Backward Search)를 결합

### 3. QCEdge 기반 그래프 강화
- **Step 2**: 단순 QCEdge로 엣지 가중치 강화
- **Step 3/4**: Intersection QCEdge + Extra QCEdge를 결합한 강화
  - Intersection: 여러 단계에서 공통으로 나타난 중요한 엣지
  - Extra: Step 1에서 높은 중요도를 가진 엣지

## 🚀 사용 방법

### 환경 설정

```bash
conda create -n hipporag python=3.10
conda activate hipporag
pip install hipporag

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=<your openai api key>
```

### 실행

```python
from Ours.Final_method import run_full_pipeline, PipelineConfig

# 설정
CONFIG = PipelineConfig()
CONFIG.dataset_names = ["musique"]  # 또는 ["hotpotqa", "2wikimultihopqa"]
CONFIG.step_values = [250, 500, 750, 1000]

# 파이프라인 실행
results_df = run_full_pipeline(
    dataset_name="musique",
    config=CONFIG,
    verbose=True
)
```

### 주요 파라미터

```python
@dataclass
class PipelineConfig:
    # Step 1 파라미터
    top_k_per_bridge: int = 5       # Bridge question당 검색할 triple 수
    top_k_per_hop: int = 5          # Sub-question당 검색할 triple 수
    top_k_edges: int = 30           # 저장할 QCEdge 수
    
    # Step 2 파라미터
    theta_step2: float = 15.0       # Edge 강화 강도
    wub_step2: float = 3.0          # Weight upper bound
    
    # Step 3 파라미터
    percentile_step3: float = 50.0  # Extra QCEdge 선택 percentile
    theta_mult_step3: float = 15.0   # theta multiplier
    wub_step3: float = 6.0
    
    # Step 4 파라미터
    theta_mult_step4: float = 30.0  # 더 강한 강화
    wub_step4: float = 20.0
```

## 📁 프로젝트 구조

```
HippoRAG/
├── Ours/
│   ├── Final_method.ipynb          # 메인 파이프라인 구현
│   ├── prompts/
│   │   └── QD_bridge2_prompts_reasoning/
│   │       ├── birdge_extraction_with_description.txt
│   │       └── simple_query_decomposition.txt
│   └── _hippo_rag_MHQA_CL/         # 데이터셋별 결과 저장
│       ├── musique/
│       ├── hotpotqa/
│       └── 2wikimultihopqa/
├── src/hipporag/                    # HippoRAG 핵심 모듈
│   ├── HippoRAG.py
│   ├── embedding_store.py
│   ├── rerank.py
│   └── ...
└── reproduce/
    └── dataset/                     # 데이터셋 파일
        ├── musique.json
        ├── hotpotqa.json
        └── 2wikimultihopqa.json
```

## 🔬 실험 결과

이 방법론은 다음 데이터셋에서 평가되었습니다:
- **MuSiQue**: Multi-hop 질문 답변
- **HotpotQA**: Wikipedia 기반 multi-hop QA
- **2WikiMultihopQA**: Wikipedia 기반 multi-hop QA

각 단계별로 검색 성능(Recall@K, Hit@K)이 점진적으로 개선됩니다.

## 📝 주요 기능

### 1. Atomic Bridge Question 추출
Multi-hop 질문을 해결하기 위한 중간 단계의 factual question을 추출합니다.

### 2. Context-aware Query Decomposition
이전 검색 결과를 컨텍스트로 활용하여 더 정확한 질문 분해를 수행합니다.

### 3. STPPR 기반 QCEdge 추출
- Forward PPR: Passage에서 시작하여 관련 노드 탐색
- Backward RBS: Query에서 역방향으로 중요한 경로 탐색
- QCEdge = Forward score × Backward flow

### 4. 점진적 그래프 강화
- Step 2: 단순 QCEdge 강화
- Step 3: Intersection + Extra QCEdge 강화
- Step 4: 더 강한 파라미터로 최종 강화

## 🛠️ 의존성

- `hipporag`: HippoRAG 핵심 라이브러리
- `openai`: GPT 모델 사용
- `numpy`, `pandas`: 데이터 처리
- `dataclasses`: 설정 관리

## 📄 라이선스

원본 HippoRAG 프로젝트의 라이선스를 따릅니다.

## 👤 작성자

kimminyeol

## 📚 참고 문헌

- HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models
- From RAG to Memory: Non-Parametric Continual Learning for Large Language Models
