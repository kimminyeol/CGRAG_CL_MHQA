# CGRAG: Continual Learning with Query-Conditioned Edges for Multi-Hop QA

Multi-hop 질문 답변을 위한 지속 학습 기반 검색 시스템입니다.

## 실행 순서

1. **`Ours/_hippo_rag_MHQA_CL/Create_embedding_graph.ipynb`**  
   스텝별 그래프와 임베딩을 먼저 생성한다.

2. **`Ours/Final_method.ipynb`**  
   위에서 생성한 결과를 사용해 파이프라인을 실행할 수 있다.

## 환경 설정

```bash
conda create -n hipporag python=3.10
conda activate hipporag
pip install hipporag

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<path to Huggingface home directory>
export OPENAI_API_KEY=<your openai api key>
```

## 라이선스

원본 HippoRAG 프로젝트의 라이선스를 따릅니다.