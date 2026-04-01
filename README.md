# pretrain-data-validator

사전학습 데이터의 품질을 검증하기 위한 도구. JSONL 데이터를 LLM API(vLLM 등)에 보내 생성 결과를 CSV로 출력한다.

부가 기능으로, 여러 JSONL 파일에서 랜덤 샘플을 뽑아 검증용 JSONL을 만드는 샘플러를 제공한다.

## 요구사항

- Python 3.14+
- vLLM 등 OpenAI-compatible API 서버 (validate 시)

## 설치

```bash
pip install -r requirements.txt

# 개발 환경
pip install -r requirements-dev.txt
```

## 명령어

### validate — LLM API로 데이터 검증

JSONL의 각 행을 LLM에 보내고, 생성 결과를 CSV로 저장한다.

```bash
python run.py validate --input data.jsonl --output result.csv [--config config.yml] [--dry-run]
```

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--input` | 입력 JSONL 파일 경로 | (필수) |
| `--output` | 출력 CSV 파일 경로 | (필수) |
| `--config` | YAML 설정 파일 경로 | `config.yml` |
| `--dry-run` | 설정/입력 검증만 수행, API 호출 안 함 | |

#### 입력 JSONL 형식

각 행은 JSON 객체. `prompt` 필드가 필수이며, `id`가 없으면 자동 부여된다.

```jsonl
{"id": "doc-001", "type": "qa", "prompt": "질문 내용"}
{"prompt": "id 없이도 가능"}
```

`prompt` 외의 필드명은 `config.yml`의 `input.prompt_field`로 변경 가능.

#### 출력 CSV

```
# model=my-model max_tokens=100 temperature=0.1 top_p=0.9 repetition_penalty=1.1 seed=42 input=data.jsonl
id,type,prompt,generated,model,finish_reason,prompt_tokens,completion_tokens
doc-001,qa,질문 내용,생성된 답변,my-model,stop,15,42
```

### sample — 랜덤 샘플링으로 검증 세트 생성

여러 JSONL 파일에서 랜덤으로 문서를 고르고, 각 문서에서 1~5줄을 뽑아 검증용 JSONL을 만든다.

```bash
python run.py sample corpus1.jsonl corpus2.jsonl -n 20 -o validation.jsonl [--text-field text] [--seed 42]
```

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `inputs` | 입력 JSONL 파일 경로들 (1개 이상) | (필수) |
| `-n` | 샘플링할 문서 수 | `10` |
| `-o`, `--output` | 출력 JSONL 파일 경로 | (필수) |
| `--text-field` | 텍스트 필드명 | `text` |
| `--seed` | 랜덤 시드 (재현성) | |

#### 출력 JSONL

```jsonl
{"file": "corpus1.jsonl", "id": "doc-42", "text": "추출된 일부 텍스트\n두 번째 줄"}
```

## 설정 (config.yml)

```yaml
api:
  base_url: "http://localhost:8000/v1"
  model: "your-model-name"
  concurrency: 8      # 동시 요청 수 (기본: 8)

generation:
  max_tokens: 100
  temperature: 0.1
  top_p: 0.9           # 기본: 1.0
  repetition_penalty: 1.1  # 기본: 1.0
  seed: 42

input:
  prompt_field: "prompt"   # 기본: "prompt"
```

## Docker

```bash
# 빌드
docker build -t hanati-pretrain-data-validator .

# validate
docker run --rm --network host -v $(pwd):/data hanati-pretrain-data-validator \
  validate --config /data/config.yml --input /data/data.jsonl --output /data/result.csv

# sample
docker run --rm -v $(pwd):/data hanati-pretrain-data-validator \
  sample /data/corpus1.jsonl /data/corpus2.jsonl -n 20 -o /data/validation.jsonl
```

### run.sh (Docker 래퍼)

```bash
# validate — 현재 디렉토리의 config.yml과 .jsonl 사용
./run.sh validate [input.jsonl] [--dry-run]

# sample
./run.sh sample file1.jsonl file2.jsonl -n 20 -o output.jsonl [--seed 42]
```

## 테스트

```bash
python -m pytest tests/ -v
```

## 프로젝트 구조

```
├── run.py              # CLI 엔트리포인트 (validate / sample)
├── run.sh              # Docker 래퍼 스크립트
├── config.yml          # 설정 파일
├── config.example.yml  # 설정 예시
├── Dockerfile
├── src/
│   ├── config.py       # YAML 설정 로드/검증
│   ├── loader.py       # JSONL 로더
│   ├── sampler.py      # 랜덤 샘플링
│   ├── generator.py    # LLM API 호출
│   └── writer.py       # CSV 출력
└── tests/
```
