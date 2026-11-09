# rg

> Define Mathematics. Compile with Confidence. Run at Speed.

Python으로 수학 함수를 정의하면 Rust 엔진이 추적·미분·최적화·검증한 뒤, 스텝 사이에 Python으로 돌아오지 않고 반복 실행합니다. 문자 단위 GPT를 학습하고 샘플링할 수 있습니다.

**문서:** [`docs/`](docs/index.md) (`.venv/bin/mkdocs serve`로 사이트 보기)

| 시작점 | 내용 |
| --- | --- |
| [설치](docs/install.md) | 소스 빌드, 플랫폼 상태 |
| [빠른 시작](docs/quickstart.md) | 텐서, 미분, 컴파일, 실행, 저장 |
| [언어 모델 튜토리얼](docs/tutorial-gpt.md) | GPT 학습과 샘플링 |
| [진단 시스템](docs/diagnostics.md) | 오류 형식과 코드 목록 |
| [API 레퍼런스](docs/api.md) | 공개 API 전체 |
| [성능](docs/performance.md) | PyTorch·JAX 대조 측정 |
| [비교와 공백](docs/comparison.md) | 없는 기능 목록 |

**백엔드:** CPU와 Metal(Apple GPU)을 지원합니다. CUDA는 아직 없습니다.

## 빠른 시작

```bash
uv venv --python 3.13 .venv
```

```bash
uv pip install --python .venv/bin/python maturin numpy pytest
```

```bash
.venv/bin/maturin develop --release
```

```bash
.venv/bin/python examples/gpt.py --steps 3000
```

## 테스트

```bash
cargo test --release
```

```bash
cargo test --release --features portable
```

```bash
uv pip install --python .venv/bin/python torch
```

```bash
.venv/bin/python -m pytest tests -q
```

PyTorch가 없으면 `tests/test_framework.py`와 대조 테스트는 건너뜁니다.

## 벤치마크

```bash
uv pip install --python .venv/bin/python torch jax
```

```bash
.venv/bin/python benches/compare.py
```

```bash
.venv/bin/python benches/gpt_compare.py
```
