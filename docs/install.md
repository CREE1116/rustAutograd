# 설치

아직 패키지 저장소에 배포하지 않았으므로 소스에서 빌드합니다. Rust 1.80 이상과 Python 3.10 이상이 필요합니다.

```bash
uv venv --python 3.13 .venv
```

```bash
uv pip install --python .venv/bin/python maturin numpy pytest
```

```bash
.venv/bin/maturin develop --release
```

## 플랫폼

| 플랫폼 | 커널 | 상태 |
| --- | --- | --- |
| macOS (Apple Silicon) | CPU: Accelerate BLAS·vForce, rayon. GPU: Metal | 테스트와 벤치마크 수행 환경 |
| Linux, Windows | 순수 Rust 폴백, rayon | 폴백 코드는 macOS에서 `portable` 기능으로 테스트, 두 타깃 모두 `cargo check` 통과. 실제 OS에서 실행해 본 적은 없음 |

폴백 커널의 행렬곱은 BLAS보다 훨씬 느립니다. 폴백 경로를 직접 시험하려면 다음을 실행합니다.

```bash
cargo test --release --features portable
```

## 스레드 수

원소 연산과 배치 행렬곱은 rayon을 씁니다. `RAYON_NUM_THREADS`로 조절하며, 결과는 스레드 수와 무관하게 같습니다.

## 테스트

```bash
cargo test --release
```

```bash
.venv/bin/python -m pytest tests -q
```

## 문서 빌드

```bash
uv pip install --python .venv/bin/python mkdocs-material
```

```bash
.venv/bin/mkdocs serve
```
