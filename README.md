# art_line_clean

직접 그린 러프(RGBA PNG)를 원본 선의 위치·형태·비율을 유지한 채 깔끔한 선화로 정리하는 개인용 로컬 도구.
생성형 AI로 다시 그리는 방식이 아니라, 원본 선을 분석해 정리한다.

## 상태
초기 개발 중 — 진행 상황은 [docs/ROADMAP.md](docs/ROADMAP.md) 참고.

## 설치 (PowerShell)
```powershell
uv sync
```

## 사용
```powershell
uv run python -m lineclean .\samples\input\rough.png -o .\output\rough_clean.png
# 비교 시트(입력 | 출력 | 차이)는 output\ 에 함께 저장된다.
```

기본은 항등(그대로 복사) 처리다. 덧그은 선 병합(M2)은 두 단계로 쓴다.

```powershell
# 1) 후보 보기: 확실한 덧그림(겹침 80% 이상)만 자동으로 합치고, 나머지 후보에는 번호를 붙여 보여 준다
uv run python -m lineclean .\samples\input\rough.png -o .\output\step1.png --merge
#    → output\step1_merge_overlay.png (후보 번호), output\step1_merge_log.csv (후보별 근거)

# 2) 고르기: 합칠 번호만 지정한다 (자동 병합은 하지 않음, 원본에서 다시 계산)
uv run python -m lineclean .\samples\input\rough.png -o .\output\step2.png --merge-ids 3,7,12
```

처리 후에는 항상 위상 검사(처리 영역 밖의 교차·분기·끝점이 그대로인지)를 출력한다.
`--topology-overlay`를 주면 획과 교차점(X 빨강, T/Y 주황, 끝점 초록)을 그린 `<출력>_topology.png`도 저장한다.

후보 번호는 같은 입력·같은 설정·같은 버전이면 항상 같다. 설정을 바꾸면 1단계부터 다시 확인한다.
고른 후보라도 안전 검사(대표선이 한 줄인지, 원본 잉크 밖에 그리지 않는지)에 걸리면 합치지 않고 이유를 출력한다.

## 개발 (PowerShell)
```powershell
uv run pytest                                          # 테스트
uv run python -m lineclean.synthetic .\samples\synthetic  # 합성 테스트 이미지 생성 (기존 파일은 건너뜀)
uv run python experiments\m1_extract.py                 # M1 선 그래프 추출 실험 → output\m1\
```

## 주요 기능 (계획)
- 덧그은 선을 대표 선 하나로 정리
- 미세한 떨림 완화 (형태는 유지)
- 교차·분기 구조 유지
- (선택) 굵기·가장자리 정리

## 폴더
- `src/lineclean/` 코드 · `tests/` 테스트 · `samples/input/` 원본(읽기 전용)
- `samples/synthetic/` 합성 테스트 이미지 · `output/` 결과물(git 제외)

## 문서
- [요구사항](docs/REQUIREMENTS.md) · [설계·검증](docs/DESIGN.md) · [로드맵](docs/ROADMAP.md) · [결정 기록](docs/DECISIONS.md)
