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
