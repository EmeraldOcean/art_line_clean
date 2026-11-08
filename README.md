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

`--smooth`를 주면 떨림 완화(M4)를 병합 뒤에 적용한다. 선을 새로 그리지 않고 원본 픽셀을 최대 1px 옮기므로
연필 질감은 그대로 남는다. 분기점·교차점·끝점·모서리는 움직이지 않는다.

```powershell
uv run python -m lineclean .\samples\input\rough.png -o .\output\clean.png --merge-ids 3,7 --smooth
```

`--render`를 주면 맨 마지막에 깔끔한 선화로 다시 그린다(M5). 획마다 원본 굵기 변화와 잉크 양을 유지한 채
연필 질감·들쭉날쭉한 가장자리·부스러기를 정리한다. 원본 잉크 밖에는 그리지 않는다.

```powershell
uv run python -m lineclean .\samples\input\rough.png -o .\output\clean.png --merge-ids 3,7 --smooth --render
```

## GUI (M7)
```powershell
uv run python -m lineclean.gui                              # 창을 연 뒤 PNG를 끌어다 놓기
uv run python -m lineclean.gui .\samples\input\test.png   # 바로 열기
uv run python -m lineclean.gui --browser                    # 창 대신 브라우저에서 (127.0.0.1)
```
- 왼쪽 원본 | 가운데 결과 | 오른쪽 파라미터. 슬라이더를 놓으면 다시 계산합니다 (바꾼 단계부터만).
- 보기: 결과 / 차이 / 위상 / 병합 후보. 확대·이동은 두 칸이 함께 움직입니다 (휠 확대, 끌어서 이동, 0 = 맞춤, 1 = 100%).
- 병합 후보 보기에서 선이나 번호를 클릭하면 그 후보를 고르거나 뺍니다 (`--merge-ids`와 같음).
- 파라미터 이름에 마우스를 올리면 설명이 나오고, 더블클릭하면 기본값으로 돌아갑니다. ●는 기본값과 다른 값입니다.
- 저장(⌘S / Ctrl+S)은 새 파일로만 합니다 (입력 폴더·기존 파일 거부). `_settings.toml`을 함께 저장하면
  그 파일 맨 위에 적힌 CLI 명령으로 같은 결과를 다시 만들 수 있습니다. 비생성 위반이 있으면 저장하지 않습니다.
- 시작 설정은 현재 폴더의 `lineclean.toml`(없으면 기본값)입니다.
- Windows는 WebView2(Edge) 런타임을 씁니다 (Windows 10/11에 보통 설치되어 있음).

## 파라미터 조정 (설정 파일)
모든 파라미터는 템플릿 [lineclean.default.toml](lineclean.default.toml)에 기본값·주석과 함께 정리되어 있다.
템플릿은 직접 고치지 말고 `lineclean.toml`로 복사해 쓴다. `lineclean.toml`은 내 설정 파일이라 git에 올라가지 않는다.

```powershell
Copy-Item lineclean.default.toml lineclean.toml
```

현재 폴더에 `lineclean.toml`이 있으면 자동으로 읽고(없으면 코드 기본값), 실행할 때 `설정: ...` 줄에 어느 파일을 썼는지 출력한다.
여러 설정을 비교할 때는 복사본을 더 만들어 `--config`로 지정한다. 바꾸고 싶은 값만 적어도 된다.

```powershell
New-Item -ItemType Directory -Force configs
Copy-Item lineclean.default.toml configs\try1.toml
uv run python -m lineclean .\samples\input\rough.png -o .\output\try1.png --merge-ids 3,7 --smooth --render --config configs\try1.toml
```

선 굵기를 바꾸려면 `[render] target_width_px`에 원하는 굵기(px)를 적는다(0이면 원본 굵기). 필압에 따른 굵기 변화는 비율대로 남고, 가늘어진 만큼 진하게 그려 잉크 양을 유지한다.

잘못된 키나 범위 밖 값은 실행 전에 이유와 함께 거부한다. 설정을 바꾸면 덧그림 후보 번호가 달라질 수 있으니 `--merge`로 번호부터 다시 확인한다.

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
