# art_line_clean — Claude Code 규칙

러프 RGBA PNG를 원본 형태를 보존한 채 선화로 정리하는 로컬 Python 도구.
새 그림을 생성하지 않는다. 원본 선을 분석·정리할 뿐이다.

## 상세 문서
- 요구사항: @docs/REQUIREMENTS.md
- 설계·검증: @docs/DESIGN.md
- 진행 상황: @docs/ROADMAP.md
- 결정 기록: @docs/DECISIONS.md

## 절대 원칙
1. 원본에 없는 선을 만들지 않는다 (출력 잉크 ⊆ 원본 잉크 + 허용치). 테스트로 확인한다.
2. 입력 파일·`samples/input/`은 읽기 전용. 출력은 항상 별도 경로, 덮어쓰기 금지.
3. 애매하면 합치거나 지우지 않는다.
4. 선 구조 정리(A~C)와 굵기·가장자리 정리(D)는 별도 모듈이다.

## 작업 방식
- 마일스톤은 Plan Mode로 범위·완료 조건을 먼저 제시하고 승인 후 구현한다.
- 현재 마일스톤 범위 밖 기능을 구현하지 않는다.
- docs/ 내용과 DECISIONS의 결정은 임의로 바꾸지 않는다. 변경은 이유와 함께 제안한다.
- 의존성 추가 전 사용자에게 확인한다. `uv add`만 사용, pip 직접 사용 금지.
- 완료 보고에는 pytest 결과, 비교 시트 경로, 미해결 문제를 포함한다.
  확인하지 못한 항목이 있으면 "완료"라고 하지 않는다.
- 마일스톤 종료 시 ROADMAP 상태 갱신, 커밋 메시지 제안.

## 명령 (PowerShell)
- 테스트: `uv run pytest`
- 실행: `uv run python -m lineclean <입력.png> -o <출력.png>`
- 경로는 코드에서 `pathlib`로만 다룬다 (Windows/macOS 공통).
