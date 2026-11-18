# 손글씨 숫자 분류 서비스

클라우드 MLOps 개인 대체 실습 프로젝트입니다. 8×8 픽셀로 정규화된 단일 숫자 이미지를 다루는 교육용 도구입니다. JSON 문서를 입력하면 명령줄에서 예측한 숫자(0~9)와 모델 버전을 JSON으로 반환합니다. 한 건씩 처리하는 용도이므로 상시 웹 서버나 데이터베이스 없이 실행합니다.

## 데이터와 모델

[scikit-learn 1.8의 load_digits](https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_digits.html)는 [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)의 원본 테스트 부분을 제공하며, UCI는 데이터를 CC BY 4.0으로 공개합니다. 원본 수집자는 E. Alpaydin과 C. Kaynak입니다. 한 행은 손글씨 숫자 이미지 한 장과 그 정답 숫자를 의미합니다. 각 8×8 셀은 원본 이미지에서 대응하는 4×4 영역의 켜진 픽셀 수로, 단위는 픽셀 개수이고 범위는 0~16입니다. 원본 전체를 내려받을 필요 없이 라이브러리에서 1,797건을 불러옵니다. 전체 데이터 파일은 저장소에 복제하지 않습니다. 예제 JSON은 직접 만든 숫자 0 모양입니다.

클래스 비율을 유지하며 난수 시드 42로 이 1,797건을 학습 1,347건과 평가 450건으로 분할했습니다. 입력은 이미지 64개 셀을 행 우선 순서로 펼쳐 16으로 나눈 값입니다. 이 고정 범위는 데이터 정의에서 가져왔습니다. 정답 숫자와 데이터 행 번호는 예측 입력에서 제외합니다. JSON 오류, 필수 필드 누락, 8×8 형태 불일치, 범위 밖 값, 정수가 아닌 값은 종료 코드 2로 거부합니다.

모델은 scikit-learn(BSD-3-Clause)의 로지스틱 회귀를 직접 학습한 `digits-logreg-v1`입니다. C=1.0, 최대 반복 2,000회 설정을 사용합니다. 외부 사전학습 가중치가 필요 없고, 학습 파일은 실행할 때 다시 생성합니다. 임의 출처의 joblib 파일은 역직렬화하지 말고 이 코드에서 생성한 파일을 사용합니다.

## 설치 및 실행

Git과 Python 3.12가 필요합니다. 저장소 또는 압축 해제한 폴더에서 아래 명령을 실행합니다. GitHub에서 복제하는 경우 `git clone <공개-저장소-URL>`을 먼저 실행한 뒤 복제한 폴더로 이동합니다. 압축본은 코드 커밋 검증을 위해 동봉된 Git 번들에서 먼저 복제합니다.

```bash
git clone digit-classifier.bundle digit-classifier
cd digit-classifier
```

저장소에서 다음 명령을 실행합니다.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py runs/verification
```

실행 폴더는 새 경로를 사용합니다. Windows에서는 `.venv/bin/python` 대신 `.venv/Scripts/python.exe`를 사용합니다. 단건 추론은 아래 두 명령으로 확인합니다.

```bash
.venv/bin/python service.py train --model runs/verification/digits.joblib
.venv/bin/python service.py predict --model runs/verification/digits.joblib --input examples/digit.json
```

정상 예측은 `{"digit": 0, "model_version": "digits-logreg-v1"}`입니다. 위 예제로 수동 생성한 가중치 파일은 검증 후 삭제할 수 있습니다.

## 평가 결과

평가 자료에서 정확도 90% 이상이며 최빈 클래스 기준 모델을 넘으면 품질 검사를 통과합니다. 정상 동작은 고정 예제 0의 출력으로 별도 검사합니다.

| 방법 | 정답 수 / 평가 건수 | 정확도 | 매크로 F1 |
| --- | ---: | ---: | ---: |
| 학습 자료의 최빈 클래스인 3을 반환 | 46 / 450 | 10.22% | 0.0185 |
| 로지스틱 회귀 | 433 / 450 | 96.22% | 0.9620 |

정확도는 정답 건수의 비율이고 매크로 F1은 숫자별 정밀도와 재현율의 조화 평균을 열 클래스에 동일 가중치로 평균 낸 값입니다. 모델은 같은 평가 자료에서 기준보다 387건 더 맞혔습니다. 혼동 행렬은 행이 실제 숫자, 열이 예측 숫자입니다.

오분류 사례는 행 번호 363(0부터 시작)입니다. 실제 1을 8로 예측했습니다. 아래에서 각 행의 기호 한 자리가 픽셀 하나이며 0~F는 0~15이고 `G`는 16입니다. 예제의 1은 중앙 획이 굵고 위아래가 넓어 중앙의 모양이 둥근 숫자에 가까웠을 가능성이 있습니다. 이는 픽셀을 관찰한 해석입니다.

```text
008GF800
01GGG200
02GGA000
02GGC000
06GGD000
01GGB000
00GGA000
007FF000
```

같은 공개 데이터의 한 번의 분할로 검증했습니다. 사용자 또는 필기자 단위로 분리된 성능, 휴대전화 사진용 전처리, 여러 숫자가 있는 이미지, 거의 빈 이미지, 처리량은 측정하지 않았습니다.

## 클라우드 실행, 오류 복구, 재실행

최초 실행은 Codex Cloud의 관리형 Linux 환경에서 CPU로 수행했습니다. 리전은 공개된 환경 정보에서 확인할 수 없었습니다. 데이터는 버전 고정된 Python 패키지에서 로드하고, 모델과 요청은 일회성 임시 폴더에 생성했습니다. 결과 JSON은 `evidence/`에 복사해 보관했습니다. CPU에서 끝나는 짧은 배치 작업이므로 별도 VM, 상시 서버, 외부 저장소가 필요하지 않습니다.

- [최초 실행](evidence/codex-cloud/run.json)은 UTC 시각, Git 커밋, 모델 버전 및 SHA-256, Python과 패키지 버전, 입력 해시, 명령 및 출력 위치를 기록합니다. [평가 JSON](evidence/codex-cloud/evaluation.json)에 혼동 행렬과 오분류 이미지가 있습니다.
- [실패 입력](evidence/codex-cloud/input_before.json)은 빈 JSON 객체입니다. `predict`가 필수 `pixels` 필드를 읽지 못해 종료 코드 2를 반환했고 [오류 메시지](evidence/codex-cloud/failed_request.stderr.txt)를 남겼습니다. 이 요청에서는 예측을 생성하지 않았습니다. 예상치 못한 종료 코드는 파이프라인을 중단합니다.
- [수정 입력](evidence/codex-cloud/input_after.json)의 8×8 배열로 같은 경로의 요청을 교체하고 실패했던 명령을 다시 실행했습니다. [복구 요청](evidence/codex-cloud/recovered_request.json)과 [별도 정상 요청](evidence/codex-cloud/normal_request.json)은 숫자 0을 반환했습니다.
- 새 클론과 새 가상 환경에서 의존성을 재설치하고 전체 절차를 재실행했습니다. [새 클론 실행 기록](evidence/clean-clone/run.json)과 [평가 결과](evidence/clean-clone/evaluation.json)를 보관했습니다.

## 공개 저장소와 재실행 설정

공개 저장소의 `main`에 소스와 실행 기록을 보관했습니다. [GitHub Actions 실행](https://github.com/PhilHeath04/cloud-mlops-digit-classifier/actions/runs/44223963148)은 Ubuntu 24.04 호스팅 러너에서 학습, 평가, 오류 재현과 복구, 새 클론 재실행을 완료했습니다. 실행 리전은 이 러너에서 지정할 수 없으며 실제 위치도 표시되지 않습니다.

- [클라우드 실행 기록](evidence/github-actions/cloud/run.json)과 [평가 결과](evidence/github-actions/cloud/evaluation.json)에서 평가 450건 중 모델 433건과 기준 모델 46건의 정답을 확인할 수 있습니다. 필수 입력 누락으로 종료 코드 2를 반환했고, 같은 명령을 다시 실행하자 숫자 0이 반환됐습니다.
- [새 클론 실행 기록](evidence/github-actions/clean-clone/run.json)과 [정리 결과](evidence/github-actions/clone-cleanup.json)에서 의존성 재설치, 재실행, 임시 모델과 복제 폴더 삭제를 확인할 수 있습니다.
- 재실행은 GitHub의 **Actions → Cloud digit classification → Run workflow**에서 시작합니다. 워크플로는 두 실행의 JSON과 로그를 `execution-evidence` 아티팩트에 담습니다.

## 권한, 비용 및 정리

Codex Cloud에서는 작업 폴더 쓰기와 PyPI 패키지 설치 접근만 사용했습니다. GitHub에 공개 게시할 때는 저장소를 만들고 main에 푸시할 권한이 필요합니다. 워크플로에는 저장소 읽기 권한(`contents: read`)만 부여했고 별도 시크릿은 사용하지 않습니다. 공개 저장소의 표준 GitHub 호스팅 러너를 사용하며 작업별 10분 제한, 아티팩트 7일 보존을 설정했습니다. 별도 사용량 과금 자원을 생성하지 않았습니다.

모델과 임시 요청 파일이 들어 있던 각 실행 폴더는 종료 후 경로를 다시 조회해 `still_exists: false`로 기록했습니다. 새 클론과 그 가상 환경도 삭제하고 [삭제 뒤 상태](evidence/clone-cleanup.json)를 확인했습니다. 코드, 예제, 평가 수치와 실행 기록은 재현과 제출 검토를 위해 보관합니다.

## 구현 범위 및 참고 자료

프로젝트 구현은 JSON 검증, 분할, 학습 및 추론 연결, 기준 모델과 비교, 실패·복구 검사, 로그 및 정리 검증, GitHub Actions 설정입니다. 학습 알고리즘과 지표 계산은 scikit-learn을 사용합니다. 코드 초안과 실행 문서는 Codex의 도움으로 작성했습니다.

- [데이터 출처와 이용 조건](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
- [scikit-learn 데이터 로딩 문서](https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_digits.html)
- [GitHub Actions 호스팅 러너 문서](https://docs.github.com/en/actions/concepts/runners/github-hosted-runners)
