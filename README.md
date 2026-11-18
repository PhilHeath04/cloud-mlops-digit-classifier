# 손글씨 숫자 분류

클라우드 MLOps 개인 대체과제입니다. 손글씨 숫자 한 장을 8×8 픽셀 배열로 입력하면, 어떤 숫자인지 0부터 9까지 중 하나를 알려줍니다. 숫자 하나씩 확인하는 실습이라 웹 화면 없이 명령줄에서 JSON을 넣고 결과를 받는 방식으로 구성했습니다.

## 데이터와 모델

scikit-learn의 `load_digits()`로 숫자 이미지 1,797장을 불러왔습니다. E. Alpaydin과 C. Kaynak이 수집한 UCI 손글씨 숫자 데이터의 일부이며, 원본 데이터는 CC BY 4.0으로 공개되어 있습니다. 이미지 한 장에는 정답 숫자 하나가 붙어 있습니다. 배열의 각 칸은 원본 4×4 구역에서 켜진 픽셀 수라서 0부터 16까지의 정수입니다. [UCI 데이터 출처](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) · [scikit-learn 로딩 문서](https://scikit-learn.org/1.8/modules/generated/sklearn.datasets.load_digits.html)

숫자별 비율을 유지하면서 시드 42로 학습용 1,347장과 평가용 450장을 나눴습니다. 픽셀 64개를 순서대로 펼쳐 각각 16으로 나눈 값을 입력으로 사용했습니다. 정답 숫자와 데이터 행 번호는 모델에 입력하지 않았습니다.

모델은 scikit-learn(BSD-3-Clause)의 로지스틱 회귀입니다. 설정은 `C=1.0`, 최대 반복 횟수는 2,000회이고 모델 버전은 `digits-logreg-v1`입니다. 저장소를 받아서 실행하면 데이터는 설치한 패키지에서 불러오고 모델 파일은 그때 생성됩니다.

## 실행 방법

Git과 Python 3.12가 필요합니다. 저장소를 복제한 뒤 아래 순서로 실행하면 됩니다.

```bash
git clone https://github.com/PhilHeath04/cloud-mlops-digit-classifier.git
cd cloud-mlops-digit-classifier
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py runs/verification
```

Windows에서는 `.venv/bin/python` 대신 `.venv/Scripts/python.exe`를 사용하면 됩니다. 다시 실행할 때는 `runs/verification-2`처럼 새 경로를 지정합니다. 학습부터 평가, 정상 입력과 오류 입력 검사까지 실행되고 결과는 지정한 폴더에 저장됩니다.

숫자 하나만 예측하려면 아래처럼 실행합니다. 예제 입력은 숫자 0 모양이고 결과는 `{"digit": 0, "model_version": "digits-logreg-v1"}`입니다. 확인이 끝나면 여기서 만든 모델 파일은 삭제해도 됩니다.

```bash
.venv/bin/python service.py train --model runs/verification/digits.joblib
.venv/bin/python service.py predict --model runs/verification/digits.joblib --input examples/digit.json
```

JSON이 깨져 있거나 `pixels`가 빠진 경우, 배열이 8×8이 아니거나 0~16 범위의 정수가 아닌 값이 들어오면 오류를 출력하고 종료 코드 2를 반환합니다.

## 평가 결과

같은 이미지 450장으로 두 방법을 비교했습니다. 모델의 정확도가 90% 이상이고 기준 방법보다 높으면 통과로 정했습니다. 서비스가 실제로 숫자를 반환하는지는 예제 입력으로 따로 확인했습니다. 매크로 F1은 숫자별 F1 점수를 구해 열 숫자에 같은 비중으로 평균한 값입니다.

| 방법 | 맞힌 수 | 정확도 | 매크로 F1 |
| --- | ---: | ---: | ---: |
| 학습 데이터에서 가장 많았던 3만 반환 | 46 / 450 | 10.22% | 0.0185 |
| 로지스틱 회귀 | 433 / 450 | 96.22% | 0.9620 |

틀린 17건 중 인덱스 363(0부터 시작)은 정답이 1인데 8로 분류됐습니다. 픽셀 배열을 보면 가운데 획이 굵고 위아래가 넓습니다. 이 부분 때문에 8로 분류됐을 가능성이 있지만 픽셀 모양만 보고 내린 추정입니다. 평가 자료는 같은 데이터셋에서 한 번 나눴기 때문에 다른 필기자의 글씨나 휴대전화로 찍은 사진에서도 같은 정확도가 나오는지는 확인하지 못했습니다. [평가 결과와 혼동 행렬](evidence/github-actions/cloud/evaluation.json)

## 클라우드 실행과 오류 복구

한 번 실행하는 데 오래 걸리지 않는 작업이라 상시 서버 없이 GitHub Actions를 사용했습니다. Ubuntu 24.04 호스팅 러너에서 CPU로 학습과 예측을 실행했습니다. 러너의 실제 리전은 표시되지 않았습니다. 실행 시각, Git 커밋, 모델 버전과 해시, 입력과 설정은 `run.json`에 있습니다. 모델과 요청은 임시 폴더에 만들고 결과 JSON과 로그는 저장소에 보관했습니다. [Actions 실행](https://github.com/PhilHeath04/cloud-mlops-digit-classifier/actions/runs/44223963148) · [실행 기록](evidence/github-actions/cloud/run.json)

오류를 확인할 때는 요청 파일에 빈 JSON 객체 `{}`를 넣었습니다. `필수 필드 pixels가 없습니다.`가 출력됐고 예측은 생성되지 않았습니다. 같은 요청 파일을 예제의 8×8 배열로 바꿔 똑같은 명령을 실행하니 숫자 0이 나왔습니다. 원래 예제 파일도 별도로 실행해 같은 결과를 확인했습니다. 예상과 다른 종료 코드가 나오면 다음 단계는 중단됩니다. [실패 로그](evidence/github-actions/cloud/failed_request.stderr.txt) · [수정 후 결과](evidence/github-actions/cloud/recovered_request.json)

새 폴더에 저장소를 복제하고 새 가상 환경에서 패키지를 다시 설치한 뒤 전체 절차를 실행했습니다. 평가 결과와 예제 예측은 첫 실행과 같았습니다. GitHub에서 **Actions → Cloud digit classification → Run workflow**를 누르면 다시 실행할 수 있습니다. [새 폴더 실행 기록](evidence/github-actions/clean-clone/run.json) · [Codex Cloud 최초 실행](evidence/codex-cloud/run.json)

## 권한, 비용, 정리

GitHub에 코드를 올리려면 저장소 쓰기 권한이 필요합니다. Actions에는 저장소 읽기 권한만 설정했고 별도 시크릿은 쓰지 않았습니다. 공개 저장소의 표준 러너를 사용하고, 작업 제한을 10분, 아티팩트 보관 기간을 7일로 설정했습니다. 별도로 과금되는 서버나 저장소는 만들지 않았습니다.

실행 후 모델과 요청이 담긴 임시 폴더, 새 클론과 그 가상 환경이 삭제됐는지 경로를 확인했습니다. 다시 확인할 수 있도록 코드, 예제 입력, 평가 결과와 로그를 저장소에 남겼습니다. [정리 확인](evidence/github-actions/clone-cleanup.json)

`service.py`에서 입력 확인과 학습·평가·예측을 처리하고, `run.py`에서 순서대로 실행하며 결과를 기록합니다. 로지스틱 회귀 학습과 점수 계산에는 scikit-learn을 사용했습니다. 코드와 README 초안 작성에 Codex의 도움을 받았습니다.
