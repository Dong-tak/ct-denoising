# 저선량 CT 복원 실습

`01_ct_read_and_pair.ipynb`를 열고 프로젝트의 `.venv` 커널에서 위에서 아래로 실행합니다. 현재 구조는 7개 장과 77개 코드 셀입니다.

| 장 | 목적 | 주요 코드 |
|---|---|---|
| 1. CT 파일을 숫자로 읽기 | 파일 주소 → DICOM → HU 배열 | `read_hu` |
| 2. 데이터 검사와 환자 분리 | 같은 사례·촬영 조건 검사, 환자 단위 분할 | `scan_patient`, `pair_slices`, `splits` |
| 3. 모델 입력 준비 | 정규화 → 패치 → Tensor → 배치 | `CTPairDataset`, `train_loader` |
| 4. 보정 전 기준 성능 | 저선량 그대로의 점수 기록 | `calculate_metrics`, `baseline_results` |
| 5. 모델 만들고 학습하기 | 고정 패치 검사, 전체 학습·중간 검증·저장 | `SmallResidualCNN`, `train_one_epoch` |
| 6. 모델 평가와 결과 해석 | 검증 결과 해석 후 별도 환자 최종 테스트 | `load_trained_model`, `test_rows` |
| 7. ONNX 변환과 비교 데모 | 내보내기, 출력 일치, CPU 측정, 독립 데모 | `InferenceSession`, `show_demo` |

장 안에서는 작업 이름을 Ctrl+F로 찾습니다. `def`와 `class`는 처리 방법을 등록하고, 호출 셀이 실제 계산합니다. 함수의 입력·반환값과 배열 크기를 먼저 확인하세요.

첫 설정 셀의 `PATIENT`는 초기 한 환자 비교에 사용합니다. 전체 학습의 환자 분할은 2장에서 정합니다. 전체 학습 실행은 `outputs/cnn_v1`을 갱신합니다. 저장 모델만 보고 싶으면 7장의 **독립 ONNX 데모**를 사용합니다. 원본 데이터·CSV·ONNX 파일은 필요합니다.

## 데이터와 결과

원본 CT는 `data/raw/`, 환자 분할 경로 목록은 `data/splits/`에 있습니다. HU 학습 범위는 [-1000,2000], 화면 표시 범위는 [-160,240]입니다. `outputs/notebook/<환자>/`에는 초기 비교, `outputs/cnn_v1/`에는 학습·평가·내보내기 결과가 저장됩니다. 원본 데이터·가중치·개인 대화 로그는 Git에 올리지 않습니다.

## 읽기 자료

- `docs/book/index.html`: 모든 학습 페이지와 함수로 이동하는 전체 목차. 각 HTML 위쪽에도 페이지 선택 메뉴가 있습니다.
- `docs/book/CT_Project_Overview_ELI5.html`: 노트북과 동일한 7장 그림 지도
- `docs/book/CT_Code_Walkthrough_ELI5.html`: 실제 함수·클래스·메서드 정의 38개, 해설 114단계. 원문과 입력·출력·호출 관계를 설명하고 함수별 직접 이동을 지원합니다.
- `docs/book/CT_Data_Flow_ELI5.html`: 파일에서 학습까지 데이터 흐름
- `docs/book/CT_AI_Code_Book.html`: 7장 전체 코드 교재
- `docs/book/CT_AI_Reading_Guide.html`: 사전 지식과 작업별 풀이

## 저장소에 남긴 파일

```text
01_ct_read_and_pair.ipynb     실행·학습 코드
README.md                   프로젝트 안내
requirements.txt            Python 패키지 목록
.gitignore                  데이터·가상환경·모델 업로드 제외
docs/book/*.html            전체 목차·개요·함수 해설·교재·사전 지식
```

로컬의 `data/`(원본 CT·분할 목록), `.venv/`, `outputs/data_audit/`, `outputs/baseline/`, `outputs/cnn_v1/`(저장 모델·평가 결과·ONNX)은 보존하며 Git에 올리지 않습니다. 저장 모델을 이용하는 독립 데모에는 원본 CT·val.csv·ONNX·export_config.json이 필요합니다. 전체 학습을 위에서부터 실행하면 초기 비교 결과와 이후 결과 폴더를 다시 생성합니다.

HTML은 `docs/book/index.html`에서 시작하세요. 공통 상단 메뉴로 페이지를 오갈 수 있고, 함수 해설에는 전체 38개 정의와 설명 114단계가 있습니다. Markdown 중복 교재·빌드 도구·검증 복사본·인수인계 문서는 저장소 밖 백업으로 옮겼습니다.

## 데이터 출처

[AAPM/Mayo README](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/file/858370564530)

Mayo Grand Challenge의 Training_Image_Data/3mm B30, 정상선량과 시뮬레이션 저선량 영상.
원 대회의 공개 학습 사례 10명을 이 프로젝트에서 학습 7명·검증 1명·테스트 2명으로 나눴습니다.
최종 테스트는 자체 분리한 두 환자의 455개 단면입니다. 모델 성능과 실행 시간의 실제 수치는 `outputs/cnn_v1`의 결과 파일에서 확인합니다.
