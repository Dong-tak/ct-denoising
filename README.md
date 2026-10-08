# CT 노이즈 제거 — 첫 실습

## Jupyter 노트북으로 따라가기

`01_ct_read_and_pair.ipynb`를 VS Code에서 열고 오른쪽 위 커널 선택에서
이 프로젝트의 `.venv/Scripts/python.exe`를 선택합니다.
위에서 아래로 Shift+Enter로 실행하거나 모두 실행을 누릅니다.
기존 두 Python 실습의 내용을 설명과 실행 셀로 나눴으며, 별도의 기존 CSV 없이 실행됩니다.
결과는 `outputs/notebook/<환자 ID>/`에 저장합니다.
`PATIENT`를 바꾸면 전체 셀을 다시 실행하세요.

현재 단계: 정상선량·저선량 DICOM을 같은 환자와 같은 촬영 위치로 짝짓고 시각화합니다.
학습은 아직 하지 않습니다. 원본 데이터는 수정하지 않습니다.

다른 환자를 보려면 노트북 첫 코드 셀의 `PATIENT = "L067"`을 `"L096"`으로 바꾸고 모두 실행하세요.

## 코드 읽는 순서

1. `scan_patient`: 환자 파일을 찾고 DICOM 헤더의 정보를 읽습니다.
2. `same_geometry`, `pair_slices`: 위치·방향·픽셀 간격·두께·커널을 비교합니다.
3. `read_hu`: 픽셀에 RescaleSlope를 곱하고 RescaleIntercept를 더해 HU로 바꿉니다.
4. 비교 셀: 가운데 슬라이스 한 쌍을 같은 밝기 범위로 저장합니다.

## 결과

`outputs/notebook/<환자 ID>/preview.png`: 저선량 / 정상선량 / 차이 영상.
`pairs.csv`: 메타데이터로 확인한 파일 짝 목록.
`report.json`: 슬라이스 개수, 재구성 조건, 읽기 오류.

HU [-160, 240]은 화면 표시 범위입니다. 학습용 정규화 범위와 구분합니다.
가운데 단면은 특정 장기나 병변을 선택한 단면이 아닙니다.
정상선량 영상은 비교 기준이며, 차이 영상 전체를 순수 노이즈라고 해석하지 않습니다.

## 데이터 출처

[AAPM/Mayo README](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/file/858370564530)

Mayo Grand Challenge의 Training_Image_Data/3mm B30, 정상선량과 시뮬레이션 저선량 영상.
다음 단계는 환자 단위 데이터 분리와 작은 PyTorch 모델 구현입니다.
