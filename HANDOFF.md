# 다른 컴퓨터에서 이어서 하기

기록 기준: 2026-10-08. 저장소: https://github.com/Dong-tak/ct-denoising (public).

## 목표와 현재 위치

정상선량·시뮬레이션 저선량 복부 CT를 이용한 노이즈 제거 포트폴리오.
설명과 실습을 통해 Python/PyTorch를 배우며, 작은 CNN과 평가·ONNX·비교 데모까지 진행한다.

프로젝트 순서:
1. DICOM 읽기와 한 환자 비교
2. 환자 10명 전체 검사와 전체 영상 짝 목록
3. 환자 단위 train/val/test 분리
4. HU 정규화와 동일 좌표의 128×128 패치
5. PyTorch Dataset과 DataLoader
6. 모델 적용 전 검증 성능(MAE/PSNR/SSIM)
7. 작은 잔차 CNN
8. 소수 패치 학습 검사 후 전체 학습·검증
9. 최종 테스트와 실패 분석
10. ONNX 및 비교 데모, 결과 문서

실제 노트북에는 17번 `PyTorch Dataset과 DataLoader` 절까지 있다.
6단계 비교 기준 평가 코드는 대화에서 안내했으나, 기록 시점에 노트북의 해당 절과
`outputs/baseline` 결과는 확인되지 않았다. 완료되었다고 가정하지 말고 먼저 확인한다.
다음 작업은 5단계 실행 성공 확인 → 6단계 검증 비교 기준 계산 → 7~8단계 모델·학습이다.
한 번에 두 단계씩 셀 추가 위치와 실행 방법을 안내한다.

## 데이터 다시 준비하기

Git에는 원본 데이터가 없다. 기존 컴퓨터의 `data/raw`를 별도 복사하거나 다음 자료를 다시 받는다.

- 설명: https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/file/858370564530
- 다운로드: https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/145241239366
- Mayo_Grand_Challenge / Patient_Data / Training_Image_Data / 3mm B30
- full_3mm.zip, quarter_3mm.zip. 두 압축파일 합계 약 1.4GB.

압축 해제 구조:
```
data/raw/full/full_3mm/L067/full_3mm/*.IMA
data/raw/quarter/quarter_3mm/L067/quarter_3mm/*.IMA
```
다른 사례: L096, L109, L143, L192, L286, L291, L310, L333, L506.
PatientID는 Anonymous이므로 사례는 폴더 ID로 구분한다.
L067 검증 결과는 각 선량 224장, 224쌍, 512×512, 3mm, B30f였다.

## 실행 환경

프로젝트의 Python은 3.12 계열. 새 컴퓨터에는 별도 Python 3.12 설치가 필요하다.
아래는 Windows PowerShell 예시이며 Python 설치 경로/명령은 실제 설치에 맞춘다.
```
git clone https://github.com/Dong-tak/ct-denoising.git
cd ct-denoising
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
기존 `.venv`를 복사하지 않는다. CUDA PyTorch는 새 컴퓨터의 GPU·드라이버에 맞춰 따로 확인한다.
현재 requirements.txt만으로 torch/scikit-image 설치가 보장되는지 확인해야 한다.
VS Code에서 노트북을 열고 이 프로젝트의 `.venv/Scripts/python.exe`를 커널로 선택한다.

## 다른 PC에서 반드시 다시 생성할 것

- `outputs/data_audit/all_pairs.csv`: 절대 경로가 들어 있어 새 컴퓨터에서 재생성한다.
- `data/splits`: Git 제외 대상이므로 노트북에서 분할 셀을 다시 실행한다.
- 최초 분할: train L067,L096,L109,L143,L192,L286,L291 / val L310 / test L333,L506.
- HU 제한 범위 -1000~2000, 정규화 0~1, 학습 패치 128×128, batch 4.
- 검증·테스트는 전체 단면. 지표 data_range=1, 배경 포함 여부 등 조건을 고정한다.

## 새 대화에 붙여 넣을 문장

> 이 저장소의 AGENTS.md와 HANDOFF.md를 읽고 실제 노트북의 진행 상황을 확인해줘.
> CT 노이즈 제거 프로젝트를 배우며 진행 중이고, 설명은 한국어로 해줘.
> 한 번에 두 단계씩 현재 위치, 추가할 셀, 코드의 의미, 실행 방법과 완료 기준을 알려줘.
> PyTorch Dataset 실행 확인과 검증 비교 기준 평가부터 이어가자.

이 문서는 작업 맥락의 인수인계다. 기존 대화의 전체 이력이나 UI 세션이 자동으로 이전되는 것은 아니다.
