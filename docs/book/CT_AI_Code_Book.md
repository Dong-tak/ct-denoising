# CT 영상 복원으로 배우는 Python과 PyTorch

DICOM 한 장에서 CNN 학습과 ONNX 실행까지

2026-10-14 기준. 원본 노트북 31개 절, 59개 비어 있지 않은 코드 셀 전체 해설.

## 이 책을 읽는 방법

이 책은 01_ct_read_and_pair.ipynb의 실제 코드를 처음부터 끝까지 읽기 위한 개인 교재다. Python으로 파일을 다뤄 본 경험은 있지만 PyTorch의 학습 과정은 낯선 독자를 기준으로 썼다. DICOM 한 장이 숫자 배열이 되고, Tensor로 묶이고, CNN을 통과하고, 평가와 ONNX 실행까지 이어지는 과정을 따라간다.

처음부터 함수 이름을 외울 필요는 없다. 각 코드를 읽으며 입력의 자료형과 크기, 수행하는 계산, 출력의 자료형과 크기를 설명해 보자. 특히 파일 경로와 영상 배열, 영상 배열과 모델 파라미터를 구분하는 것이 중요하다. 코드를 실행할 수 있다는 것과 코드의 판단 근거를 설명할 수 있다는 것은 서로 다른 능력이다. 이 책에서는 두 가지를 함께 연습한다.

본문의 원본 코드는 2026년 10월 14일에 저장된 노트북에서 그대로 가져왔다. 절 번호 1부터 31까지를 책의 장 번호와 맞췄으며, 각 코드에는 원본 노트북에서의 코드 셀 순번을 붙였다. 이 순번은 Jupyter의 In[숫자]와 다르다. In 번호는 실행할 때마다 바뀌지만 책의 순번은 저장된 셀의 순서를 센 값이다. 마지막 빈 코드 셀은 설명할 실행문이 없으므로 수록 대상에서 제외한다.

모든 실습을 새로 학습할 필요는 없다. 저장된 결과를 읽는 부분과 파라미터를 변경하는 부분을 구분하며 진행한다. 특히 전체 학습 셀을 다시 실행하면 새 모델을 만들고 best_model.pt를 덮어쓸 수 있다. 읽기 연습을 할 때는 원본 노트북을 보존하고 별도의 메모용 셀에 작은 확인 코드를 추가하는 편이 좋다.

독학 순서는 장의 목표를 읽고, 개념 설명을 읽고, 원본 코드를 해설과 함께 살피고, 확인 문제에 답하고, 해설을 펼치는 방식이다. 한 번에 한두 장씩 읽어도 좋다. 장 끝에서 자신의 말로 한 문장을 설명할 수 있으면 다음 장으로 넘어간다. 처음 읽을 때 이해하지 못한 용어는 부록의 용어 사전에서 확인한다.

이 책의 의료영상은 2016 AAPM Low Dose CT Grand Challenge의 공개 학습 사례 중 3mm B30 복부 CT다. 정상선량 영상은 비교 기준이며 완전한 무잡음 영상은 아니다. 환자 quarter-dose 영상은 투영 데이터에 노이즈를 삽입해 정상선량의 25% 선량 수준을 시뮬레이션한 것이다. 실제로 환자를 두 번 촬영해서 만든 쌍으로 설명하면 안 된다. 우리는 원 대회의 공식 테스트 20명을 사용한 것이 아니라 공개 학습 사례 10명을 자체적으로 학습 7명, 검증 1명, 테스트 2명으로 나눴다.

## 준비 장 Python과 노트북의 기본 규칙

### 변수는 이름표다

train_rows라는 글자를 썼다고 데이터가 생기는 것은 아니다. 먼저 대입문이 실행되어야 이름과 값이 연결된다. a = 3에서 오른쪽의 3을 준비하고 왼쪽 이름 a에 연결한다. a = a + 1은 기존 a를 읽어 계산한 다음 결과를 같은 이름에 다시 연결한다. Python의 등호는 수학에서의 양변이 영원히 같다는 선언과 역할이 다르다. NameError는 현재 실행 공간에 그 이름이 없다는 뜻이다.

```python
a = 3
a = a + 1
print(a)  # 4
```

### 리스트 딕셔너리 집합을 구분하자

리스트는 순서대로 값을 담으므로 train_rows[0]으로 첫 항목을 고른다. 딕셔너리는 키로 값을 찾아 row["quarter_path"]처럼 파일 경로를 꺼낸다. 집합은 중복을 제거하고 교집합을 검사한다. train_ids & test_ids가 비어 있어야 환자 분할에 중복이 없는 것이다. 괄호도 구분하자. []는 리스트나 인덱싱, {}는 딕셔너리 또는 집합, ()는 함수 호출이나 튜플에 등장한다. 빈 {}는 집합이 아니라 딕셔너리이며 빈 집합은 set()으로 만든다.

```python
rows = [{"patient_id": "L067", "quarter_path": "example.IMA"}]
row = rows[0]
print(row["patient_id"])
print(set(["L067", "L067", "L096"]))
```

### 반복문과 컴프리헨션

for row in rows는 리스트의 항목을 하나씩 row에 연결해 들여쓴 문장을 실행한다. [row["patient_id"] for row in rows]는 각 항목에서 환자 ID를 뽑아 새 리스트를 만든다. 처음에는 압축된 표현을 보통 반복문으로 풀어 써도 된다. 조건이 붙으면 그 조건을 통과한 항목만 담는다. for가 두 번 등장하는 딕셔너리 컴프리헨션도 바깥 반복문부터 읽는다.

```python
ids = []
for row in rows:
    ids.append(row["patient_id"])
# 같은 의미
ids = [row["patient_id"] for row in rows]
```

### 함수는 정의한 뒤 호출한다

def read_hu(path):는 함수의 처리 방법을 등록한다. 이 줄을 실행하는 순간 파일을 읽는 것은 아니다. read_hu("파일 경로")처럼 호출하면 매개변수 path에 전달한 값을 연결하고 함수 몸체를 실행한다. return은 결과를 호출한 쪽에 전달하고 함수를 끝낸다. print는 화면에 쓰는 동작이며 return을 대신하지 않는다. 함수 안의 지역 변수는 보통 밖에서 직접 읽을 수 없다.

```python
def add_one(value):
    return value + 1

result = add_one(10)
print(result)  # 11
```

### 클래스는 관련 상태와 동작을 묶는다

class는 객체의 구조를 정의한다. **init**은 객체를 만들 때 초기 상태를 준비한다. self.rows는 현재 객체에 저장한 목록이다. 객체가 여럿이면 각 객체가 자신의 rows를 가질 수 있다. **len**과 **getitem**처럼 양쪽에 밑줄이 붙은 메서드는 Python의 len(dataset), dataset[0] 같은 문법과 연결된다. PyTorch 모델 클래스의 forward는 입력을 출력으로 바꾸는 계산이다.

```python
class Box:
    def __init__(self, value):
        self.value = value

b = Box(3)
print(b.value)
```

### 배열의 shape를 읽는 법

NumPy 배열은 많은 숫자를 일정한 모양으로 담는다. (512,512)는 512행 512열이며 첫 축이 행, 다음 축이 열이다. image[100,200]은 100번 행과 200번 열의 숫자다. 인덱스는 0부터 시작한다. image[10:20,30:40]에서 끝 인덱스는 포함하지 않으므로 크기는 10×10이다. dtype은 각각의 숫자를 어떻게 저장하는지 나타내며 float32는 32비트 부동소수점 수다.

```python
import numpy as np
a = np.arange(12).reshape(3, 4)
print(a)
print(a.shape)
print(a[1, 2])  # 6
print(a[:2, :2])
```

### 커널은 실행 상태를 기억한다

ipynb 파일은 셀의 코드와 저장된 출력을 담고, 커널은 현재 실행 중인 Python 상태를 가진다. 화면에 과거 출력이 보여도 새 커널에 해당 변수가 있다는 보장은 없다. 커널을 재시작하면 변수·함수·클래스 정의가 사라진다. 저장된 CSV, 모델 파일, ONNX 파일은 디스크에 남는다. Ctrl+S는 파일을 저장하고 Shift+Enter는 셀을 실행한다. 두 동작은 다르다. 셀을 여러 번 실행하면 난수 상태가 진행되거나 모델을 새로 만들 수 있다.

```python
import sys
print(sys.executable)
print("train_rows" in globals())
# globals()는 현재 전역 이름 공간을 보여준다.
```

### 오류는 종류와 마지막 줄부터 읽는다

NameError는 정의 또는 실행 순서, FileNotFoundError는 파일 경로, ModuleNotFoundError는 현재 커널의 패키지 설치, KeyError는 딕셔너리 키, AssertionError는 코드가 요구한 조건, RuntimeError는 장치나 크기 등 다양한 실행 조건을 확인한다. 오류를 보고 전체 학습을 다시 실행하기 전에 원인을 만드는 가장 작은 셀을 찾는다. 커널이 바쁘면 다음 셀은 대기열에 들어간다. 먼저 실행 중단을 하고 print("커널 응답") 같은 짧은 셀로 확인한다.

```python
assert 2 + 2 == 4
# assert 조건이 거짓일 때 AssertionError가 난다.
```

## 1장 라이브러리와 프로젝트 경로

학습 목표: 실행 환경을 확인하고 프로젝트의 기준 폴더를 찾는다.

컴퓨터는 ct-denoising이라는 프로젝트를 자동으로 아는 것이 아니다. 코드가 현재 작업 폴더에서 출발해 requirements.txt 파일과 data/raw 폴더가 함께 있는 위치를 찾아 ROOT에 저장한다. ROOT는 이후 모든 상대 경로의 출발점이다. from pathlib import Path는 경로를 문자열 조작보다 명확하게 다루기 위한 준비다. Path 객체의 / 연산은 숫자 나눗셈이 아니라 경로 조합으로 정의되어 있다.

ROOT를 찾는 한 줄은 초보자에게 압축되어 보인다. [Path.cwd(), *Path.cwd().parents]는 현재 폴더와 상위 폴더들을 목록에 넣고, if 뒤 조건은 프로젝트 표시 파일·폴더가 있는 후보만 남긴다. next(..., None)은 첫 후보를 선택하고 없으면 None을 반환한다. 별표는 부모 경로 목록을 펼쳐 넣는다. 이는 파일을 읽거나 학습하는 코드가 아니라 경로를 정하는 코드다.

PATIENT는 첫 실습에서 살펴볼 사례 ID다. 저장된 원본은 L096으로 설정되어 있다. 이를 바꾸면 초반 한 환자의 출력 폴더가 달라지지만 이후 전체 환자 분할의 구성까지 자동으로 바뀌지는 않는다. mkdir(parents=True, exist_ok=True)는 필요한 부모 폴더를 만들고 이미 폴더가 있어도 오류를 내지 않는다. sys.executable은 현재 커널의 Python을 확인하는 가장 빠른 방법이다.

### 핵심 용어

| 이름    | 뜻                           |
| ------- | ---------------------------- |
| ROOT    | 프로젝트 루트의 Path 객체    |
| PATIENT | 처음 살펴볼 사례 폴더 이름   |
| OUTPUT  | 선택한 사례의 결과 저장 폴더 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 1 · 노트북 셀 위치 3

import로 도구를 준비하고 ROOT 후보를 찾는다. PATIENT는 L096으로 설정되어 있다. OUTPUT은 ROOT 아래 outputs/notebook/L096이다. 이 셀은 파일 탐색·함수 정의·학습 셀의 기준 상태를 만든다.

```python
from pathlib import Path
import sys
import csv
import json
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from IPython.display import display, Image

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "requirements.txt").is_file() and (p / "data/raw").is_dir()), None)
if ROOT is None:
    raise FileNotFoundError("ct-denoising 폴더에서 노트북을 열어 실행하세요.")
PATIENT = "L096"  # L096 등으로 바꾼 뒤 전체 셀을 다시 실행할 수 있습니다.
OUTPUT = ROOT / "outputs" / "notebook" / PATIENT
OUTPUT.mkdir(parents=True, exist_ok=True)
print("Python:", sys.executable)
print("프로젝트:", ROOT)
print("선택한 환자:", PATIENT)
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(type(ROOT))
print(ROOT / "data" / "raw")
print(sys.executable)
```

### 이해 확인 문제

1. 왜 ROOT를 먼저 찾을까?

해설: 뒤의 파일 경로를 한 기준점에서 만들기 위해서다.

2. Ctrl+S를 누르면 함수도 실행될까?

해설: 아니다. 파일 저장과 커널 실행은 다르다.

## 2장 파일 경로 찾기

학습 목표: 환자 폴더에서 CT 파일을 찾아 경로 목록으로 만든다.

patient_folder는 정상선량 사례 폴더다. rglob("\*.IMA")는 해당 폴더의 하위 폴더까지 재귀적으로 찾아 확장자가 IMA인 경로를 반환한다. sorted는 경로를 정렬한 리스트로 만든다. files에는 영상 픽셀 대신 파일 위치를 나타내는 Path 객체들이 들어 있다. len(files)는 파일 수이고 files[0]은 첫 경로다.

if not files는 빈 목록인지를 확인한다. 파일이 없는데 files[0]을 읽으면 IndexError가 나므로, 먼저 의미가 있는 FileNotFoundError를 낸다. f"...{patient_folder}"는 변수를 문자열에 넣는 f-string이다. 파일 이름 정렬은 사람이 보기 위한 편의일 뿐, 같은 위치의 정상·저선량을 짝지었다는 증거가 아니다. 실제 짝짓기는 뒤에서 촬영 정보를 검사한다.

### 핵심 용어

| 이름     | 뜻                      |
| -------- | ----------------------- |
| rglob    | 하위 폴더까지 파일 검색 |
| sorted   | 정렬된 새 리스트 생성   |
| files[0] | 첫 파일의 경로          |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 2 · 노트북 셀 위치 5

patient_folder를 조합하고 rglob로 IMA 경로를 찾는다. sorted의 결과는 리스트다. not files를 검사한 뒤 files[0]을 출력하므로 빈 목록을 먼저 처리한다.

```python
patient_folder = ROOT / "data/raw/full/full_3mm" / PATIENT
files = sorted(patient_folder.rglob("*.IMA"))
if not files:
    raise FileNotFoundError(f"IMA 파일이 없습니다: {patient_folder}")
print("검색한 폴더:", patient_folder)
print("찾은 파일 개수:", len(files))
print("첫 번째 파일:", files[0])
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(type(files))
print(type(files[0]))
print(files[0].is_file())
```

### 이해 확인 문제

1. 이 단계에서 픽셀을 읽었을까?

해설: 아직 아니다. 파일 경로만 찾았다.

2. 파일 이름 순서만으로 짝지어도 될까?

해설: 안 된다. 위치와 재구성 조건을 검사해야 한다.

## 3장 DICOM 파일 읽기

학습 목표: DICOM 객체와 헤더의 역할을 구분한다.

pydicom.dcmread(file_path)는 DICOM을 읽어 pydicom Dataset 객체를 만든다. 여기서 Dataset은 뒤에서 등장하는 PyTorch Dataset 클래스와 역할이 다르다. pydicom Dataset은 파일의 태그와 값들을 담는 객체다. ds.Rows처럼 속성을 통해 특정 태그의 값을 꺼낼 수 있다.

PatientID는 익명화된 이 자료에서 Anonymous일 수 있다. Modality의 CT는 영상 종류이며 Rows와 Columns는 저장된 배열의 크기다. ImagePositionPatient는 이미지 첫 픽셀의 물리적 위치를 나타내는 세 좌표이고, SliceThickness는 재구성 두께다. 단면 번호와 mm 좌표는 같은 개념이 아니다. 이 정보를 읽는 작업은 모델을 학습하는 과정과 독립적이다.

### 핵심 용어

| 이름                 | 뜻                                    |
| -------------------- | ------------------------------------- |
| ds                   | pydicom Dataset 객체                  |
| Rows Columns         | 행·열 개수                            |
| ImagePositionPatient | 환자 좌표계에서 이미지 첫 픽셀의 위치 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 3 · 노트북 셀 위치 7

첫 경로를 file_path로 받고 dcmread를 호출한다. ds 속성들은 촬영 정보를 꺼내는 접근이다. 이 단계의 ds는 다음 셀에서 재사용한다.

```python
file_path = files[0]
ds = pydicom.dcmread(file_path)
print("객체 타입:", type(ds))
print("환자 ID:", ds.PatientID)
print("영상 종류:", ds.Modality)
print("행과 열:", ds.Rows, ds.Columns)
print("단면 두께:", ds.SliceThickness)
print("영상 위치:", ds.ImagePositionPatient)
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(ds.Modality)
print(ds.Rows, ds.Columns)
print(ds.ImagePositionPatient)
```

### 이해 확인 문제

1. ds는 Tensor일까?

해설: 아니다. DICOM 정보를 담는 객체다.

2. PatientID만으로 10명을 구분할 수 있을까?

해설: Anonymous가 반복되므로 이 자료에서는 사례 폴더 ID를 사용한다.

## 4장 픽셀 배열 꺼내기

학습 목표: 픽셀 숫자 배열을 꺼내고 인덱싱한다.

ds.pixel_array는 저장된 PixelData를 해석해 NumPy 배열로 제공한다. 이 속성에 접근할 때 실제 픽셀 해석이 수행될 수 있다. pixels.shape는 행·열 수이고 dtype은 숫자의 저장 형식이다. 숫자는 CT의 물리적 HU로 변환하기 전의 저장 값이다.

pixels[:5,:5]는 왼쪽 위 5행·5열을 고른다. 작은 배열을 출력해 보면 영상이 실제로 숫자의 모음이라는 점이 분명해진다. min과 max는 전체 배열의 값 범위를 살핀다. 저장 정수 형식에서는 빼기나 계산 중 범위 문제가 생길 수 있으므로, 뒤에서는 float32로 변환해 HU를 계산한다.

### 핵심 용어

| 이름        | 뜻                       |
| ----------- | ------------------------ |
| pixel_array | 해석된 픽셀의 NumPy 배열 |
| shape       | 각 축의 길이             |
| dtype       | 각 숫자의 자료형         |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 4 · 노트북 셀 위치 9

ds.pixel_array로 실제 숫자 배열을 얻는다. .shape/.dtype와 .min()/.max()의 속성·메서드 문법을 구분하자. [:5,:5]의 두 콜론은 행·열 영역을 뜻한다.

```python
pixels = ds.pixel_array
print("배열 크기:", pixels.shape)
print("자료형:", pixels.dtype)
print("최솟값 / 최댓값:", pixels.min(), pixels.max())
print("왼쪽 위 5×5 픽셀:")
print(pixels[:5, :5])
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(pixels[0, 0])
print(pixels[100:103, 200:203])
```

### 이해 확인 문제

1. pixels[:5,:5] 크기는?

해설: (5,5)다.

2. shape와 dtype은 같은 정보일까?

해설: 아니다. shape는 모양, dtype은 숫자의 저장 형식이다.

## 5장 저장된 숫자를 HU로 바꾸기

학습 목표: 저장된 픽셀 값을 HU로 변환한다.

HU 변환은 저장 값에 slope를 곱하고 intercept를 더하는 선형 계산이다. 예를 들어 저장 값 1500, slope 1, intercept -1024라면 HU는 476이다. 실제 파일의 두 값을 읽어 사용해야 하며 모든 CT가 같은 변환값을 쓴다고 가정하면 안 된다. float(...)는 DICOM의 수치 값을 Python 실수로 명확히 변환한다.

astype(np.float32)는 배열의 모든 숫자를 실수로 바꾼다. 배열에 실수를 곱하고 더하면 각 픽셀에 같은 계산이 적용된다. 이것이 NumPy의 배열 연산이다. np.isfinite(hu)는 각 픽셀이 NaN이나 무한대가 아닌지 검사하는 불리언 배열이고 .all()은 전부 정상일 때 True다. assert는 잘못된 상태를 뒤로 넘기지 않도록 중단시킨다.

HU로 변환했다고 정상선량과 저선량의 오차가 사라지는 것은 아니다. 두 영상이 같은 물리적 수치 체계에서 비교될 준비를 한 것이다. 이 프로젝트의 입력은 이미 재구성된 영상이고, 투영 데이터로부터 CT를 재구성하는 알고리즘을 구현한 것은 아니다.

### 핵심 용어

| 이름             | 뜻                    |
| ---------------- | --------------------- |
| RescaleSlope     | 저장 값에 곱하는 값   |
| RescaleIntercept | 곱한 결과에 더하는 값 |
| isfinite         | 유한한 값인지 검사    |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 5 · 노트북 셀 위치 11

slope와 intercept를 실수로 바꿔 전체 배열에 선형 계산을 적용한다. [100,200]은 한 픽셀이다. 마지막 assert는 전체 결과가 유한한 숫자인지 확인한다.

```python
slope = float(ds.RescaleSlope)
intercept = float(ds.RescaleIntercept)
hu = pixels.astype(np.float32) * slope + intercept
print("Slope / Intercept:", slope, intercept)
print("HU 배열 크기:", hu.shape)
print("HU 최솟값 / 최댓값:", hu.min(), hu.max())
print("같은 픽셀 [100, 200]:", pixels[100, 200], "→", hu[100, 200])
assert np.isfinite(hu).all(), "HU 배열에 비정상 수치가 있습니다."
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
raw = 1500
print(raw * 1.0 - 1024.0)  # 476.0
```

### 이해 확인 문제

1. HU 계산 순서는?

해설: 저장 값 × slope + intercept다.

2. astype는 영상 크기를 바꿀까?

해설: 아니다. 숫자의 자료형을 바꾼다.

## 6장 첫 단면을 그림으로 보기

학습 목표: 숫자 배열을 같은 표시 범위의 그림으로 만든다.

plt.subplots는 전체 그림 fig와 그 안에 그릴 축 ax를 만든다. imshow는 배열 숫자를 색으로 표현한다. cmap="gray"는 흑백이고 vmin=-160, vmax=240은 화면 표시 범위다. 범위 밖 값은 어둡거나 밝은 끝 색으로 표시되지만 hu 배열을 직접 잘라 바꾸지 않는다.

colorbar는 숫자와 색의 관계를 표시한다. set_title과 axis("off")는 설명과 축 표시를 조절한다. tight_layout은 글과 그림의 겹침을 줄이는 배치 기능이다. savefig는 이미지 파일을 저장하고 close는 그림 객체를 닫아 누적을 줄인다. display(Image(...))는 저장한 PNG를 노트북에 보여준다. PNG는 발표용 그림이며 정량 분석용 HU 배열을 대신하지 않는다.

### 핵심 용어

| 이름      | 뜻                           |
| --------- | ---------------------------- |
| fig       | 그림 전체                    |
| ax        | 하나의 그리기 영역           |
| vmin vmax | 색상 매핑의 최소·최대 표시값 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 6 · 노트북 셀 위치 13

fig,ax 두 결과를 받고 흑백 HU 화면을 그린다. 저장 파일을 먼저 만든 다음 Image로 표시한다. 그림을 close해도 저장된 PNG는 남는다.

```python
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(hu, cmap="gray", vmin=-160, vmax=240)
ax.set_title(f"{PATIENT} — first full-dose slice")
ax.axis("off")
fig.colorbar(im, ax=ax, label="HU", shrink=0.7)
fig.tight_layout()
first_image = OUTPUT / "first_ct.png"
fig.savefig(first_image, dpi=120)
plt.close(fig)
display(Image(filename=str(first_image)))
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(hu.min(), hu.max())
# 표시 범위를 바꿔도 위 숫자가 바뀌지 않는지 확인한다.
```

### 이해 확인 문제

1. imshow의 vmin 설정은 데이터 clipping일까?

해설: 아니다. 표시 방법을 정한다.

2. PNG를 학습 정답 대신 써도 같은가?

해설: 표시·저장 과정에서 정보가 달라질 수 있으므로 현재 코드는 HU 배열을 사용한다.

## 7장 재사용할 HU 읽기 함수

학습 목표: 반복할 파일 읽기 코드를 함수로 묶는다.

read_hu(path)는 경로 하나를 받아 DICOM 읽기, float32 픽셀 변환, HU 계산을 수행하고 배열을 반환한다. 정의 셀을 실행하면 함수가 준비되며, 이후 호출할 때마다 전달한 경로를 읽는다. 함수 안의 ds와 pixels는 지역 변수다. 밖에서 ds라는 이름을 사용하고 있더라도 호출 안의 지역 ds와는 별개다.

return은 배열을 돌려준다. full_hu = read_hu(path)에서 호출 결과가 full_hu에 연결된다. 이 함수는 경로에 따라 정상선량이나 저선량을 읽을 뿐, 모델이나 정답을 판단하지 않는다. 두 영상에 같은 함수를 적용해야 처리 방식이 일관된다. 이 함수는 필수 태그가 없으면 오류가 난다. 조용히 임의의 값으로 채우기보다 이 프로젝트에서는 데이터 검사에서 오류를 확인한다.

### 핵심 용어

| 이름      | 뜻                         |
| --------- | -------------------------- |
| path      | 함수에 전달한 파일 경로    |
| return    | 호출한 쪽에 결과 전달      |
| 지역 변수 | 함수 내부 실행에 속한 이름 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 7 · 노트북 셀 위치 15

함수 정의 셀이다. return의 배열을 호출 쪽에서 받는다. path를 바꾸면 같은 규칙으로 다른 영상을 읽는다.

```python
def read_hu(path):
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float32)
    # 저장된 픽셀 값 → Hounsfield Unit. PNG 변환 전에 이 값을 유지한다.
    return pixels * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
sample_hu = read_hu(files[0])
print(sample_hu.shape, sample_hu.dtype)
```

### 이해 확인 문제

1. def 실행만으로 모든 파일을 읽을까?

해설: 아니다. 함수 호출에서 파일을 읽는다.

2. print와 return의 차이는?

해설: print는 표시, return은 함수의 결과 전달이다.

## 8장 환자의 DICOM 헤더 모으기

학습 목표: 헤더를 모으고 실패한 파일을 기록한다.

scan_patient는 특정 사례의 촬영 정보를 모은다. stop_before_pixels=True는 픽셀 데이터 전까지 읽어 메타데이터 검사에 필요한 비용을 줄인다. 폴더 안의 모든 경로를 순회하고, 파일인지와 사례 폴더 이름이 경로 구성에 들어 있는지 확인한다. continue는 조건에 맞지 않는 항목의 처리를 건너뛰고 다음 항목으로 간다.

각 정상 파일은 path, position, orientation, spacing, shape, kernel, thickness, series 등을 담은 딕셔너리로 변환된다. np.asarray(..., dtype=float)는 좌표를 계산 가능한 실수 배열로 만든다. getattr(ds,"NumberOfFrames",1)은 해당 속성이 없으면 1을 사용하는 접근이다. 단일 프레임 CT라는 조건을 확인하며, 여러 프레임 파일을 그대로 같은 방식으로 처리하지 않는다.

try 안에서 읽기·검사가 실패하면 except가 파일 경로와 오류 문자열을 errors에 담는다. 즉 오류를 지워 버리지 않고 나중에 볼 수 있게 기록한다. 유효한 단면이 없거나 여러 SeriesInstanceUID가 섞이면 함수 자체가 중단된다. 마지막 return slices, errors는 튜플 두 항목을 반환한다. full, errors_full = ...은 그 두 결과를 각각 받는다.

### 핵심 용어

| 이름              | 뜻                        |
| ----------------- | ------------------------- |
| slices            | 단면별 헤더 딕셔너리 목록 |
| errors            | 실패 경로·오류 목록       |
| SeriesInstanceUID | 하나의 영상 시리즈 식별자 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 8 · 노트북 셀 위치 17

scan_patient의 내부 순서는 경로 필터, 헤더 읽기, 형식 검사, 정보 저장, 오류 수집, 전체 시리즈 검사다. slices와 errors 두 목록을 반환한다.

```python
def scan_patient(folder, patient):
    """픽셀은 아직 읽지 않고, 특정 환자의 헤더만 읽는다."""
    slices, errors = [], []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or patient not in path.relative_to(folder).parts:
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if ds.Modality != "CT" or int(getattr(ds, "NumberOfFrames", 1)) != 1:
                raise ValueError("단일 프레임 CT가 아닙니다")
            # 공개 데이터의 PatientID는 Anonymous이다. 사례 구분에는 폴더 ID를 쓴다.
            if str(ds.PatientID) not in (patient, "Anonymous"):
                raise ValueError(f"PatientID 불일치: {ds.PatientID}")
            slices.append({
                "path": path,
                "position": np.asarray(ds.ImagePositionPatient, dtype=float),
                "orientation": np.asarray(ds.ImageOrientationPatient, dtype=float),
                "spacing": np.asarray(ds.PixelSpacing, dtype=float),
                "shape": (int(ds.Rows), int(ds.Columns)),
                "kernel": str(ds.ConvolutionKernel),
                "thickness": float(ds.SliceThickness),
                "series": str(ds.SeriesInstanceUID),
                "dicom_patient_id": str(ds.PatientID),
            })
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
    if not slices:
        raise ValueError(f"{folder}: {patient}의 유효한 CT가 없습니다. 오류: {errors[:3]}")
    if len({s["series"] for s in slices}) != 1:
        raise ValueError("시리즈가 여러 개입니다. 사용할 시리즈를 먼저 선택해야 합니다.")
    return slices, errors
```

#### 원본 코드 셀 9 · 노트북 셀 위치 18

함수를 정상·저선량 폴더에 각각 호출한다. 헤더 오류가 하나라도 있으면 raise로 중단한다. display(full[0])는 첫 헤더 사전을 표시한다.

```python
full, errors_full = scan_patient(ROOT / "data/raw/full", PATIENT)
quarter, errors_quarter = scan_patient(ROOT / "data/raw/quarter", PATIENT)
print("정상 / 저선량 파일 수:", len(full), len(quarter))
print("읽기 오류:", errors_full + errors_quarter)
if errors_full or errors_quarter:
    raise ValueError("읽기 오류를 해결한 뒤 진행하세요.")
print("첫 정상선량 헤더:")
display(full[0])
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(full[0].keys())
print(full[0]["position"])
print(errors_full)
```

### 이해 확인 문제

1. 왜 헤더만 먼저 읽을까?

해설: 픽셀 전체를 읽기 전에 촬영 조건과 파일 구성을 검사하기 위해서다.

2. except가 있다고 오류가 해결된 걸까?

해설: 아니다. 기록했을 뿐이며 호출 쪽에서 오류가 있으면 진행을 중단한다.

## 9장 같은 위치와 재구성 조건인지 비교하기

학습 목표: 정상·저선량의 위치와 재구성 조건을 비교한다.

same_geometry(a,b)는 두 헤더가 짝이 될 수 있는지를 True 또는 False로 반환한다. shape와 kernel은 같은 값을 요구한다. 두께·좌표·방향·간격은 소수점 표현 차이를 고려해 isclose 또는 allclose로 검사한다. isclose는 수치 하나 또는 원소별 비교, allclose는 배열 전체가 허용 범위에서 가까운지 확인할 때 쓴다.

atol은 절대 허용 오차다. rtol=0은 값의 크기에 비례한 추가 허용을 쓰지 않는다는 뜻이다. 위치·두께 0.001, 방향·간격 0.00001의 허용값은 현재 구현의 조건이며 모든 데이터에 자동으로 적절하다는 뜻은 아니다. and로 연결한 조건 중 하나라도 실패하면 False다. PatientID의 같음은 같은 폴더 사례 안에서 추가 확인이며 Anonymous만으로 사례를 구분하지 않는다.

### 핵심 용어

| 이름     | 뜻                                         |
| -------- | ------------------------------------------ |
| allclose | 배열 전체가 허용 오차 안에서 가까운지 확인 |
| atol     | 절대 허용 오차                             |
| rtol     | 값의 크기에 비례한 허용 오차               |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 10 · 노트북 셀 위치 20

조건들을 and로 결합해 True/False를 돌려준다. np.isclose는 두께, np.allclose는 벡터 정보에 적용한다. rtol=0으로 절대 허용 오차만 쓴다.

```python
def same_geometry(a, b):
    return (
        a["dicom_patient_id"] == b["dicom_patient_id"]
        and a["shape"] == b["shape"]
        and a["kernel"] == b["kernel"]
        and np.isclose(a["thickness"], b["thickness"], atol=0.001, rtol=0)
        and np.allclose(a["position"], b["position"], atol=0.001, rtol=0)
        and np.allclose(a["orientation"], b["orientation"], atol=0.00001, rtol=0)
        and np.allclose(a["spacing"], b["spacing"], atol=0.00001, rtol=0)
    )
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
a = np.array([1.0, 2.0])
b = np.array([1.0001, 2.0])
print(np.allclose(a, b, atol=0.001, rtol=0))
```

### 이해 확인 문제

1. 배열 좌표를 ==만으로 비교하지 않는 이유는?

해설: 소수점 표현 차이를 허용하면서 모든 좌표를 검사하기 위해서다.

2. 조건 하나가 False면 반환값은?

해설: False다.

## 10장 영상 짝 목록 만들기

학습 목표: 영상 쌍을 유일하게 만들고 물리적 순서로 정렬한다.

pair_slices는 저선량 단면 q마다 same_geometry를 만족하는 정상선량 후보 f를 찾는다. matches가 정확히 하나여야 하고, 그 정상선량 파일을 다른 저선량과 이미 사용했으면 안 된다. used라는 집합은 중복 사용을 방지한다. 마지막에는 모든 정상선량도 사용했는지 확인해 누락을 검사한다.

같은 두 영상 정보를 (full_info, quarter_info) 튜플로 묶어 pairs 리스트에 저장한다. 그 뒤 방향 벡터 두 개의 외적 np.cross로 단면의 법선 방향을 구한다. 위치를 이 법선에 내적 np.dot으로 투영하면 단면을 진행 방향에 따라 정렬할 수 있다. 단순히 z 좌표만 정렬하는 것보다 방향을 반영한다. 법선의 부호에 따라 정렬 방향은 달라질 수 있지만 두 선량의 짝은 함께 움직인다.

index = len(pairs)//2는 가운데 단면을 고르는 정수 나눗셈이다. 가운데 단면이 특정 장기나 병변을 대표한다는 보장은 없다. f,q = pairs[index]는 한 쌍을 풀어서 받는다. 여기서 불러오는 full_hu와 quarter_hu는 실제 HU 배열이며, 평균 절대 차이는 아직 모델 적용 전 두 영상의 차이다.

### 핵심 용어

| 이름     | 뜻                               |
| -------- | -------------------------------- |
| used     | 이미 사용한 정상선량 경로의 집합 |
| np.cross | 두 방향에서 법선 계산            |
| np.dot   | 위치를 법선 방향에 투영          |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 11 · 노트북 셀 위치 22

저선량 하나당 matches를 찾고 일대일 관계를 확인한다. used로 정상선량 중복 사용을 막는다. 마지막 cross·dot·sorted는 물리적 방향의 정렬이다.

```python
def pair_slices(full, quarter):
    pairs, used = [], set()
    for q in quarter:
        matches = [f for f in full if same_geometry(f, q)]
        if len(matches) != 1 or matches[0]["path"] in used:
            raise ValueError(f"대응하는 정상선량 영상이 유일하지 않습니다: {q['path']}")
        used.add(matches[0]["path"])
        pairs.append((matches[0], q))
    if len(used) != len(full):
        raise ValueError("짝이 없는 정상선량 슬라이스가 있습니다.")
    # 촬영 방향의 법선에 투영한 위치로 정렬한다.
    orientation = pairs[0][0]["orientation"]
    normal = np.cross(orientation[:3], orientation[3:])
    return sorted(pairs, key=lambda pair: float(np.dot(pair[0]["position"], normal)))
```

#### 원본 코드 셀 12 · 노트북 셀 위치 23

pairs의 길이를 검사하고 가운데 한 쌍을 골라 실제 픽셀을 읽는다. 평균 절대 HU 차이는 두 선량의 차이며 CNN 평가는 아니다.

```python
pairs = pair_slices(full, quarter)
print("짝 개수:", len(pairs))
assert len(pairs) == len(full) == len(quarter)
index = len(pairs) // 2  # 나중에 30 등으로 바꿔 보세요.
#index = 30
f, q = pairs[index]
full_hu = read_hu(f["path"])
quarter_hu = read_hu(q["path"])
assert np.isfinite(full_hu).all() and np.isfinite(quarter_hu).all()
print("정상선량 경로:", f["path"])
print("저선량 경로:", q["path"])
print("촬영 위치:", f["position"])
print("두 배열 크기:", full_hu.shape, quarter_hu.shape)
print("평균 절대 차이 (HU):", float(np.abs(quarter_hu - full_hu).mean()))
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(len(pairs))
print(pairs[0][0]["position"])
print(pairs[-1][0]["position"])
```

### 이해 확인 문제

1. 왜 matches 길이가 하나여야 할까?

해설: 정답이 유일한 일대일 대응이어야 하기 때문이다.

2. //2는 무엇을 선택할까?

해설: 리스트 가운데의 정수 인덱스다.

## 11장 저선량·정상선량·차이 영상 비교

학습 목표: 같은 표시 조건으로 두 영상과 차이 영상을 읽는다.

plt.subplots(1,3)는 한 줄에 세 그림 영역을 만든다. zip은 축·영상·제목을 같은 순서로 묶어 반복한다. 처음 두 그림은 같은 HU [-160,240] 범위다. 따로 자동 밝기를 적용하면 실제 차이가 가려질 수 있으므로 비교에서는 같은 표시 조건을 유지한다.

세 번째 그림은 quarter_hu - full_hu의 부호 있는 차이다. 양수는 저선량 값이 더 큰 픽셀, 음수는 더 작은 픽셀이다. coolwarm 색상과 [-100,100] 범위를 쓴다. 뒤 25장의 절대 오차 지도는 np.abs를 사용하므로 음수와 양수를 합쳐 차이 크기만 표시한다. 두 그림의 색과 역할을 혼동하지 말자. 정상선량도 노이즈를 포함하므로 이 차이를 순수 노이즈라고 단정하지 않는다.

### 핵심 용어

| 이름      | 뜻                                |
| --------- | --------------------------------- |
| zip       | 여러 목록을 같은 순서로 묶어 반복 |
| 차이 영상 | 두 배열을 픽셀별로 뺀 결과        |
| 절대 오차 | 부호를 버린 차이의 크기           |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 13 · 노트북 셀 위치 25

세 영역 중 앞 두 개는 선량별 영상, 셋째는 부호 있는 차이다. zip의 세 목록이 같은 순서로 묶인다. title과 suptitle은 축 제목·전체 제목이다.

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, image, title in zip(axes[:2], [quarter_hu, full_hu],
                            ["Quarter dose", "Full dose reference"]):
    ax.imshow(image, cmap="gray", vmin=-160, vmax=240)
    ax.set_title(title + " | HU [-160, 240]")
    ax.axis("off")
diff = axes[2].imshow(quarter_hu - full_hu, cmap="coolwarm", vmin=-100, vmax=100)
axes[2].set_title("Quarter - Full (HU)")
axes[2].axis("off")
fig.colorbar(diff, ax=axes[2], shrink=0.7)
fig.suptitle(f"{PATIENT} | pair {index}/{len(pairs)} | position {f['position'].tolist()}")
fig.tight_layout()
preview_path = OUTPUT / "preview.png"
fig.savefig(preview_path, dpi=140)
plt.close(fig)
display(Image(filename=str(preview_path)))
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
a = np.array([120.0, 80.0])
b = np.array([100.0, 100.0])
print(a - b)
print(np.abs(a - b))
```

### 이해 확인 문제

1. 차이가 -20 HU면 무슨 뜻일까?

해설: 저선량의 그 픽셀 값이 정상선량보다 20 HU 낮다.

2. 이 그림이 CNN 출력일까?

해설: 아직 모델이 등장하기 전의 데이터 비교다.

## 12장 CSV 짝 목록과 검사 결과 저장

학습 목표: 경로 목록과 검사 결과를 디스크에 저장한다.

CSV는 같은 열을 가진 행의 목록을 저장하기 적합하다. writerow는 한 행, writerows는 여러 행을 쓴다. with ... open("w")는 쓰기 모드이며 같은 파일이 있으면 덮어쓴다. newline=""는 CSV 줄바꿈 처리, utf-8-sig는 UTF-8 BOM을 포함해 일부 Windows 프로그램에서 한글 인식을 돕는 설정이다.

JSON은 중첩된 설정·결과를 저장하기 좋다. json.dumps는 Python 딕셔너리를 문자열로 바꾸고 write_text는 그 문자열을 파일에 쓴다. ensure_ascii=False는 한글을 읽기 쉬운 형태로 유지하고 indent=2는 들여쓰기한다. report의 note는 이 단계가 전체 픽셀 검사가 아니라 헤더 짝짓기와 선택한 한 쌍의 픽셀 검사라는 범위를 기록한다.

### 핵심 용어

| 이름       | 뜻                             |
| ---------- | ------------------------------ |
| CSV        | 행과 열로 저장한 목록          |
| JSON       | 설정·중첩 결과를 저장한 텍스트 |
| write_text | 문자열을 파일에 기록           |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 14 · 노트북 셀 위치 27

CSV 한 행에 사례·두 경로·좌표 세 값을 쓴다. \*position은 세 좌표를 펼친다. JSON report는 이 검사의 범위와 오류 목록을 함께 저장한다.

```python
pairs_path = OUTPUT / "pairs.csv"
with pairs_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["patient_id", "full_path", "quarter_path", "x_mm", "y_mm", "z_mm"])
    for full_info, quarter_info in pairs:
        writer.writerow([PATIENT, str(full_info["path"]), str(quarter_info["path"]),
                         *full_info["position"]])
report = {
    "patient": PATIENT, "full_count": len(full), "quarter_count": len(quarter),
    "paired_count": len(pairs), "shape": list(full_hu.shape), "kernel": f["kernel"],
    "thickness_mm": f["thickness"], "preview_pair_index": index,
    "dicom_patient_id": f["dicom_patient_id"], "errors": errors_full + errors_quarter,
    "note": "전체 픽셀 검사가 아니라 헤더 짝짓기 및 선택한 한 쌍의 픽셀 검사입니다.",
}
(OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
display(report)
print("결과 폴더:", OUTPUT)
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(pairs_path.exists())
print((OUTPUT / "report.json").read_text(encoding="utf-8")[:200])
```

### 이해 확인 문제

1. 경로 CSV에 픽셀이 들어 있을까?

해설: 아니다. 원본 파일을 찾아갈 경로와 위치 정보다.

2. 초기 report가 모든 단면의 픽셀 정상임을 증명할까?

해설: 아니다. 전체 픽셀 검사는 14장에서 수행한다.

## 13장 저장한 CSV에서 다시 읽어오기

학습 목표: 저장한 경로로 같은 배열을 다시 읽는지 검사한다.

csv.DictReader는 첫 행의 열 이름을 딕셔너리 키로 사용한다. 읽은 값은 기본적으로 문자열이므로 z_mm 같은 숫자는 계산 전에 float 변환이 필요하다. list(...)는 반복해서 읽을 수 있도록 모든 행을 리스트에 담는다. saved_pairs[index]로 전에 선택한 행을 다시 찾고 두 경로를 read_hu에 전달한다.

np.array_equal은 배열 크기와 모든 값이 정확히 같은지 검사한다. 같은 파일을 같은 방식으로 다시 읽는 이 검사는 저장·다시 읽기 과정의 연결을 확인한다. 모델 출력의 부동소수점 비교에서는 보통 정확한 같음보다 허용 오차가 있는 allclose를 사용한다. CSV에는 절대 경로가 저장되어 있어 다른 컴퓨터로 옮기면 목록을 다시 생성해야 할 수 있다.

### 핵심 용어

| 이름        | 뜻                               |
| ----------- | -------------------------------- |
| DictReader  | CSV 각 행을 딕셔너리로 읽기      |
| array_equal | 배열 모양과 값의 정확한 같음     |
| 절대 경로   | 컴퓨터의 루트부터 정한 파일 위치 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 15 · 노트북 셀 위치 29

DictReader로 저장 목록을 읽고 같은 index의 경로를 다시 사용한다. array_equal로 같은 파일을 같은 처리로 읽었는지 확인한다.

```python
with pairs_path.open(encoding="utf-8-sig", newline="") as file:
    saved_pairs = list(csv.DictReader(file))
pair = saved_pairs[index]
full_from_csv = read_hu(pair["full_path"])
quarter_from_csv = read_hu(pair["quarter_path"])
assert np.array_equal(full_from_csv, full_hu)
assert np.array_equal(quarter_from_csv, quarter_hu)
print("CSV에서 다시 읽은 배열이 이전 배열과 같습니다.")
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(saved_pairs[0].keys())
print(type(saved_pairs[0]["z_mm"]))
```

### 이해 확인 문제

1. CSV의 z_mm 자료형은?

해설: 기본적으로 문자열이다.

2. 다른 컴퓨터에서 같은 경로가 없으면?

해설: 원본 데이터의 새 위치를 기준으로 짝 목록을 다시 생성한다.

## 14장 전체 환자의 데이터 검사

학습 목표: 10명 전체의 헤더와 모든 픽셀을 검사한다.

한 환자에서 코드가 동작했다고 모든 데이터가 정상인 것은 아니다. full_root.iterdir()는 바로 아래 폴더들을 순회하고 집합으로 사례 이름을 모은다. 양쪽 집합의 차이는 한쪽에만 있는 사례다. 두 집합이 같아야 모든 환자에게 두 선량이 있는 구성으로 진행할 수 있다.

환자별로 scan_patient와 pair_slices를 호출한 뒤, 모든 짝의 두 영상을 실제 read_hu로 읽는다. 크기가 헤더 정보와 같은지, 유한한 수치인지 확인한다. 한 환자의 전체 검사가 끝나야 그 환자의 목록을 all_pair_rows에 추가하므로 검사 중 실패한 사례가 일부만 성공 목록에 섞이는 것을 줄인다. 실패 사례는 failed_patients와 summary에 기록한다.

HTML 표를 만드는 부분은 검사 결과를 보기 좋게 표시하는 코드다. escape는 오류 문자열 등에 들어 있는 <, > 같은 글자를 HTML 태그로 해석하지 않도록 바꾼다. 이 표시는 품질 검사 자체와 별개다. 검사 보고서는 먼저 저장하고, failed_patients가 있으면 전체 학습용 manifest 저장을 중단한다. 실제 결과는 환자 10명, 2,378쌍이다.

### 핵심 용어

| 이름     | 뜻                                |
| -------- | --------------------------------- |
| manifest | 원본 파일의 위치·짝을 명시한 목록 |
| summary  | 사례별 검사 결과                  |
| escape   | 문자열을 HTML에서 안전하게 표시   |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 16 · 노트북 셀 위치 32

full/quarter의 바로 아래 사례 폴더를 집합으로 모은다. 집합 차이를 출력하고 양쪽 전체 목록이 같은지 assert한다.

```python
full_root = ROOT / "data" / "raw" / "full" / "full_3mm"
quarter_root = ROOT / "data" / "raw" / "quarter" / "quarter_3mm"

full_patients = {
    folder.name
    for folder in full_root.iterdir()
    if folder.is_dir()
}

quarter_patients = {
    folder.name
    for folder in quarter_root.iterdir()
    if folder.is_dir()
}

print("정상선량에만 있는 환자:", full_patients - quarter_patients)
print("저선량에만 있는 환자:", quarter_patients - full_patients)

assert full_patients == quarter_patients, "양쪽 환자 목록이 다릅니다."

patients = sorted(full_patients)

print("환자 수:", len(patients))
print("환자 목록:", patients)
```

#### 원본 코드 셀 17 · 노트북 셀 위치 33

환자 반복 안에서 모든 픽셀을 읽는다. 성공한 환자만 all_pair_rows에 등록하고 예외는 FAILED 보고서로 남긴다. 이 셀은 학습 전 QC다.

```python
summary = []       # 환자별 검사 결과
all_pair_rows = [] # 전체 영상 짝 목록
failed_patients = []

for patient in patients:
    print(f"{patient} 검사 시작")

    try:
        full_slices, full_errors = scan_patient(
            ROOT / "data" / "raw" / "full",
            patient,
        )

        quarter_slices, quarter_errors = scan_patient(
            ROOT / "data" / "raw" / "quarter",
            patient,
        )

        header_errors = full_errors + quarter_errors

        if header_errors:
            raise ValueError(f"헤더 읽기 오류: {header_errors[:3]}")

        patient_pairs = pair_slices(full_slices, quarter_slices)

        # 헤더 검사에 더해 모든 영상의 픽셀도 실제로 읽습니다.
        for full_info, quarter_info in patient_pairs:
            for info in (full_info, quarter_info):
                image = read_hu(info["path"])

                if image.shape != info["shape"]:
                    raise ValueError(f"영상 크기 불일치: {info['path']}")

                if not np.isfinite(image).all():
                    raise ValueError(f"NaN 또는 무한대 발견: {info['path']}")

        # 환자 전체 검사가 성공한 경우에만 목록에 추가합니다.
        for full_info, quarter_info in patient_pairs:
            all_pair_rows.append({
                "patient_id": patient,
                "full_path": str(full_info["path"]),
                "quarter_path": str(quarter_info["path"]),
                "x_mm": float(full_info["position"][0]),
                "y_mm": float(full_info["position"][1]),
                "z_mm": float(full_info["position"][2]),
            })

        summary.append({
            "patient_id": patient,
            "full_count": len(full_slices),
            "quarter_count": len(quarter_slices),
            "paired_count": len(patient_pairs),
            "status": "OK",
            "error": "",
        })

        print(f"  완료: {len(patient_pairs)}쌍")

    except Exception as error:
        failed_patients.append(patient)

        summary.append({
            "patient_id": patient,
            "full_count": None,
            "quarter_count": None,
            "paired_count": None,
            "status": "FAILED",
            "error": str(error),
        })

        print(f"  실패: {error}")
```

#### 원본 코드 셀 18 · 노트북 셀 위치 34

열 이름으로 HTML 표를 만들며 escape로 문자열을 표시한다. 표시 결과가 검사 자체를 대신하지 않는다.

```python
from IPython.display import HTML
from html import escape

columns = [
    "patient_id",
    "full_count",
    "quarter_count",
    "paired_count",
    "status",
    "error",
]

header = "".join(f"<th>{column}</th>" for column in columns)

rows = ""
for item in summary:
    cells = "".join(
        f"<td>{escape(str(item[column]))}</td>"
        for column in columns
    )
    rows += f"<tr>{cells}</tr>"

display(HTML(
    f"<table><thead><tr>{header}</tr></thead>"
    f"<tbody>{rows}</tbody></table>"
))

print("성공한 환자:", len(patients) - len(failed_patients))
print("실패한 환자:", failed_patients)
print("전체 영상 짝 수:", len(all_pair_rows))
```

#### 원본 코드 셀 19 · 노트북 셀 위치 35

보고서를 먼저 CSV로 쓰고 실패 환자가 없을 때만 all_pairs.csv를 생성한다. DictWriter의 fieldnames는 열 이름·순서를 정한다.

```python
audit_folder = ROOT / "outputs" / "data_audit"
audit_folder.mkdir(parents=True, exist_ok=True)

summary_path = audit_folder / "patient_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(summary)

print("검사 결과 저장:", summary_path)

# 실패가 있으면 전체 학습 목록 생성은 중단합니다.
assert not failed_patients, (
    f"실패한 환자를 먼저 확인하세요: {failed_patients}"
)

manifest_path = audit_folder / "all_pairs.csv"

with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "patient_id",
            "full_path",
            "quarter_path",
            "x_mm",
            "y_mm",
            "z_mm",
        ],
    )
    writer.writeheader()
    writer.writerows(all_pair_rows)

print("전체 짝 목록 저장:", manifest_path)
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(len(all_pair_rows))
print([item["patient_id"] for item in summary if item["status"] != "OK"])
```

### 이해 확인 문제

1. 헤더 검사만으로 압축 픽셀 손상을 확인할 수 있을까?

해설: 충분하지 않다. 여기서는 모든 영상의 픽셀도 읽는다.

2. 실패 사례가 있으면 다음 학습으로 넘어갈까?

해설: 원본 코드는 목록 저장 전 assert로 중단한다.

## 15장 환자 단위 학습·검증·테스트 분리

학습 목표: 환자를 겹치지 않게 나누고 목록을 저장한다.

splits는 split 이름을 키로, 환자 ID 목록을 값으로 가진 딕셔너리다. 집합의 &는 교집합, |는 합집합, -는 차집합이다. train과 val, train과 test, val과 test의 교집합이 모두 비어 있는지 확인한다. 데이터의 모든 사례가 정확히 분할에 배정되었는지도 비교한다.

같은 환자의 인접 단면은 비슷하므로 단면을 섞어 나누면 평가 환자의 정보가 학습에 들어갈 수 있다. 우리는 사례 단위로 7/1/2명을 배정했다. train은 파라미터 변경, val은 모델 선택, test는 선택 후 평가에 쓴다. test를 반복해서 보고 모델 선택에 반영하면 독립적인 최종 평가의 의미가 약해진다.

patient_to_split은 환자 ID에서 split 이름을 바로 찾는 역방향 사전이다. row.copy()는 기존 행을 복사해 split 열을 추가하며 원본 pair_rows의 행에 직접 추가하지 않는다. 단순한 값들로 구성된 현재 행에서는 얕은 복사가 충분하다. 저장된 train/val/test.csv에는 원본 영상 자체가 아니라 경로가 들어 있다. 1,709/214/455라는 수는 환자 수가 아니라 영상 쌍 수다.

### 핵심 용어

| 이름        | 뜻                                               |
| ----------- | ------------------------------------------------ |
| 교집합      | 두 집합에 함께 있는 항목                         |
| row.copy    | 원본 딕셔너리의 얕은 복사                        |
| 데이터 누출 | 평가에 독립적이어야 할 정보가 개발 과정에 들어감 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 20 · 노트북 셀 위치 37

이미 생성한 QC 파일을 다시 읽는다. len==10과 status==OK를 확인한 뒤 전체 짝 목록을 pair_rows로 준비한다.

```python
audit_folder = ROOT / "outputs" / "data_audit"

with (audit_folder / "patient_summary.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    patient_summary = list(csv.DictReader(file))

assert len(patient_summary) == 10, "검사한 환자 수를 확인하세요."
assert all(row["status"] == "OK" for row in patient_summary), (
    "검사에 실패한 환자가 있습니다."
)

with (audit_folder / "all_pairs.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    pair_rows = list(csv.DictReader(file))

print("전체 영상 짝 수:", len(pair_rows))
print("환자 목록:", sorted({row["patient_id"] for row in pair_rows}))
```

#### 원본 코드 셀 21 · 노트북 셀 위치 38

splits는 역할별 사례 ID를 수동으로 정한 사전이다. 데이터 수치 배열이나 모델을 담는 변수가 아니다.

```python
splits = {
    "train": ["L067", "L096", "L109", "L143", "L192", "L286", "L291"],
    "val": ["L310"],
    "test": ["L333", "L506"],
}
```

#### 원본 코드 셀 22 · 노트북 셀 위치 39

세 교집합이 비었는지와 모든 데이터 사례가 배정됐는지 확인한다. 오류 메시지의 집합 차이는 누락·불일치 ID다.

```python
train_ids = set(splits["train"])
val_ids = set(splits["val"])
test_ids = set(splits["test"])

# 교집합이 비어 있어야 합니다.
assert not (train_ids & val_ids), "train과 val에 같은 환자가 있습니다."
assert not (train_ids & test_ids), "train과 test에 같은 환자가 있습니다."
assert not (val_ids & test_ids), "val과 test에 같은 환자가 있습니다."

data_ids = {row["patient_id"] for row in pair_rows}
assigned_ids = train_ids | val_ids | test_ids

assert data_ids == assigned_ids, (
    f"분할에서 빠진 환자: {data_ids - assigned_ids}, "
    f"데이터에 없는 환자: {assigned_ids - data_ids}"
)

print("환자 중복 없음 / 모든 환자 배정 완료")
```

#### 원본 코드 셀 23 · 노트북 셀 위치 40

중첩 컴프리헨션으로 환자→split 사전을 만들고 원본 행을 copy한 뒤 split 열을 더한다. split_rows는 새 목록이다.

```python
patient_to_split = {
    patient: split_name
    for split_name, patient_ids in splits.items()
    for patient in patient_ids
}

split_rows = []

for row in pair_rows:
    new_row = row.copy()
    new_row["split"] = patient_to_split[row["patient_id"]]
    split_rows.append(new_row)

print(split_rows[0])
```

#### 원본 코드 셀 24 · 노트북 셀 위치 41

split 이름별로 행을 골라 사례 수와 영상 수를 따로 센다. 전체 행 수가 보존됐는지 마지막에 검사한다.

```python
for split_name in splits:
    selected_rows = [
        row for row in split_rows
        if row["split"] == split_name
    ]

    selected_patients = {
        row["patient_id"] for row in selected_rows
    }

    print(
        f"{split_name}: "
        f"환자 {len(selected_patients)}명 / "
        f"영상 {len(selected_rows)}쌍"
    )

assert len(split_rows) == len(pair_rows)
```

#### 원본 코드 셀 25 · 노트북 셀 위치 42

split 설정 JSON, 전체 목록 CSV, 역할별 CSV를 저장한다. 이 셀을 실행해야 뒤의 train.csv 파일이 존재한다.

```python
split_folder = ROOT / "data" / "splits"
split_folder.mkdir(parents=True, exist_ok=True)

# 환자 분할 설정
(split_folder / "splits.json").write_text(
    json.dumps(splits, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# 전체 짝 목록 + split 열
fieldnames = list(pair_rows[0].keys()) + ["split"]

with (split_folder / "pairs_with_split.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(split_rows)

# 학습·검증·테스트 목록을 별도 파일로도 저장
for split_name in splits:
    selected_rows = [
        row for row in split_rows
        if row["split"] == split_name
    ]

    with (split_folder / f"{split_name}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

print("저장 위치:", split_folder)
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(set(splits["train"]) & set(splits["test"]))
print(len(split_rows))
```

### 이해 확인 문제

1. 테스트 455장은 환자 455명일까?

해설: 아니다. 두 환자의 455개 단면이다.

2. val의 역할은?

해설: 학습 중 성능을 확인하고 체크포인트를 선택하는 것이다.

## 16장 HU 정규화와 학습용 패치 만들기

학습 목표: HU 정규화와 동일 좌표 패치를 이해한다.

normalize_hu는 HU를 [-1000,2000]으로 clip한 뒤 3000으로 나누어 [0,1]에 대응시킨다. -1000은 0, 0은 약 0.3333, 500은 0.5, 2000은 1이다. clip은 범위 밖 값을 끝값으로 바꾸므로 데이터 처리이며, imshow의 vmin/vmax와 다르다. clip으로 사라진 범위 밖 정보는 역변환으로 되찾을 수 없다.

crop_pair는 같은 크기의 두 배열에서 같은 top·left를 사용해 잘라낸다. rng.integers의 끝값은 포함하지 않으므로 height - patch_size + 1을 써서 마지막 가능한 시작 위치도 선택한다. 512에서 128을 자를 때 시작 행은 0부터 384까지다. 두 영상의 위치를 따로 뽑으면 입력·정답이 다른 해부학적 영역이 된다.

return에 두 패치와 두 시작 좌표를 함께 넣고 호출 쪽에서 네 값을 받는다. 난수 생성기 seed 42는 같은 초기 상태에서 같은 난수열을 시작하도록 한다. 같은 rng 객체를 다시 호출하면 상태가 진행되어 다음 좌표가 나온다. 패치는 순수 배경을 포함할 수 있으므로 배치 그림을 확인한다. Rectangle은 화면의 x가 열 left, y가 행 top이라는 순서에 주의한다.

### 핵심 용어

| 이름 | 뜻                             |
| ---- | ------------------------------ |
| clip | 범위 밖의 값을 제한            |
| rng  | 상태를 가진 난수 생성기        |
| 패치 | 전체 영상에서 선택한 작은 영역 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 26 · 노트북 셀 위치 44

train.csv를 읽고 가운데 행의 두 경로를 read_hu로 처리한다. 클래스 정의보다 먼저 기본 배열을 확인하는 실습이다.

```python
train_csv = ROOT / "data" / "splits" / "train.csv"

with train_csv.open(encoding="utf-8-sig", newline="") as file:
    train_rows = list(csv.DictReader(file))

assert train_rows, "학습 목록이 비어 있습니다."
assert all(row["split"] == "train" for row in train_rows)

# 학습 목록 가운데에 있는 한 쌍 선택
row = train_rows[len(train_rows) // 2]

quarter_hu = read_hu(row["quarter_path"])
full_hu = read_hu(row["full_path"])

assert quarter_hu.shape == full_hu.shape

print("선택한 환자:", row["patient_id"])
print("영상 위치:", row["z_mm"])
print("저선량 크기:", quarter_hu.shape)
print("정상선량 크기:", full_hu.shape)
```

#### 원본 코드 셀 27 · 노트북 셀 위치 45

HU 상수와 normalize_hu 함수를 정의한다. clip, 빼기, 나누기, float32 변환 순서다.

```python
HU_MIN = -1000.0
HU_MAX = 2000.0

def normalize_hu(image):
    clipped = np.clip(image, HU_MIN, HU_MAX)

    normalized = (clipped - HU_MIN) / (HU_MAX - HU_MIN)

    return normalized.astype(np.float32)
```

#### 원본 코드 셀 28 · 노트북 셀 위치 46

두 배열을 동일 함수로 정규화하고 수치·자료형·유효성을 확인한다. 이름과 배열을 튜플 목록으로 묶어 반복한다.

```python
quarter_normalized = normalize_hu(quarter_hu)
full_normalized = normalize_hu(full_hu)

for name, image in [
    ("저선량", quarter_normalized),
    ("정상선량", full_normalized),
]:
    print(
        name,
        "크기:", image.shape,
        "자료형:", image.dtype,
        "범위:", float(image.min()), float(image.max()),
    )

    assert np.isfinite(image).all()
    assert 0 <= image.min() <= image.max() <= 1
```

#### 원본 코드 셀 29 · 노트북 셀 위치 47

crop_pair는 위치를 한 번 뽑고 두 배열에 똑같은 슬라이싱을 한다. 네 값을 반환한다. RNG 끝값이 제외되어 +1이 필요하다.

```python
def crop_pair(quarter, full, patch_size, rng):
    assert quarter.shape == full.shape

    height, width = quarter.shape

    if patch_size > min(height, width):
        raise ValueError("패치가 원본 영상보다 큽니다.")

    top = int(rng.integers(0, height - patch_size + 1))
    left = int(rng.integers(0, width - patch_size + 1))

    quarter_patch = quarter[
        top:top + patch_size,
        left:left + patch_size,
    ]

    full_patch = full[
        top:top + patch_size,
        left:left + patch_size,
    ]

    return quarter_patch, full_patch, top, left
```

#### 원본 코드 셀 30 · 노트북 셀 위치 48

seed 42로 난수 생성기를 만들고 패치를 선택한다. 함수 정의와 실제 호출이 서로 다른 셀이다.

```python
rng = np.random.default_rng(42)

quarter_patch, full_patch, top, left = crop_pair(
    quarter_normalized,
    full_normalized,
    patch_size=128,
    rng=rng,
)

print("자른 위치 — 행:", top, "/ 열:", left)
print("입력 패치:", quarter_patch.shape)
print("기준 패치:", full_patch.shape)

assert quarter_patch.shape == (128, 128)
assert full_patch.shape == (128, 128)
```

#### 원본 코드 셀 31 · 노트북 셀 위치 49

정규화 영상 표시 범위를 계산하고 전체 영상에 선택 영역 사각형을 그린다. Rectangle의 좌표는 열·행 순서다.

```python
from matplotlib.patches import Rectangle

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 기존 CT 표시 범위를 정규화된 숫자로 변환
display_min = (-160 - HU_MIN) / (HU_MAX - HU_MIN)
display_max = (240 - HU_MIN) / (HU_MAX - HU_MIN)

axes[0].imshow(
    quarter_normalized,
    cmap="gray",
    vmin=display_min,
    vmax=display_max,
)

axes[0].add_patch(
    Rectangle(
        (left, top),
        128,
        128,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
)
axes[0].set_title("Selected patch location")

axes[1].imshow(
    quarter_patch,
    cmap="gray",
    vmin=display_min,
    vmax=display_max,
)
axes[1].set_title("Input: quarter dose")

axes[2].imshow(
    full_patch,
    cmap="gray",
    vmin=display_min,
    vmax=display_max,
)
axes[2].set_title("Reference: full dose")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(normalize_hu(np.array([-1000, 0, 500, 2000], dtype=np.float32)))
print(quarter_patch.shape)
```

### 이해 확인 문제

1. 정규화는 shape를 바꿀까?

해설: 아니다. 숫자의 범위를 바꾼다.

2. 512에서 128을 자를 때 마지막 시작 위치는?

해설: 384이며 384:512를 선택한다.

## 17장 PyTorch Dataset과 DataLoader

학습 목표: NumPy 배열을 Tensor와 배치로 묶는다.

PyTorch Tensor는 숫자 배열이며 GPU 연산과 자동미분에 연결할 수 있다. torch.from_numpy는 NumPy 배열의 메모리를 공유할 수 있어, 코드에서는 .copy()로 배열을 복사한 뒤 Tensor를 만든다. unsqueeze(0)은 크기 1의 채널 축을 추가해 (128,128)을 (1,128,128)로 바꾼다. 픽셀 값 자체를 늘려 만들지 않는다.

CTPairDataset은 PyTorch Dataset을 상속한다. **init**은 목록과 설정을 저장하고 **len**은 단면 수를 반환하며 **getitem**은 요청받은 인덱스의 파일을 읽고 변환한다. training=True일 때만 패치를 자르고 False이면 전체 단면을 반환한다. 이 클래스는 model.train()과 무관하다. 이름이 비슷해도 데이터 처리 조건과 모델 모드는 다른 설정이다.

DataLoader가 네 샘플을 모으면 (4,1,128,128)이 된다. 차원은 N,C,H,W 순서다. next(iter(train_loader))는 새 반복자를 만들어 첫 배치를 받아 확인한다. shuffle은 샘플 순서를 섞고 num_workers=0은 현재 프로세스에서 읽는 설정이다. train_dataset[0]을 여러 번 호출하면 학습용 랜덤 패치는 달라질 수 있다. 이것은 원본 단면이 달라진 것이 아니다.

### 핵심 용어

| 이름               | 뜻                           |
| ------------------ | ---------------------------- |
| Tensor             | PyTorch 숫자 배열            |
| N C H W            | 배치 수·채널 수·높이·너비    |
| Dataset DataLoader | 샘플 만드는 규칙과 샘플 묶기 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 32 · 노트북 셀 위치 52

torch와 Dataset·DataLoader 클래스를 import한다. 버전 출력만으로 모델이 생성되거나 학습되지는 않는다.

```python
import torch
from torch.utils.data import Dataset, DataLoader

print("PyTorch 버전:", torch.__version__)
```

#### 원본 코드 셀 33 · 노트북 셀 위치 53

train.csv를 다시 읽어 train_rows를 준비한다. NameError라면 이 셀 또는 이전의 목록 읽기 셀을 실행해야 한다.

```python
with (ROOT/"data/splits/train.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    train_rows = list(csv.DictReader(file))

print("학습 영상 쌍:", len(train_rows))
```

#### 원본 코드 셀 34 · 노트북 셀 위치 54

CTPairDataset은 상태 저장, 길이, 인덱스 요청 처리 세 메서드를 구현한다. training=False면 패치 없이 전체 단면을 반환한다.

```python
class CTPairDataset(Dataset):
    def __init__(self, rows, training=True, patch_size=128, seed=42):
        self.rows = rows
        self.training = training
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        # 1. DICOM → HU 배열
        quarter = read_hu(row["quarter_path"])
        full = read_hu(row["full_path"])

        # 2. 두 영상에 같은 정규화 적용
        quarter = normalize_hu(quarter)
        full = normalize_hu(full)

        # 3. 학습할 때만 같은 위치에서 패치 생성
        if self.training:
            quarter, full, _, _ = crop_pair(
                quarter,
                full,
                self.patch_size,
                self.rng,
            )

        # 4. NumPy → PyTorch 텐서, 채널 차원 추가
        x = torch.from_numpy(quarter.copy()).unsqueeze(0)
        y = torch.from_numpy(full.copy()).unsqueeze(0)

        return x, y
```

#### 원본 코드 셀 35 · 노트북 셀 위치 55

dataset[0]은 **getitem**(0) 실행이다. 랜덤 패치이므로 같은 호출을 반복해도 내용이 달라질 수 있다. .item()은 숫자 하나를 Python 값으로 꺼낸다.

```python
train_dataset = CTPairDataset(train_rows, training=True)

x, y = train_dataset[0]

print("데이터 개수:", len(train_dataset))
print("입력:", x.shape, x.dtype)
print("기준:", y.shape, y.dtype)
print("입력 범위:", x.min().item(), x.max().item())

assert x.shape == y.shape == (1, 128, 128)
assert torch.isfinite(x).all() and torch.isfinite(y).all()
```

#### 원본 코드 셀 36 · 노트북 셀 위치 56

DataLoader를 만들고 첫 배치를 받는다. 샘플 (1,128,128) 네 개가 (4,1,128,128)이 된다.

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

images, targets = next(iter(train_loader))

print("입력 배치:", images.shape)
print("기준 배치:", targets.shape)

assert images.shape == targets.shape == (4, 1, 128, 128)
```

#### 원본 코드 셀 37 · 노트북 셀 위치 57

images[0,0]은 첫 배치 항목·첫 채널의 2차원 영상이다. 현재 CPU Tensor여서 numpy 변환이 가능하다.

```python
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for ax, image, title in zip(
    axes,
    [images[0, 0].numpy(), targets[0, 0].numpy()],
    ["Input patch", "Reference patch"],
):
    ax.imshow(
        image,
        cmap="gray",
        vmin=(-160 - HU_MIN) / (HU_MAX - HU_MIN),
        vmax=(240 - HU_MIN) / (HU_MAX - HU_MIN),
    )
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(images.shape)
print(images[0].shape)
print(images[0, 0].shape)
print(images.device)
```

### 이해 확인 문제

1. unsqueeze(0)을 두 번 적용하면?

해설: (H,W)에서 (1,H,W), 다시 (1,1,H,W)가 된다.

2. DataLoader는 모델을 학습할까?

해설: 아니다. 데이터를 공급하며 파라미터 변경은 학습 루프에서 한다.

## 18장 저선량 영상의 비교 기준 평가

학습 목표: 모델 적용 전 기준점과 세 평가 지표를 만든다.

모델이 좋아졌는지 알려면 저선량을 그대로 사용한 성능이 필요하다. val_dataset은 training=False로 전체 단면을 반환한다. x[0]은 채널 축을 선택해 2차원 배열로 만들고 .numpy()로 NumPy 연산에 전달한다. 현재 검증 데이터는 CPU Tensor여서 가능하며 GPU Tensor는 먼저 .cpu()가 필요하다.

MAE는 |예측-기준|의 평균이다. [0.4,0.6]과 [0.5,0.5]의 MAE는 0.1이다. MSE는 차이를 제곱한 평균으로 0.01이고, data_range=1일 때 PSNR은 10log10(1/MSE)=20dB다. 오차가 0이면 PSNR은 무한대다. PSNR의 평균은 단면별로 계산한 값을 평균한 것으로 전체 MSE를 합쳐 PSNR을 한 번 계산한 값과 일반적으로 다르다.

SSIM은 국소 영역의 밝기·대비·구조 관계를 사용한다. 현재 structural_similarity의 기본 설정과 data_range=1을 사용한다. 반환된 SSIM은 진단 정확도가 아니다. 전신 배경까지 포함한 전체 단면 지표이고 환자 한 명의 여러 단면이다. float64는 평가 계산의 숫자 정밀도를 높이려는 선택으로 학습 Tensor까지 float64로 바꾸는 것은 아니다.

baseline_results는 각 단면의 결과 목록, baseline_average는 각 지표의 평균 사전이다. \*\*metrics는 지표 사전의 항목들을 행 사전에 펼쳐 넣는다. CSV는 단면별 결과, JSON은 조건과 평균을 저장한다. 실제 검증 기준은 MAE 0.0053097575, PSNR 42.7088358, SSIM 0.9537145다.

### 핵심 용어

| 이름 | 뜻                                   |
| ---- | ------------------------------------ |
| MAE  | 절대 오차 평균                       |
| PSNR | 정해진 범위와 MSE로 계산하는 dB 지표 |
| SSIM | 국소 밝기·대비·구조 유사성 지표      |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 38 · 노트북 셀 위치 59

scikit-image의 structural_similarity 함수만 import한다. 설치가 없으면 이 줄에서 ModuleNotFoundError가 난다.

```python
from skimage.metrics import structural_similarity
```

#### 원본 코드 셀 39 · 노트북 셀 위치 60

두 배열을 float64로 처리하고 MAE·MSE·PSNR·SSIM을 계산한다. MSE==0에서는 log 계산 대신 inf를 반환한다.

```python
def calculate_metrics(prediction, reference):
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    assert prediction.shape == reference.shape
    assert np.isfinite(prediction).all()
    assert np.isfinite(reference).all()

    difference = prediction - reference

    mae = np.mean(np.abs(difference))
    mse = np.mean(difference ** 2)

    # 정규화 범위가 0~1이므로 data_range는 1
    psnr = (
        float("inf")
        if mse == 0
        else 10 * np.log10(1.0 / mse)
    )

    ssim = structural_similarity(
        prediction,
        reference,
        data_range=1.0,
    )

    return {
        "mae": float(mae),
        "psnr": float(psnr),
        "ssim": float(ssim),
    }
```

#### 원본 코드 셀 40 · 노트북 셀 위치 61

val.csv를 읽고 CTPairDataset(training=False)로 전체 단면을 준비한다. 검증 입력은 (1,512,512)다.

```python
with (ROOT/"data/splits/val.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    val_rows = list(csv.DictReader(file))

assert val_rows
assert all(row["split"] == "val" for row in val_rows)

val_dataset = CTPairDataset(
    val_rows,
    training=False,
)

x, y = val_dataset[0]

print("검증 입력 크기:", x.shape)
print("검증 기준 크기:", y.shape)
```

#### 원본 코드 셀 41 · 노트북 셀 위치 62

모든 검증 단면에서 저선량을 그대로 평가한다. \*\*metrics로 결과 사전의 항목을 펼쳐 넣고 단면별 지표의 평균을 만든다.

```python
baseline_results = []

for index in range(len(val_dataset)):
    x, y = val_dataset[index]

    quarter_image = x[0].numpy()
    full_image = y[0].numpy()

    metrics = calculate_metrics(quarter_image, full_image)

    baseline_results.append({
        "patient_id": val_rows[index]["patient_id"],
        "slice_index": index,
        **metrics,
    })

    if (index + 1) % 50 == 0:
        print(f"{index + 1}/{len(val_dataset)}쌍 평가 완료")

baseline_average = {
    metric: float(np.mean([
        result[metric] for result in baseline_results
    ]))
    for metric in ["mae", "psnr", "ssim"]
}

print("평가한 단면 수:", len(baseline_results))
print("저선량 영상의 평균 성능:", baseline_average)
```

#### 원본 코드 셀 42 · 노트북 셀 위치 63

baseline의 단면 CSV와 조건·평균 JSON을 저장한다. 이후 CNN 평가의 비교 기준이다.

```python
baseline_folder = ROOT / "outputs" / "baseline"
baseline_folder.mkdir(parents=True, exist_ok=True)

with (baseline_folder / "val_metrics.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(baseline_results[0].keys()),
    )
    writer.writeheader()
    writer.writerows(baseline_results)

report = {
    "method": "저선량 영상 그대로",
    "split": "val",
    "patients": sorted({row["patient_id"] for row in val_rows}),
    "slice_count": len(baseline_results),
    "hu_clip": [HU_MIN, HU_MAX],
    "data_range": 1.0,
    "evaluation_region": "전체 단면, 배경 포함",
    "aggregation": "단면별 지표의 산술평균",
    "average": baseline_average,
}

(baseline_folder / "val_summary.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("저장 위치:", baseline_folder)
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
a = np.array([0.4, 0.6])
b = np.array([0.5, 0.5])
print(np.abs(a-b).mean())
print(10*np.log10(1/np.square(a-b).mean()))
```

### 이해 확인 문제

1. SSIM에 data_range를 정하는 이유는?

해설: 정규화된 숫자의 기준 범위를 명확히 하기 위해서다.

2. 지표가 높으면 진단 성능도 입증됐을까?

해설: 현재 실험은 영상 복원 지표이며 병변 판독 성능은 평가하지 않았다.

## 19장 작은 잔차 CNN 만들기

학습 목표: 합성곱과 잔차 모델의 계산을 이해한다.

CNN의 합성곱 층은 작은 필터를 이동시키며 주변 숫자에 가중치를 곱해 합하고 편향을 더한다. 필터의 값들은 학습으로 바뀐다. Conv2d(1,32,3,padding=1)은 흑백 1채널에서 특징 32채널을 만든다. 출력 채널 각각은 입력 채널들의 필터 계산을 결합한다. 특징 채널은 환자나 장기 32개를 의미하지 않는다.

stride 기본값 1, dilation 기본값 1, kernel 3, padding 1이면 높이·너비가 유지된다. ReLU는 max(0,z)로 음수 값을 0으로 만드는 활성화 함수이며 여러 층이 복잡한 관계를 학습하게 한다. nn.Sequential은 작성한 순서대로 층을 연결한다. 마지막 층은 ReLU 없이 1채널 보정값을 출력하므로 양수와 음수 보정이 가능하다.

forward는 x + self.net(x)를 반환한다. 입력을 그대로 통과시키는 경로와 보정값 경로가 함께 있다. 마지막 가중치·편향을 0으로 만들면 초기 보정값이 0이고 출력이 입력과 같다. 나머지 층까지 모두 0으로 초기화하지 않는다. 첫 업데이트에서는 마지막 층의 0 가중치 때문에 앞 층으로 가는 기울기가 0일 수 있고, 마지막 층이 바뀐 뒤 앞 층도 학습된다.

파라미터 수는 첫 층 32×(1×3×3)+32=320, 둘째 층 32×(32×3×3)+32=9248, 마지막 층 1×(32×3×3)+1=289, 총 9857이다. parameter.numel()은 텐서에 들어 있는 숫자 개수다. 세 합성곱의 수용 영역은 stride 1 기준 7×7이며 큰 해부학적 맥락을 충분히 활용하는지 한계가 있다.

저장된 19장 클래스 이름은 ResidualDenoiser이고, 20장에서 같은 구조를 SmallResidualCNN으로 정의한다. 서로 다른 논문 모델을 비교한 것이 아니다. 최종 학습·체크포인트·ONNX에서는 SmallResidualCNN을 사용한다. 발표에서 이 이름 혼동을 정리해야 한다.

### 숫자 9개로 합성곱 이해하기

합성곱의 작은 숫자 예제다. 3×3 숫자와 필터의 같은 위치를 곱한 뒤 모두 더한다. 모든 필터 값을 1/9로 두면 이 영역의 평균인 5가 나온다. 실제 CNN에서는 이 필터 값들이 학습으로 바뀌고, 필터를 이동시켜 여러 위치의 출력을 만든다. 입력 채널이 여러 개면 채널별 곱의 합도 함께 계산한다. 이 예제는 구조를 이해하는 계산이며 실제 모델의 필터를 평균 필터로 바꾸는 코드가 아니다.

```python
image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
weight = np.ones((3, 3), dtype=np.float32) / 9
print((image * weight).sum())  # 5.0
```

### 핵심 용어

| 이름      | 뜻                                  |
| --------- | ----------------------------------- |
| 합성곱    | 필터 가중치로 주변 픽셀·특징을 계산 |
| 파라미터  | 학습으로 바꾸는 가중치·편향         |
| 잔차 연결 | 입력에 예측한 보정값을 더하는 경로  |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 43 · 노트북 셀 위치 65

초기 모델 이름은 ResidualDenoiser다. 같은 계산 구조를 뒤에서 SmallResidualCNN으로 다시 정의한다. model.parameters는 가중치와 편향을 제공한다.

```python
import torch
from torch import nn

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("학습 장치:", device)


class ResidualDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        # 시작할 때는 입력을 그대로 출력하도록 설정
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        correction = self.net(x)
        return x + correction


model = ResidualDenoiser().to(device)

parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)
print(model)
print("학습할 매개변수 수:", parameter_count)
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(sum(p.numel() for p in model.parameters()))
print(model)
```

### 이해 확인 문제

1. 32채널은 CT 32장인가?

해설: 아니다. 영상에서 계산한 특징 채널이다.

2. 두 모델 이름은 다른 구조인가?

해설: 현재 원본 코드에서는 같은 구조이며 최종 명칭은 SmallResidualCNN이다.

## 20장 고정된 패치 4장으로 학습 검사

학습 목표: Loss 역전파 최적화를 작은 데이터에서 확인한다.

먼저 학습 목록 가운데의 네 단면에서 중앙 128×128을 고정해 만든다. np.stack은 네 배열을 묶어 (4,128,128)을 만들고 unsqueeze(1)은 채널 축을 추가한다. 랜덤 패치를 계속 바꾸면 동일 데이터에서 오차가 줄어드는지 검사하기 어려워 이 실습에서는 고정한다. 정상선량도 같은 좌표를 사용한다.

L1Loss는 예측과 기준의 절대 차이를 평균한 하나의 수다. 두 픽셀 오차가 0.1과 0.3이면 손실은 0.2다. 학습 중 Loss Tensor는 자동미분 계산 그래프와 연결되지만 .item()으로 꺼낸 Python 숫자는 연결되지 않는다. loss.item()은 기록에 쓰고 loss.backward()는 원래 Tensor에 호출한다.

기울기는 파라미터를 조금 바꿀 때 손실이 어떻게 변하는지 나타낸다. 간단한 예측 y_hat=w×x에서 x=2, 정답 y=6, w=1이면 L1 손실은 |2-6|=4다. 현재 지점에서 w를 증가시키면 손실이 줄어드는 방향이므로 기울기는 -2다. 단순 경사하강이라면 w_new=w-lr×기울기로 w를 늘린다. 실제 Adam은 기울기의 이동 평균 등을 이용하므로 이 식 그대로만 업데이트하는 것은 아니다.

zero_grad(set_to_none=True)는 이전 기울기를 지워 다음 배치의 기울기가 누적되지 않도록 한다. backward는 새 기울기를 계산하지만 파라미터를 직접 바꾸지 않는다. optimizer.step이 파라미터를 바꾼다. predictions는 모델의 복원 영상이며 loss는 복원 영상 자체가 아니다. model.train은 학습 모드 전환이고 모델 파일을 불러오거나 자동으로 학습하는 명령이 아니다.

eval은 Dropout·BatchNorm 같은 층의 동작 모드를 바꾸는 기능이며, no_grad는 자동미분 기록을 생략하는 기능이다. 현재 모델에는 Dropout·BatchNorm이 없어 train/eval이 출력 계산을 직접 바꾸지는 않지만 올바른 루프 구조로 구분한다. 고정 패치에서 오차가 줄어든 것은 학습 연결 검사다. 새 환자 성능은 이후에 별도로 평가한다. 이 셀만 다시 실행하면 기존 모델을 이어 학습하며 모델 생성 셀부터 실행하면 새 모델로 시작한다.

### 파라미터 하나로 자동미분 확인하기

앞의 기울기 설명을 PyTorch로 직접 확인한다. requires_grad=True는 w에 대한 기울기를 계산하도록 한다. w.grad가 -2로 계산되어도 w 값은 여전히 1이다. 따라서 backward는 학습 방향의 계산이고 파라미터 업데이트와는 다르다는 것을 실제 출력으로 볼 수 있다. 이 w는 책의 작은 예제용 숫자이며 CNN model의 파라미터를 변경하지 않는다.

```python
import torch
w = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(2.0)
y = torch.tensor(6.0)
prediction = w * x
loss = torch.abs(prediction - y)
loss.backward()
print("예측:", prediction.item())  # 2.0
print("손실:", loss.item())        # 4.0
print("w의 기울기:", w.grad.item())  # -2.0
print("현재 w:", w.item())         # 1.0; backward만으로 바뀌지 않음
```

### 핵심 용어

| 이름     | 뜻                                         |
| -------- | ------------------------------------------ |
| Loss     | 모델의 예측과 기준 사이 차이를 나타내는 수 |
| backward | 계산 그래프를 통해 기울기 계산             |
| step     | 최적화 알고리즘으로 파라미터 업데이트      |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 44 · 노트북 셀 위치 67

네 단면의 중앙 패치를 고정한다. np.stack 뒤 unsqueeze(1)으로 채널 축을 넣고 두 Tensor를 같은 장치에 올린다.

```python
# 기존에 만든 학습 목록 사용
middle = len(train_rows) // 2
debug_rows = train_rows[middle - 2:middle + 2]

assert len(debug_rows) == 4

input_patches = []
target_patches = []

for row in debug_rows:
    quarter = normalize_hu(read_hu(row["quarter_path"]))
    full = normalize_hu(read_hu(row["full_path"]))

    assert quarter.shape == full.shape

    height, width = quarter.shape
    top = (height - 128) // 2
    left = (width - 128) // 2

    # 입력과 정답에서 같은 위치 자르기
    input_patch = quarter[top:top + 128, left:left + 128]
    target_patch = full[top:top + 128, left:left + 128]

    input_patches.append(input_patch)
    target_patches.append(target_patch)

# (4, 128, 128) → (4, 1, 128, 128)
fixed_x = torch.from_numpy(
    np.stack(input_patches)
).unsqueeze(1).to(device)

fixed_y = torch.from_numpy(
    np.stack(target_patches)
).unsqueeze(1).to(device)

print("환자:", [row["patient_id"] for row in debug_rows])
print("입력 크기:", fixed_x.shape)
print("정답 크기:", fixed_y.shape)

assert torch.isfinite(fixed_x).all()
assert torch.isfinite(fixed_y).all()
```

#### 원본 코드 셀 45 · 노트북 셀 위치 68

SmallResidualCNN을 정의하고 새 객체를 만든다. fixed_x.device를 따라 모델을 올리고 출력이 초기 입력과 같은지 확인한다.

```python
import torch
from torch import nn

# 앞에서 준비한 패치가 있는지 확인
if "fixed_x" not in globals() or "fixed_y" not in globals():
    raise RuntimeError(
        "패치 준비 셀을 먼저 실행하세요. "
        "fixed_x와 fixed_y가 필요합니다."
    )

# 패치가 올라가 있는 장치 사용
device = fixed_x.device
assert fixed_y.device == device


# 모델 구조 정의
class SmallResidualCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

        # 처음에는 출력이 입력과 같도록 설정
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        correction = self.net(x)
        return x + correction


# 정의한 구조로 새로운 모델 생성
torch.manual_seed(42)
model = SmallResidualCNN().to(device)

criterion = nn.L1Loss()

# 초기 출력 확인
model.eval()

with torch.no_grad():
    initial_prediction = model(fixed_x)

    initial_loss = criterion(
        initial_prediction, fixed_y
    ).item()

    identity_error = (
        initial_prediction - fixed_x
    ).abs().max().item()

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("학습 장치:", device)
print("파라미터 수:", parameter_count)
print("입력 크기:", fixed_x.shape)
print("출력 크기:", initial_prediction.shape)
print("초기 MAE:", initial_loss)
print("초기 출력과 입력의 최대 차이:", identity_error)

assert initial_prediction.shape == fixed_y.shape
assert identity_error < 1e-6

print("모델 생성과 초기 출력 확인 완료")
```

#### 원본 코드 셀 46 · 노트북 셀 위치 69

고정 배치를 100번 반복하며 backward와 step으로 학습한다. loss.item은 기록용이며 final_loss는 마지막 업데이트 후에 다시 계산한 값이다.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

loss_history = []

model.train()

for step in range(1, 101):
    # 이전 단계의 기울기 지우기
    optimizer.zero_grad(set_to_none=True)

    # 복원 영상 예측
    predictions = model(fixed_x)

    # 예측과 정상선량 영상의 차이 계산
    loss = criterion(predictions, fixed_y)

    if not torch.isfinite(loss).item():
        raise RuntimeError("학습 오차가 NaN 또는 Inf입니다.")

    # 기울기 계산 → 모델 파라미터 변경
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if step == 1 or step % 20 == 0:
        print(f"step {step:3d} | MAE {loss.item():.6f}")


# 학습이 끝난 모델의 오차 확인
model.eval()

with torch.no_grad():
    final_prediction = model(fixed_x)
    final_loss = criterion(
        final_prediction, fixed_y
    ).item()

print()
print(f"학습 전 MAE: {initial_loss:.6f}")
print(f"학습 후 MAE: {final_loss:.6f}")

if final_loss < initial_loss:
    print("고정 패치에서 학습 오차가 감소했습니다.")
else:
    print("오차가 줄지 않았습니다. 학습 출력을 확인해야 합니다.")
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
x = 2.0
y = 6.0
w = 1.0
print(abs(w*x-y))
print(abs(1.1*x-y))  # w를 늘리면 손실 감소
```

### 이해 확인 문제

1. zero_grad는 입력 영상을 0으로 만들까?

해설: 아니다. 파라미터의 이전 기울기를 지운다.

2. backward만 하면 모델이 바뀔까?

해설: 기울기만 계산한다. 파라미터 업데이트는 step이 한다.

## 21장 전체 학습 데이터 준비

학습 목표: 최종 학습용 Dataset과 반복 가능한 설정을 준비한다.

load_split_rows는 split 이름을 받아 해당 CSV를 읽는다. required_names와 globals 검사는 앞 셀에서 필요한 함수가 준비되었는지 확인한다. 이름이 없으면 전체 학습을 시작하기 전에 알려준다. 이것은 패키지 설치 검사와 다르다.

TrainingPatchDataset은 17장의 CTPairDataset 중 학습 패치 역할에 집중한 클래스다. 두 클래스를 한꺼번에 실행한다고 데이터를 두 배로 학습하는 것은 아니다. train_dataset과 train_loader를 이 셀에서 다시 연결하면 이후 루프는 새 객체를 사용한다. 이 중복은 학습 중 보완된 코드의 흔적이며 구조 차이를 알면 혼동을 줄일 수 있다.

DataLoader의 generator seed는 샘플 순서 난수에, Dataset의 rng seed는 패치 위치 난수에 사용된다. torch.manual_seed는 모델 초기값 등 PyTorch 난수와 관련된다. 각 난수 흐름이 별개다. seed를 적었다고 하드웨어·라이브러리가 달라도 완전히 같은 결과를 보장한 것은 아니다. len(train_loader)는 1709를 배치 4로 묶은 428이며 마지막 배치는 1개다. 현재 drop_last를 사용하지 않는다.

### 핵심 용어

| 이름          | 뜻                             |
| ------------- | ------------------------------ |
| generator     | DataLoader가 사용할 난수 상태  |
| 마지막 배치   | 나누어 떨어지지 않은 남은 샘플 |
| num_workers=0 | 현재 프로세스에서 데이터 읽기  |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 47 · 노트북 셀 위치 71

최종 학습용 목록·Dataset·DataLoader를 다시 준비한다. 같은 이름 train_dataset/train_loader가 새 객체에 연결된다. seed가 위치·순서에 각각 적용된다.

```python
import csv
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 앞에서 실행해야 하는 함수와 모델 확인
required_names = [
    "ROOT", "read_hu", "normalize_hu",
    "crop_pair", "SmallResidualCNN",
]

missing = [
    name for name in required_names
    if name not in globals()
]

if missing:
    raise RuntimeError(f"앞의 정의 셀을 먼저 실행하세요: {missing}")


# 저장했던 환자별 분할 목록 읽기
def load_split_rows(split_name):
    path = ROOT / "data" / "splits" / f"{split_name}.csv"

    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows, f"{split_name} 목록이 비어 있습니다."
    assert all(row["split"] == split_name for row in rows)
    return rows


train_rows = load_split_rows("train")
val_rows = load_split_rows("val")

train_patients = {row["patient_id"] for row in train_rows}
val_patients = {row["patient_id"] for row in val_rows}

assert not (train_patients & val_patients)


# 학습용: 단면을 읽고 무작위 패치를 반환
class TrainingPatchDataset(Dataset):
    def __init__(self, rows, patch_size=128, seed=42):
        self.rows = rows
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        quarter = normalize_hu(read_hu(row["quarter_path"]))
        full = normalize_hu(read_hu(row["full_path"]))

        quarter_patch, full_patch, _, _ = crop_pair(
            quarter,
            full,
            self.patch_size,
            self.rng,
        )

        x = torch.from_numpy(quarter_patch.copy()).unsqueeze(0)
        y = torch.from_numpy(full_patch.copy()).unsqueeze(0)

        return x, y


train_dataset = TrainingPatchDataset(train_rows)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    generator=torch.Generator().manual_seed(42),
)

print("학습 환자:", sorted(train_patients))
print("검증 환자:", sorted(val_patients))
print("학습 단면:", len(train_dataset))
print("검증 단면:", len(val_rows))
print("1 epoch의 학습 배치 수:", len(train_loader))
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(len(train_loader))
print(len(train_dataset) % 4)
print(train_patients & val_patients)
```

### 이해 확인 문제

1. Dataset seed와 DataLoader seed는 같은 일을 할까?

해설: 각각 패치 위치와 샘플 순서를 제어한다.

2. 428번째 배치도 4개일까?

해설: 현재 목록에서는 마지막 배치가 1개다.

## 22장 전체 학습·검증 및 최적 모델 저장

학습 목표: 전체 학습과 검증을 분리하고 최적 모델을 저장한다.

이 장은 프로젝트의 중심이다. 고정 패치 실습의 모델 대신 새 SmallResidualCNN을 만든다. 모델과 x,y를 같은 device에 올린다. GPU 모델에 CPU 입력을 넣으면 장치 불일치 오류가 난다. EPOCHS=5는 전체 학습 목록을 다섯 번 사용한다는 뜻이며 step 다섯 번과 다르다. 한 epoch에는 428번의 배치 업데이트가 있다.

학습 루프는 배치마다 예측, 손실, backward, step 순서로 진행한다. total_train_loss += loss.item()\*batch_size는 마지막 작은 배치가 일반 배치와 같은 비중을 갖지 않도록 샘플 수를 반영한다. 마지막에 전체 sample_count로 나눈다. 모든 패치 크기가 같아 이 계산은 전체 패치 픽셀에 대한 평균과 대응한다. 각 배치의 손실은 업데이트 직전의 모델에서 계산하므로 epoch 끝 모델 하나의 정확한 전체 학습 성능과 같지는 않다.

검증에서는 model.eval과 no_grad를 쓰고, val_rows의 전체 512×512를 한 장씩 처리한다. 학습 패치와 검증 전체 단면의 MAE는 계산 영역과 배경 비중이 달라 두 수의 크기만 보고 과적합을 단정할 수 없다. 검증 평균의 추세와 학습 설정을 함께 본다. CNN은 전결합층 없이 합성곱으로 구성되어 PyTorch에서는 다른 높이·너비에도 적용할 수 있지만 나중 ONNX는 512로 고정해 변환한다.

best_val_mae를 무한대로 시작하면 첫 정상 검증값은 더 작으므로 저장한다. 이후 최저값보다 낮은 경우만 덮어쓴다. state_dict는 파라미터와 등록된 버퍼의 이름·Tensor를 담는다. detach().cpu().clone()은 자동미분에서 분리하고 CPU로 이동한 독립 복사본을 저장하기 위한 것이다. model 클래스 소스 자체를 복원하는 파일은 아니므로 다시 읽을 때 같은 구조 정의가 필요하다.

config는 설정, history는 epoch별 기록, best_model.pt는 선택된 모델이다. 현재 기록은 검증 MAE가 0.004611에서 0.003432로 내려갔고 5번째 epoch가 선택됐다. 이 다섯 값은 하이퍼파라미터 탐색 결과가 아니다. 학습률은 고정 0.001이고 별도 scheduler·회전 증강·mixed precision은 구현하지 않았다. 5 epoch가 충분한 최적 횟수인지도 별도 실험이 필요하다.

### 핵심 용어

| 이름       | 뜻                                 |
| ---------- | ---------------------------------- |
| epoch      | 전체 학습 목록 한 번 사용          |
| checkpoint | 특정 시점의 모델·조건 저장         |
| state_dict | 파라미터·버퍼를 이름으로 담은 사전 |

주의: 원본 학습 셀을 다시 실행하면 새 실험이 시작되고 같은 결과 파일을 덮어쓸 수 있다. 읽기 연습은 저장된 history를 읽는 작은 예제로 한다.

### 원본 코드를 따라 읽기

#### 원본 코드 셀 48 · 노트북 셀 위치 73

5 epoch 전체 학습과 전체 단면 검증을 수행한다. 검증이 최저일 때 독립 CPU 파라미터 사본과 설정을 저장한다. config/history/checkpoint의 역할을 구분하자.

```python
import time

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 고정 패치 검사 모델과 별도로 새 모델 생성
model = SmallResidualCNN().to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

run_folder = ROOT / "outputs" / "cnn_v1"
run_folder.mkdir(parents=True, exist_ok=True)

checkpoint_path = run_folder / "best_model.pt"

EPOCHS = 5
best_val_mae = float("inf")
history = []

config = {
    "model": "SmallResidualCNN",
    "seed": 42,
    "epochs": EPOCHS,
    "batch_size": 4,
    "patch_size": 128,
    "learning_rate": 0.001,
    "loss": "L1Loss",
    "hu_clip": [HU_MIN, HU_MAX],
    "train_patients": sorted(train_patients),
    "val_patients": sorted(val_patients),
    "evaluation_region": "전체 단면, 배경 포함",
    "aggregation": "단면별 MAE의 산술평균",
    "prediction_clipping": False,
}

(run_folder / "config.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("학습 장치:", device)
print("결과 폴더:", run_folder)

for epoch in range(1, EPOCHS + 1):
    started = time.perf_counter()

    # ---------- 학습 ----------
    model.train()

    total_train_loss = 0.0
    sample_count = 0

    for batch_index, (x, y) in enumerate(train_loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        prediction = model(x)
        loss = criterion(prediction, y)

        if not torch.isfinite(loss).item():
            raise RuntimeError("학습 오차가 NaN 또는 Inf입니다.")

        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_train_loss += loss.item() * batch_size
        sample_count += batch_size

        if batch_index % 100 == 0:
            print(
                f"epoch {epoch} | "
                f"batch {batch_index}/{len(train_loader)} | "
                f"MAE {loss.item():.6f}"
            )

    train_mae = total_train_loss / sample_count

    # ---------- 검증 ----------
    model.eval()
    val_maes = []

    with torch.no_grad():
        for row in val_rows:
            quarter = normalize_hu(read_hu(row["quarter_path"]))
            full = normalize_hu(read_hu(row["full_path"]))

            # 전체 단면: (H, W) → (1, 1, H, W)
            x = torch.from_numpy(quarter).unsqueeze(0).unsqueeze(0)
            prediction = model(x.to(device))

            prediction_np = prediction[0, 0].cpu().numpy()

            # 기존 기준 평가와 같이 float64로 지표 계산
            difference = (
                prediction_np.astype(np.float64)
                - full.astype(np.float64)
            )

            val_maes.append(float(np.abs(difference).mean()))

    val_mae = float(np.mean(val_maes))

    if not np.isfinite(val_mae):
        raise RuntimeError("검증 MAE가 NaN 또는 Inf입니다.")

    improved = val_mae < best_val_mae

    if improved:
        best_val_mae = val_mae

        torch.save(
            {
                "model_state_dict": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
                "epoch": epoch,
                "val_mae": val_mae,
                "config": config,
            },
            checkpoint_path,
        )

    record = {
        "epoch": epoch,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "seconds": time.perf_counter() - started,
        "saved_best": improved,
    }
    history.append(record)

    (run_folder / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[epoch {epoch}/{EPOCHS}] "
        f"train MAE={train_mae:.6f} | "
        f"val MAE={val_mae:.6f} | "
        f"best 저장={improved}"
    )

print("학습 완료")
print("최저 검증 MAE:", best_val_mae)
print("최적 모델:", checkpoint_path)
```

#### 원본 코드 셀 49 · 노트북 셀 위치 74

디스크 history를 읽어 목표 epoch까지 기록됐는지 확인한다. len만이 아니라 마지막 epoch 번호도 검사한다. 이 셀은 학습을 하지 않는다.

```python
import json

run_folder = ROOT / "outputs" / "cnn_v1"

history = json.loads(
    (run_folder / "history.json").read_text(encoding="utf-8")
)

config = json.loads(
    (run_folder / "config.json").read_text(encoding="utf-8")
)

print("목표 epoch:", config["epochs"])
print("저장된 epoch:", len(history))

for record in history:
    print(
        f'epoch {record["epoch"]} | '
        f'train MAE={record["train_mae"]:.6f} | '
        f'val MAE={record["val_mae"]:.6f}'
    )

assert history[-1]["epoch"] == config["epochs"], (
    "전체 학습이 아직 완료되지 않았습니다. "
    "학습 셀의 진행 상황이나 오류 출력을 확인하세요."
)

print("전체 학습 완료 확인")
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
import json
h = json.loads((ROOT / "outputs/cnn_v1/history.json").read_text(encoding="utf-8"))
for record in h:
    print(record["epoch"], record["val_mae"])
```

### 이해 확인 문제

1. 검증에서 optimizer.step을 할까?

해설: 하지 않는다. 검증은 파라미터 변경 없이 평가한다.

2. 마지막 모델과 최적 모델은 항상 같을까?

해설: 아니다. 이번에는 5번째 epoch가 최적이었지만 다른 실험에서는 이전 epoch일 수 있다.

## 23장 최적 모델의 검증 성능 평가

학습 목표: 선택된 모델을 다시 읽어 세 지표로 검증한다.

torch.load는 checkpoint 사전을 읽는다. map_location="cpu"는 저장 장치와 관계없이 CPU에서 읽도록 하고, weights_only=True는 제한된 방식으로 Tensor 중심 내용을 읽는 설정이다. 새 SmallResidualCNN을 만든 뒤 load_state_dict로 파라미터를 넣는다. 파일 이름만 지정한다고 모델 구조가 자동으로 생성되는 것은 아니다.

baseline의 HU 범위·data_range·환자 목록과 현재 조건을 assert로 비교한다. 같은 그림처럼 보여도 조건이 다르면 성능 비교는 달라질 수 있다. 예측의 [0,0]은 배치 한 장과 흑백 한 채널을 고르고 cpu().numpy로 평가용 배열을 만든다. no_grad 안의 출력은 자동미분 기록이 없으므로 현재는 별도 detach 없이 변환할 수 있다.

저장한 검증 지표는 MAE 0.0034324246, PSNR 46.4983663, SSIM 0.9825266다. 출력 추가 clipping을 하지 않으므로 모델 출력이 [0,1] 범위를 조금 벗어나도 그대로 평가한다. 예측이 정상선량에 가까워졌다는 지표이고 의료적 진단 효과를 입증하는 결과는 아니다. CSV 각 행에는 z_mm를 넣었으며 초기 baseline CSV에는 slice_index를 넣었다. 두 파일을 합칠 때 서로 같은 열이 있다고 가정하지 않는다.

### 핵심 용어

| 이름                | 뜻                                   |
| ------------------- | ------------------------------------ |
| load_state_dict     | 준비한 구조에 저장된 파라미터 넣기   |
| map_location        | 체크포인트를 읽을 장치 지정          |
| prediction_clipping | 출력값을 추가로 범위 제한하는지 여부 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 50 · 노트북 셀 위치 76

가장 좋은 모델을 읽고 세 지표로 검증한다. baseline 조건을 확인하고 val_metrics.csv와 val_summary.json에 기록한다.

```python
import csv
import numpy as np
import torch
from skimage.metrics import structural_similarity

# 이 프로젝트에서 직접 저장한 체크포인트 읽기
checkpoint = torch.load(
    run_folder / "best_model.pt",
    map_location="cpu",
    weights_only=True,
)

best_model = SmallResidualCNN().to(device)
best_model.load_state_dict(checkpoint["model_state_dict"])
best_model.eval()

print("선택된 epoch:", checkpoint["epoch"])
print("저장 당시 검증 MAE:", checkpoint["val_mae"])

# 기존 기준 평가와 조건이 같은지 확인
baseline = json.loads(
    (ROOT / "outputs" / "baseline" / "val_summary.json")
    .read_text(encoding="utf-8")
)

assert baseline["hu_clip"] == [HU_MIN, HU_MAX]
assert baseline["data_range"] == 1.0
assert sorted(baseline["patients"]) == sorted(val_patients)

metric_rows = []

with torch.no_grad():
    for index, row in enumerate(val_rows, start=1):
        quarter = normalize_hu(read_hu(row["quarter_path"]))
        full = normalize_hu(read_hu(row["full_path"]))

        x = (
            torch.from_numpy(quarter)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        prediction = best_model(x)[0, 0].cpu().numpy()

        # 전체 단면, 배경 포함 / 출력값 추가 clipping 없음
        prediction = prediction.astype(np.float64)
        reference = full.astype(np.float64)

        assert np.isfinite(prediction).all()

        difference = prediction - reference
        mae = float(np.abs(difference).mean())
        mse = float(np.square(difference).mean())

        psnr = (
            float(10 * np.log10(1.0 / mse))
            if mse > 0
            else float("inf")
        )

        ssim = float(
            structural_similarity(
                reference,
                prediction,
                data_range=1.0,
            )
        )

        metric_rows.append({
            "patient_id": row["patient_id"],
            "z_mm": row["z_mm"],
            "mae": mae,
            "psnr": psnr,
            "ssim": ssim,
        })

        if index % 50 == 0:
            print(f"검증 평가: {index}/{len(val_rows)}")


averages = {
    name: float(np.mean([row[name] for row in metric_rows]))
    for name in ["mae", "psnr", "ssim"]
}

# 단면별 결과 저장
with (run_folder / "val_metrics.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(metric_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(metric_rows)

# 평가 조건과 평균 결과 저장
summary = {
    "method": "SmallResidualCNN",
    "checkpoint_epoch": checkpoint["epoch"],
    "split": "val",
    "patients": sorted(val_patients),
    "slice_count": len(metric_rows),
    "hu_clip": [HU_MIN, HU_MAX],
    "data_range": 1.0,
    "evaluation_region": "전체 단면, 배경 포함",
    "aggregation": "단면별 지표의 산술평균",
    "prediction_clipping": False,
    "average": averages,
}

(run_folder / "val_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\n지표      저선량 그대로 → CNN")
for name in ["mae", "psnr", "ssim"]:
    before = baseline["average"][name]
    after = averages[name]
    print(f"{name.upper():5s}   {before:.6f} → {after:.6f}")

print("\n검증 결과 저장 완료")
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(checkpoint["epoch"])
print(checkpoint["config"]["hu_clip"])
print(best_model)
```

### 이해 확인 문제

1. torch.load만으로 클래스가 생길까?

해설: 아니다. 같은 구조를 먼저 정의해야 한다.

2. 평가에서 출력 clip을 적용했나?

해설: 현재 저장 조건은 False다.

## 24장 복원 영상 비교

학습 목표: 모델 결과를 HU 표시 범위로 돌려 비교한다.

predict_validation_slice는 인덱스 하나로 입력·기준을 읽고 모델 예측까지 반환한다. 반환값은 row, quarter, prediction, full 네 개다. quarter는 모델 입력, prediction은 출력, full은 비교 기준이다. 추론할 때 모델에 full을 넣지 않는다. 정상선량은 학습 정답 또는 평가 기준으로 사용한다.

정규화 값 n을 HU 체계로 표시할 때 n×3000-1000을 계산한다. 정규화 전에 clip했기 때문에 원본 HU 범위 밖의 정보까지 복원하는 역변환은 아니다. 세 그림에 같은 [-160,240] 범위를 사용한다. 모델이 부드럽게 보이는지뿐 아니라 작은 경계가 사라지거나 새로운 무늬가 생기지 않는지도 살펴본다. 이 눈으로 보는 확인은 전문가의 정식 판독 실험과 다르다.

### 핵심 용어

| 이름        | 뜻                                  |
| ----------- | ----------------------------------- |
| prediction  | 모델이 만든 복원 배열               |
| reference   | 비교에 사용하는 정상선량 배열       |
| 역표시 변환 | 정규화 수치를 제한된 HU 체계로 표현 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 51 · 노트북 셀 위치 78

한 검증 단면을 모델에 넣는 함수를 만들고 가운데 단면을 같은 HU 표시 범위로 비교한다. 정상선량은 모델 입력이 아니다.

```python
import matplotlib.pyplot as plt

# 시각화 결과 폴더
visual_folder = run_folder / "visuals"
visual_folder.mkdir(parents=True, exist_ok=True)


# 검증 목록의 특정 단면을 읽고 모델 적용
def predict_validation_slice(index):
    row = val_rows[index]

    quarter = normalize_hu(read_hu(row["quarter_path"]))
    full = normalize_hu(read_hu(row["full_path"]))

    x = (
        torch.from_numpy(quarter)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    best_model.eval()

    with torch.no_grad():
        prediction = best_model(x)[0, 0].cpu().numpy()

    return row, quarter, prediction, full


# 검증 환자의 가운데 단면 선택
preview_index = len(val_rows) // 2

row, quarter, prediction, full = predict_validation_slice(
    preview_index
)

# 학습용 정규화 값 → 표시용 HU 값
hu_span = HU_MAX - HU_MIN

images_hu = [
    quarter * hu_span + HU_MIN,
    prediction * hu_span + HU_MIN,
    full * hu_span + HU_MIN,
]

titles = [
    "Input: quarter dose",
    "CNN output",
    "Reference: full dose",
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, image, title in zip(axes, images_hu, titles):
    ax.imshow(
        image,
        cmap="gray",
        vmin=-160,
        vmax=240,
    )
    ax.set_title(title)
    ax.axis("off")

fig.suptitle(
    f'Patient {row["patient_id"]} | z={row["z_mm"]} mm'
)

plt.tight_layout()

save_path = visual_folder / "middle_slice_comparison.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()

print("저장 위치:", save_path)
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(quarter.shape, prediction.shape, full.shape)
print(prediction.min(), prediction.max())
```

### 이해 확인 문제

1. 추론 모델에 정상선량이 필요할까?

해설: 현재 모델의 입력은 저선량 한 장뿐이다.

2. 정규화 역변환은 모든 원본 값을 되찾을까?

해설: clip 이전의 범위 밖 정보는 되찾지 못한다.

## 25장 오차가 큰 검증 단면 분석

학습 목표: 오차가 큰 단면과 적용 전후의 차이를 구분한다.

saved_metrics의 mae는 CSV에서 읽은 문자열이다. float로 바꿔 내림차순 정렬해야 숫자 크기대로 정렬된다. [:3]은 첫 세 항목을 고른다. index_lookup은 (patient_id,z_mm) 튜플을 키로 원래 val_rows의 인덱스를 찾는다. 현재 데이터에서 키 중복이 없는지 검사한다. 여러 시리즈나 같은 위치가 반복되는 자료에서는 이 키만으로 충분하지 않을 수 있다.

가장 큰 after MAE 세 개를 고르는 것은 모델 적용 후 기준과 가장 다른 단면을 찾는 방식이다. 모델 적용으로 악화된 사례를 찾는 것과는 다르다. 각 같은 단면의 before와 after를 직접 비교해 improved를 기록한다. 실제 선택된 세 단면은 모두 MAE가 줄었다.

오차 지도는 |저선량-정상선량|과 |출력-정상선량|에 HU 폭 3000을 곱해 표시한다. 값은 제한된 HU 체계에서의 차이다. magnitude만 남겨 부호는 사라진다. 색상 범위 0~100에서 100 이상은 같은 끝색이다. 아래 오른쪽의 빈 영역은 off로 숨긴 축이며 누락된 정상선량 오차 이미지가 아니다. 정상선량 자체와의 차이는 0이라 별도 그림을 두지 않았다.

### 핵심 용어

| 이름         | 뜻                             |
| ------------ | ------------------------------ |
| reverse=True | 내림차순 정렬                  |
| 튜플 키      | 여러 값을 묶은 딕셔너리 키     |
| 오차 지도    | 기준과의 픽셀별 절대 차이 표시 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 52 · 노트북 셀 위치 80

출력 MAE가 큰 세 단면을 찾고 같은 단면의 before/after 오차를 비교한다. 오차 지도와 CSV를 저장한다.

```python
# 23번에서 저장한 단면별 지표 읽기
with (run_folder / "val_metrics.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    saved_metrics = list(csv.DictReader(file))

# patient_id와 촬영 위치로 검증 목록에 연결
index_lookup = {
    (row["patient_id"], float(row["z_mm"])): index
    for index, row in enumerate(val_rows)
}

assert len(index_lookup) == len(val_rows), (
    "동일 환자·위치에 중복 단면이 있는지 확인하세요."
)

worst_metrics = sorted(
    saved_metrics,
    key=lambda item: float(item["mae"]),
    reverse=True,
)[:3]

analysis_rows = []

for rank, metric in enumerate(worst_metrics, start=1):
    key = (
        metric["patient_id"],
        float(metric["z_mm"]),
    )

    index = index_lookup[key]

    row, quarter, prediction, full = predict_validation_slice(
        index
    )

    # 같은 단면에서 모델 적용 전후 MAE 비교
    before_mae = float(
        np.abs(
            quarter.astype(np.float64)
            - full.astype(np.float64)
        ).mean()
    )

    after_mae = float(metric["mae"])

    analysis_rows.append({
        "rank": rank,
        "patient_id": row["patient_id"],
        "z_mm": row["z_mm"],
        "before_mae": before_mae,
        "after_mae": after_mae,
        "improved": after_mae < before_mae,
    })

    print(
        f'{rank}위 | 환자 {row["patient_id"]} | '
        f'z={row["z_mm"]} mm | '
        f'MAE {before_mae:.6f} → {after_mae:.6f}'
    )

    # 오차를 HU 단위로 표현
    before_error = np.abs(quarter - full) * hu_span
    after_error = np.abs(prediction - full) * hu_span

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # 윗줄: 원본·복원·비교 기준
    for ax, image, title in zip(
        axes[0],
        [quarter, prediction, full],
        ["Quarter dose", "CNN output", "Full dose"],
    ):
        ax.imshow(
            image * hu_span + HU_MIN,
            cmap="gray",
            vmin=-160,
            vmax=240,
        )
        ax.set_title(title)
        ax.axis("off")

    # 아랫줄: 적용 전후 오차, 동일 색상 범위
    for ax, error, title in [
        (axes[1, 0], before_error, "Error before CNN"),
        (axes[1, 1], after_error, "Error after CNN"),
    ]:
        heatmap = ax.imshow(
            error,
            cmap="magma",
            vmin=0,
            vmax=100,
        )
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(
            heatmap, ax=ax, fraction=0.046, label="Absolute error (HU)"
        )

    axes[1, 2].axis("off")

    fig.suptitle(
        f'High-error slice #{rank} | '
        f'{row["patient_id"]} | z={row["z_mm"]} mm'
    )

    plt.tight_layout()
    fig.savefig(
        visual_folder / f"high_error_{rank}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


# 분석에 사용한 단면과 숫자 저장
with (run_folder / "high_error_slices.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(analysis_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(analysis_rows)

print("오차가 큰 단면 분석 저장 완료")
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print([(r["z_mm"], r["mae"]) for r in worst_metrics])
```

### 이해 확인 문제

1. 어두운 색은 병변이 없다는 뜻일까?

해설: 아니다. 정상선량과의 값 차이가 작다는 뜻이다.

2. worst 3개가 모델이 악화시킨 3개일까?

해설: 아니다. 적용 후 오차가 큰 3개다.

## 26장 최종 테스트 환자 평가

학습 목표: 고정된 모델로 처음 보는 두 환자를 평가한다.

test_model은 검증에서 선택한 checkpoint를 다시 읽어 만든다. test.csv의 환자 ID가 checkpoint의 train·val 목록과 겹치지 않는지 검사한다. 체크포인트의 HU 조건도 확인한다. 테스트에서 가장 좋은 모델을 새로 고르는 루프는 없다.

같은 단면에서 저선량과 CNN 출력을 동일 calculate_metrics로 평가한다. 결과의 baseline_mae와 cnn_mae처럼 접두어를 붙여 한 행에 비교값을 함께 둔다. 이 함수 이름은 18장에 정의했던 calculate_metrics를 덮어쓴다. 현재 핵심 계산은 같지만 입력 변환 코드가 다르므로 같은 이름의 재정의가 실행 상태를 바꾼다는 점을 알아 두자.

검증에서 보지 않은 L333과 L506의 총 455개 단면을 처리한다. 별도의 병원·촬영 장비·재구성 조건을 추가로 평가한 것은 아니다. 원 대회 공개 학습 10명에서 자체 분할한 holdout이다. 같은 환자의 단면들은 상관이 있으므로 455개 독립 사례로 표현하면 안 된다.

### 핵심 용어

| 이름           | 뜻                                            |
| -------------- | --------------------------------------------- |
| holdout        | 개발 중 분리해 둔 평가 데이터                 |
| baseline* cnn* | 한 행 안에서 두 방식의 결과를 구분하는 접두어 |
| 고정 모델      | 테스트를 위해 파라미터를 바꾸지 않는 모델     |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 53 · 노트북 셀 위치 82

체크포인트를 고정하고 테스트 목록의 환자 중복을 검사한다. calculate_metrics 이름을 재정의하고 동일 함수로 baseline·CNN을 평가한다.

```python
import csv
import json
import hashlib
import numpy as np
import torch
from skimage.metrics import structural_similarity

run_folder = ROOT / "outputs" / "cnn_v1"
checkpoint_path = run_folder / "best_model.pt"

# 테스트에서 사용할 모델을 다시 불러오기
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

test_model = SmallResidualCNN().to(device)
test_model.load_state_dict(checkpoint["model_state_dict"])
test_model.eval()

# 학습 당시 정규화 조건과 현재 조건 확인
assert checkpoint["config"]["hu_clip"] == [HU_MIN, HU_MAX]

# 테스트 목록 읽기
with (ROOT / "data" / "splits" / "test.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    test_rows = list(csv.DictReader(file))

assert test_rows
assert all(row["split"] == "test" for row in test_rows)

test_patients = {row["patient_id"] for row in test_rows}

used_patients = set(
    checkpoint["config"]["train_patients"]
    + checkpoint["config"]["val_patients"]
)

assert not (test_patients & used_patients), (
    "테스트 환자가 학습 또는 검증에 포함되어 있습니다."
)

print("테스트 환자:", sorted(test_patients))
print("테스트 단면:", len(test_rows))
print("사용할 모델 epoch:", checkpoint["epoch"])


# 같은 함수로 저선량과 CNN 출력 모두 평가
def calculate_metrics(image, reference):
    image = image.astype(np.float64)
    reference = reference.astype(np.float64)

    assert image.shape == reference.shape
    assert np.isfinite(image).all()
    assert np.isfinite(reference).all()

    difference = image - reference

    mae = float(np.abs(difference).mean())
    mse = float(np.square(difference).mean())

    psnr = (
        float(10 * np.log10(1.0 / mse))
        if mse > 0
        else float("inf")
    )

    ssim = float(
        structural_similarity(
            reference,
            image,
            data_range=1.0,
        )
    )

    return {"mae": mae, "psnr": psnr, "ssim": ssim}


test_metric_rows = []

with torch.no_grad():
    for index, row in enumerate(test_rows, start=1):
        quarter = normalize_hu(read_hu(row["quarter_path"]))
        full = normalize_hu(read_hu(row["full_path"]))

        x = (
            torch.from_numpy(quarter)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        prediction = test_model(x)[0, 0].cpu().numpy()

        baseline_metrics = calculate_metrics(quarter, full)
        cnn_metrics = calculate_metrics(prediction, full)

        result = {
            "patient_id": row["patient_id"],
            "z_mm": row["z_mm"],
        }

        for name in ["mae", "psnr", "ssim"]:
            result[f"baseline_{name}"] = baseline_metrics[name]
            result[f"cnn_{name}"] = cnn_metrics[name]

        test_metric_rows.append(result)

        if index % 100 == 0:
            print(f"평가: {index}/{len(test_rows)}")

print("테스트 평가 완료:", len(test_metric_rows), "개 단면")
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(test_patients)
print(len(test_metric_rows))
print(test_metric_rows[0])
```

### 이해 확인 문제

1. 테스트에서 backward를 호출할까?

해설: 아니다. no_grad로 평가한다.

2. test.csv가 있어도 누출 검사를 하는 이유는?

해설: 목록 이름만 신뢰하지 않고 실제 환자 ID 구성을 확인하기 위해서다.

## 27장 테스트 결과와 평가 조건 저장

학습 목표: 전체 평균과 환자별 평균을 함께 저장한다.

summarize_test_rows는 선택된 행들의 지표 평균을 만든다. overall은 모든 455개 단면 평균이고 by_patient는 환자별로 골라 계산한 결과다. 단면 수가 다른 두 환자의 전체 평균은 환자별 평균을 단순히 둘로 나눈 것과 다를 수 있다. 현재 전체 집계는 단면 수에 따른 비중을 가진다.

hashlib.sha256는 checkpoint 파일 바이트의 식별값을 만든다. 같은 파일을 다시 평가했는지 추적할 때 유용하며 모델 성능이나 안전성을 나타내는 숫자는 아니다. JSON에 모델 epoch, hash, 정규화와 집계 조건을 함께 기록한다. 결과만 따로 떼면 조건을 알 수 없어 재현과 비교가 어려워진다.

실제 전체 테스트 결과는 MAE 0.0043703579에서 0.0030516132, PSNR 43.9329587에서 47.0924435dB, SSIM 0.9652156에서 0.9853983이다. MAE 약 30.2% 감소는 (before-after)/before×100으로 계산한다. SSIM의 0.02018 증가는 SSIM 수치의 차이며 진단 정확도 2.018%p 향상으로 표현하지 않는다.

### 핵심 용어

| 이름        | 뜻                                |
| ----------- | --------------------------------- |
| aggregation | 여러 단면·환자 결과를 합치는 방법 |
| SHA256      | 파일 바이트에 기반한 식별값       |
| by_patient  | 환자별 집계 결과                  |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 54 · 노트북 셀 위치 84

전체·환자별 평균을 만들고 모델 hash와 평가 조건을 JSON으로 저장한다. \*by_patient.items()는 출력할 튜플들을 목록 안에 펼친다.

```python
metric_names = ["mae", "psnr", "ssim"]


def summarize_test_rows(rows):
    return {
        "slice_count": len(rows),
        "baseline": {
            name: float(np.mean([
                row[f"baseline_{name}"] for row in rows
            ]))
            for name in metric_names
        },
        "cnn": {
            name: float(np.mean([
                row[f"cnn_{name}"] for row in rows
            ]))
            for name in metric_names
        },
    }


overall = summarize_test_rows(test_metric_rows)

by_patient = {}

for patient_id in sorted(test_patients):
    selected = [
        row for row in test_metric_rows
        if row["patient_id"] == patient_id
    ]

    by_patient[patient_id] = summarize_test_rows(selected)


# 단면별 결과 저장
with (run_folder / "test_metrics.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(test_metric_rows[0].keys()),
    )
    writer.writeheader()
    writer.writerows(test_metric_rows)


# 평가에 사용한 모델 파일을 식별하는 값
checkpoint_hash = hashlib.sha256(
    checkpoint_path.read_bytes()
).hexdigest()

test_summary = {
    "split": "test",
    "patients": sorted(test_patients),
    "checkpoint_epoch": checkpoint["epoch"],
    "checkpoint_sha256": checkpoint_hash,
    "hu_clip": [HU_MIN, HU_MAX],
    "data_range": 1.0,
    "evaluation_region": "전체 단면, 배경 포함",
    "aggregation": "단면별 지표의 산술평균",
    "prediction_clipping": False,
    "overall": overall,
    "by_patient": by_patient,
}

(run_folder / "test_summary.json").write_text(
    json.dumps(test_summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


# 결과 출력
for label, result in [
    ("전체", overall),
    *by_patient.items(),
]:
    print(f"\n[{label}] 단면 {result['slice_count']}개")

    for name in metric_names:
        before = result["baseline"][name]
        after = result["cnn"][name]

        print(
            f"{name.upper():5s}: "
            f"{before:.6f} → {after:.6f}"
        )

print("\n테스트 결과와 평가 조건 저장 완료")
```

#### 원본 코드 셀 55 · 노트북 셀 위치 85

%pip는 Jupyter/IPython 명령이다. 패키지를 설치하는 셀이므로 이미 설치했다면 읽기 실습에서 재설치하지 않아도 된다.

```python
%pip install onnx onnxscript onnxruntime
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(overall)
print(by_patient.keys())
```

### 이해 확인 문제

1. 전체 단면 평균과 환자 평균의 평균은 같을까?

해설: 환자별 단면 수가 다르면 일반적으로 다르다.

2. hash가 높으면 모델이 좋은가?

해설: hash는 성능값이 아니라 식별값이다.

## 28장 최적 모델을 ONNX로 변환

학습 목표: 학습된 계산을 ONNX 파일로 내보낸다.

ONNX는 모델의 계산 그래프와 파라미터를 다른 실행 환경에서 사용할 수 있게 저장하는 형식이다. torch.onnx.export는 PyTorch 계산을 ONNX로 내보내고 ONNX Runtime은 그 파일을 실행한다. onnx와 onnxruntime은 역할이 다른 패키지다. %pip는 현재 Jupyter 커널 환경에 패키지를 설치하는 IPython 명령으로 일반 Python 파일의 문법과 다르다.

export_model은 CPU에 만든 같은 구조이며 checkpoint 파라미터를 넣고 eval로 전환한다. example_input은 실제 학습 데이터가 아니라 내보낼 때 입력 모양을 알려주는 예시다. (example_input,)의 쉼표는 항목 하나짜리 튜플을 만드는 데 필요하다. dynamo=True는 현재 PyTorch의 export 경로를 사용하며 external_data=False는 작은 가중치를 ONNX 파일 안에 담는다.

현재 dynamic_shapes를 지정하지 않았으므로 입력 크기는 [1,1,512,512]로 고정된다. 128 패치를 넣으면 런타임의 shape 오류가 날 수 있다. 구조 검사 onnx.checker.check_model은 ONNX 구성이 유효한지 확인하는 것이고 실제 출력의 일치는 다음 장에서 확인한다. DICOM 읽기, HU clip, 정규화는 이 파일에 포함하지 않았다. 배포 시에도 같은 전처리가 필요하다.

### 핵심 용어

| 이름         | 뜻                                         |
| ------------ | ------------------------------------------ |
| ONNX         | 계산 그래프와 파라미터 저장 형식           |
| ONNX Runtime | ONNX 모델을 실행하는 런타임                |
| 고정 입력    | 변환에서 지정한 batch·channel·height·width |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 56 · 노트북 셀 위치 87

CPU 모델을 ONNX로 내보낸다. zeros는 모양을 제공할 예시 입력이며 새 학습 데이터가 아니다. 구조 검사와 export 조건을 저장한다.

```python
import json
import torch
import onnx

run_folder = ROOT / "outputs" / "cnn_v1"
export_folder = run_folder / "export"
export_folder.mkdir(parents=True, exist_ok=True)

# 저장된 최적 모델 읽기
checkpoint = torch.load(
    run_folder / "best_model.pt",
    map_location="cpu",
    weights_only=True,
)

# CPU에서 변환할 별도 모델 생성
export_model = SmallResidualCNN()
export_model.load_state_dict(checkpoint["model_state_dict"])
export_model.eval()

onnx_path = export_folder / "small_residual_cnn.onnx"

# 이번 버전은 전체 CT 단면 한 장 입력으로 고정
example_input = torch.zeros(
    1, 1, 512, 512,
    dtype=torch.float32,
)

torch.onnx.export(
    export_model,
    (example_input,),
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    dynamo=True,
    external_data=False,
)

# ONNX 파일의 구조 검사
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)

export_config = {
    "checkpoint_epoch": checkpoint["epoch"],
    "input_shape": [1, 1, 512, 512],
    "input_dtype": "float32",
    "hu_clip": checkpoint["config"]["hu_clip"],
    "normalization": "(clip(HU) - HU_MIN) / (HU_MAX - HU_MIN)",
    "preprocessing_included": False,
    "prediction_clipping": False,
    "opsets": {
        item.domain or "ai.onnx": item.version
        for item in onnx_model.opset_import
    },
}

(export_folder / "export_config.json").write_text(
    json.dumps(export_config, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("ONNX 변환·구조 검사 완료")
print("파일:", onnx_path)
print("파일 크기:", onnx_path.stat().st_size, "bytes")
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(onnx_path.exists())
print(export_config["input_shape"])
print(export_config["opsets"])
```

### 이해 확인 문제

1. ONNX 변환은 추가 학습일까?

해설: 아니다. 학습한 모델의 계산을 다른 형식으로 내보낸다.

2. HU 변환도 ONNX 안에 들어 있나?

해설: 현재 export는 CNN만 포함하며 전처리는 밖에서 수행한다.

## 29장 PyTorch와 ONNX 출력 일치 확인

학습 목표: 같은 입력에 대한 두 실행 방식의 출력을 비교한다.

InferenceSession은 모델을 읽고 실행할 준비를 하는 객체다. get_inputs()[0].name으로 파일의 입력 이름을 확인하고 run(None,{이름:배열})로 모든 출력을 요청한다. 반환값은 출력들의 리스트이므로 [0]이 첫 출력 Tensor 배열이고 이후 [0,0]으로 배치·채널을 고를 수 있다.

quarter[None,None,:,:]는 NumPy에서 배치와 채널 축을 추가한다. ascontiguousarray는 float32 값을 연속 메모리 배치로 준비한다. 원래 2차원 데이터의 픽셀 값은 같고 구조만 [1,1,512,512]가 된다. PyTorch와 ONNX를 둘 다 CPU에서 실행해 숫자를 비교한다.

assert_allclose는 |a-b| <= atol + rtol×|기준값| 조건에 해당하는 허용 오차 검사를 수행한다. 완전히 같은 비트값보다 수치적 일치를 확인한다. 실제 세 단면 최대 차이는 5.9604645e-08로 통과했다. 이 값은 ONNX와 PyTorch 출력의 차이이고 정상선량과의 복원 오차가 아니다. 세 단면 검사만으로 모든 가능한 입력의 일치를 검증한 것은 아니다.

### 핵심 용어

| 이름        | 뜻                           |
| ----------- | ---------------------------- |
| session.run | 입력 이름과 배열로 ONNX 실행 |
| contiguous  | 숫자가 연속 배치된 메모리    |
| parity      | 변환 전후의 출력 일치 확인   |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 57 · 노트북 셀 위치 89

세 단면에서 PyTorch와 ONNX 결과를 비교한다. run 결과의 첫 항목을 가져오고 float32 허용 오차로 검사한다.

```python
import csv
import numpy as np
import onnxruntime as ort

# 검증 목록 다시 읽기
with (ROOT / "data" / "splits" / "val.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    parity_rows = list(csv.DictReader(file))

# CPU에서 ONNX 실행
session = ort.InferenceSession(
    str(onnx_path),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name

print("ONNX 입력:", session.get_inputs()[0].shape)
print("실행 장치:", session.get_providers())

# 앞·중간·끝 단면으로 확인
selected_indices = sorted({
    0,
    len(parity_rows) // 2,
    len(parity_rows) - 1,
})

parity_results = []

for index in selected_indices:
    row = parity_rows[index]

    quarter = normalize_hu(
        read_hu(row["quarter_path"])
    )

    input_array = np.ascontiguousarray(
        quarter[None, None, :, :],
        dtype=np.float32,
    )

    assert input_array.shape == (1, 1, 512, 512)

    # PyTorch 출력: CPU
    with torch.no_grad():
        pytorch_output = export_model(
            torch.from_numpy(input_array)
        ).numpy()

    # ONNX Runtime 출력: CPU
    onnx_output = session.run(
        None,
        {input_name: input_array},
    )[0]

    assert pytorch_output.shape == onnx_output.shape
    assert np.isfinite(onnx_output).all()

    difference = np.abs(pytorch_output - onnx_output)

    result = {
        "patient_id": row["patient_id"],
        "z_mm": row["z_mm"],
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
    }
    parity_results.append(result)

    print(
        f'z={row["z_mm"]} mm | '
        f'최대 차이={difference.max():.8f} | '
        f'평균 차이={difference.mean():.8f}'
    )

    # 부동소수점 계산의 작은 차이를 허용
    np.testing.assert_allclose(
        onnx_output,
        pytorch_output,
        rtol=1e-4,
        atol=1e-5,
    )

parity_report = {
    "provider": "CPUExecutionProvider",
    "rtol": 1e-4,
    "atol": 1e-5,
    "checked_slice_count": len(parity_results),
    "results": parity_results,
    "passed": True,
}

(export_folder / "parity_report.json").write_text(
    json.dumps(parity_report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("선택한 3개 단면에서 출력 일치 검사 통과")
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(session.get_inputs()[0].name)
print(session.get_inputs()[0].shape)
print(parity_report["passed"])
```

### 이해 확인 문제

1. 6e-08은 복원 MAE인가?

해설: 아니다. 두 실행 방식의 출력 차이다.

2. 모델 checker와 parity는 같은 검사인가?

해설: checker는 모델 구조, parity는 실제 출력의 비교다.

## 30장 CPU 추론 시간 비교

학습 목표: 측정 조건을 통제하고 실행 시간의 범위를 기록한다.

perf_counter는 경과 시간을 측정하는 시계다. 종료 시각에서 시작 시각을 빼고 1000을 곱해 밀리초로 바꾼다. 입력 배열 준비와 모델 로딩은 타이머 밖에 있으므로 포함되지 않는다. 두 방식 모두 CPU에서 입력 하나를 처리하고 출력 배열을 반환한다. 이것은 전체 앱의 응답 시간이나 GPU 비교가 아니다.

먼저 다섯 번 실행해 초기 준비 비용을 제외하고 30번 반복 측정한다. 두 방식의 실행 순서를 번갈아 정해 한쪽이 항상 먼저 실행되는 영향을 줄인다. 연산 내부 스레드 1개로 설정하지만 컴퓨터의 전력 상태와 다른 프로그램 등 모든 요인을 통제한 것은 아니다. try/finally는 오류가 나도 PyTorch의 원래 스레드 수를 복구한다.

inference_mode는 추론에서 자동미분과 추가 추적 부담을 줄이는 컨텍스트다. 학습 루프를 이 안에 넣으면 정상적인 backward 학습과 맞지 않는다. mean은 평균, median은 중앙값, percentile(...,95)는 P95다. 비율은 PyTorch 중앙값/ONNX 중앙값이며 1보다 크면 이번 조건에서 ONNX 시간이 짧다.

실제 중앙값은 170.77ms와 133.23ms다. 약 1.28배의 시간 비율과 약 22%의 시간 감소는 같은 결과를 다른 방식으로 표현한다. 같은 영상 반복 30회 결과이며 노트북 전력·열 상태에 따라 재측정 값이 달라질 수 있다. 원시 측정값과 환경 버전까지 저장해 해석할 근거를 남겼다.

### 핵심 용어

| 이름    | 뜻                                  |
| ------- | ----------------------------------- |
| warmup  | 초기 준비를 위해 측정 전 실행       |
| P95     | 측정값의 약 95%가 이 값 이하인 경계 |
| finally | 오류 여부와 관계없이 정리 코드 실행 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 58 · 노트북 셀 위치 91

준비된 입력에서 출력 배열까지의 CPU 시간만 잰다. warmup·반복·교대 순서를 사용하고 finally로 스레드 설정을 되돌린다.

```python
import time
import platform
import onnxruntime as ort

# 검증 환자의 가운데 영상 사용
row = parity_rows[len(parity_rows) // 2]

quarter = normalize_hu(read_hu(row["quarter_path"]))

benchmark_input = np.ascontiguousarray(
    quarter[None, None, :, :],
    dtype=np.float32,
)
benchmark_tensor = torch.from_numpy(benchmark_input)

export_model.cpu().eval()

# ONNX CPU 실행 설정
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.add_session_config_entry(
    "session.intra_op.allow_spinning", "0"
)

benchmark_session = ort.InferenceSession(
    str(onnx_path),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)

benchmark_input_name = benchmark_session.get_inputs()[0].name

WARMUP = 5
REPEATS = 30

times = {"pytorch": [], "onnx": []}


def run_pytorch():
    return export_model(benchmark_tensor).numpy()


def run_onnx():
    return benchmark_session.run(
        None,
        {benchmark_input_name: benchmark_input},
    )[0]


original_threads = torch.get_num_threads()

try:
    torch.set_num_threads(1)

    with torch.inference_mode():
        # 초기 실행 준비 비용을 측정에서 제외
        for _ in range(WARMUP):
            run_pytorch()
            run_onnx()

        # 실행 순서를 번갈아 배치
        for repeat in range(REPEATS):
            order = (
                ["pytorch", "onnx"]
                if repeat % 2 == 0
                else ["onnx", "pytorch"]
            )

            for name in order:
                started = time.perf_counter()

                if name == "pytorch":
                    run_pytorch()
                else:
                    run_onnx()

                elapsed_ms = (
                    time.perf_counter() - started
                ) * 1000

                times[name].append(elapsed_ms)

finally:
    torch.set_num_threads(original_threads)


timing_summary = {
    name: {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }
    for name, values in times.items()
}

benchmark_report = {
    "device": "CPU",
    "intra_op_threads": 1,
    "input_shape": list(benchmark_input.shape),
    "dtype": str(benchmark_input.dtype),
    "warmup": WARMUP,
    "repeats": REPEATS,
    "scope": "준비된 입력부터 출력 배열 생성까지",
    "excluded": [
        "DICOM 파일 읽기", "정규화", "모델 불러오기"
    ],
    "platform": platform.platform(),
    "processor": platform.processor(),
    "torch_version": str(torch.__version__),
    "onnxruntime_version": ort.__version__,
    "results": timing_summary,
    "raw_times_ms": times,
}

# 저장할 설명문은 한국어로 작성한다
benchmark_report["scope"] = "준비된 입력에서 출력 배열까지"
benchmark_report["excluded"] = [
    "DICOM 읽기", "정규화", "모델 로딩"
]

(export_folder / "benchmark_cpu.json").write_text(
    json.dumps(benchmark_report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

for name, result in timing_summary.items():
    print(
        f"{name:7s} | "
        f'중앙값 {result["median_ms"]:.2f} ms | '
        f'평균 {result["mean_ms"]:.2f} ms | '
        f'P95 {result["p95_ms"]:.2f} ms'
    )

ratio = (
    timing_summary["pytorch"]["median_ms"]
    / timing_summary["onnx"]["median_ms"]
)

print(f"PyTorch 시간 / ONNX 시간: {ratio:.2f}")
print("추론 시간과 측정 조건 저장 완료")
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
print(timing_summary)
print(170.77 / 133.23)
print((170.77 - 133.23) / 170.77 * 100)
```

### 이해 확인 문제

1. DICOM 읽는 시간도 포함됐을까?

해설: 아니다. 준비된 입력에서 출력 배열까지 측정했다.

2. ONNX로 바꾸면 항상 빨라질까?

해설: 현재 장비·조건에서 측정한 결과이며 일반적인 보장은 아니다.

## 31장 ONNX 기반 CT 복원 비교 데모

학습 목표: 앞 셀에 의존하지 않는 ONNX 비교 화면을 실행한다.

최신 31장 코드는 슬라이더 대신 일반 그림 호출 방식이다. ipywidgets 표시가 오래 대기한 뒤 run_folder가 없다는 오류가 생겨 필요한 경로와 함수를 이 셀에서 직접 준비하도록 보완했다. %matplotlib inline은 그림을 노트북 출력에 표시하는 IPython 설정이다. 원본 31장에는 현재 위젯 생성 코드가 없다.

독립 셀이라고 원본 데이터와 모델이 필요 없는 것은 아니다. ROOT의 Windows 절대 경로, val.csv, export_config.json, ONNX 파일과 CSV 안의 원본 경로가 실제로 있어야 한다. 앞 셀의 변수에는 의존하지 않지만 디스크 파일에는 의존한다. Mac에서는 ROOT와 CSV 경로를 새 환경에 맞춰 준비해야 한다.

read_normalized_ct는 read_hu와 normalize_hu를 한 함수에 넣는다. config에서 학습 당시 HU_MIN,HU_MAX를 읽어 동일한 전처리를 적용한다. show_demo는 인덱스로 선택한 저선량만 모델에 넣고 정상선량은 비교 그림에 사용한다. run(...)[0][0,0]은 첫 출력, 배치 첫 장, 첫 채널의 2차원 배열을 얻는다. 이미지 저장 후 close로 그림 객체를 정리한다.

flush=True는 진행 문구를 즉시 내보내 어느 단계에 도달했는지 알기 쉽게 한다. 'ONNX 모델 불러오는 중', 'DICOM 읽는 중', 'ONNX 추론 중', '비교 그림 만드는 중' 중 마지막 문구로 지연 위치를 좁힐 수 있다. 13분은 기존 단면 추론 약 0.13초와 맞지 않아 커널·화면 출력·파일 접근을 별도로 확인해야 한다. 아래 연습은 인덱스만 바꿔 다른 단면을 본다.

### 핵심 용어

| 이름      | 뜻                                    |
| --------- | ------------------------------------- |
| 독립 셀   | 필요한 변수·함수를 해당 셀에서 준비   |
| flush     | 출력 버퍼를 즉시 비워 진행 표시       |
| show_demo | 단면 선택부터 추론·비교 그림까지 실행 |

### 원본 코드를 따라 읽기

#### 원본 코드 셀 59 · 노트북 셀 위치 93

최신 데모 셀은 필요한 import·경로·전처리를 자체 정의한다. 앞 셀을 실행하지 않아도 모델·CSV·원본 데이터가 있으면 비교 화면을 만들 수 있다.

```python
%matplotlib inline

from pathlib import Path
import csv
import json
import time

import numpy as np
import pydicom
import matplotlib.pyplot as plt
import onnxruntime as ort

# 1. 필요한 경로를 이 셀에서 직접 정의
ROOT = Path(r"C:\Users\User\projects\ct-denoising")

run_folder = ROOT / "outputs" / "cnn_v1"
export_folder = run_folder / "export"
demo_folder = run_folder / "demo"

demo_folder.mkdir(parents=True, exist_ok=True)

onnx_path = export_folder / "small_residual_cnn.onnx"

assert onnx_path.exists(), f"ONNX 파일이 없습니다: {onnx_path}"

# 2. 모델 변환 당시 정규화 조건 읽기
config = json.loads(
    (export_folder / "export_config.json")
    .read_text(encoding="utf-8")
)

HU_MIN, HU_MAX = config["hu_clip"]
hu_span = HU_MAX - HU_MIN

# 3. 검증 목록 읽기
with (ROOT / "data" / "splits" / "val.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    demo_rows = list(csv.DictReader(file))

assert demo_rows, "검증 목록이 비어 있습니다."

# 4. DICOM을 읽고 정규화하는 함수
def read_normalized_ct(path):
    ds = pydicom.dcmread(path)

    pixels = ds.pixel_array.astype(np.float32)

    hu = (
        pixels * float(ds.RescaleSlope)
        + float(ds.RescaleIntercept)
    )

    normalized = (
        np.clip(hu, HU_MIN, HU_MAX) - HU_MIN
    ) / hu_span

    return normalized.astype(np.float32)


# 5. ONNX CPU 실행 환경 생성
print("ONNX 모델 불러오는 중", flush=True)

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.add_session_config_entry(
    "session.intra_op.allow_spinning", "0"
)

demo_session = ort.InferenceSession(
    str(onnx_path),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)

demo_input_name = demo_session.get_inputs()[0].name

# 6. 선택한 단면을 복원하고 표시
def show_demo(slice_index):
    row = demo_rows[slice_index]

    print("DICOM 읽는 중", flush=True)

    quarter = read_normalized_ct(row["quarter_path"])
    full = read_normalized_ct(row["full_path"])

    assert quarter.shape == full.shape == (512, 512)

    input_array = np.ascontiguousarray(
        quarter[None, None, :, :],
        dtype=np.float32,
    )

    print("ONNX 추론 중", flush=True)

    started = time.perf_counter()

    prediction = demo_session.run(
        None,
        {demo_input_name: input_array},
    )[0][0, 0]

    elapsed_ms = (time.perf_counter() - started) * 1000

    print("비교 그림 만드는 중", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, image, title in zip(
        axes,
        [quarter, prediction, full],
        ["Quarter dose", "ONNX output", "Full dose"],
    ):
        ax.imshow(
            image * hu_span + HU_MIN,
            cmap="gray",
            vmin=-160,
            vmax=240,
        )
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f'{row["patient_id"]} | '
        f'Slice {slice_index} | z={row["z_mm"]} mm'
    )

    plt.tight_layout()

    fig.savefig(
        demo_folder / "latest_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(f"ONNX 추론 시간: {elapsed_ms:.2f} ms")
    print("비교 화면 생성 완료", flush=True)


# 7. 가운데 단면 실행
show_demo(len(demo_rows) // 2)
```

### 직접 확인하기

앞의 필요한 정의 셀을 실행한 뒤 메모용 셀에서 확인한다.

```python
show_demo(50)
# 유효한 범위는 0부터 len(demo_rows)-1까지다.
```

### 이해 확인 문제

1. run_folder NameError가 왜 났을까?

해설: 현재 커널에 그 변수가 정의되지 않았기 때문이다.

2. 데모 실행마다 학습할까?

해설: 저장된 ONNX를 읽어 추론하며 학습은 하지 않는다.

## 부록 A 실행 순서와 변수 지도

- 초기 파일 읽기: 1장 ROOT, 2장 files, 3장 ds. 한 환자 파일 탐색·변환.
- 데이터 품질검사: 7장 read_hu, 8장 scan_patient, 9장 same_geometry, 10장 pair_slices. 14장 전체 QC 파일 생성.
- 환자 분할: 14장 all_pairs.csv와 patient_summary.csv. 15장 train/val/test.csv 생성.
- 패치 학습: 16장 normalize_hu·crop_pair, 20장 SmallResidualCNN 정의. 21장 Dataset 준비 후 22장 전체 학습.
- 최적 모델 평가: 모델 구조 정의, ROOT·device, 저장된 checkpoint·분할 CSV. 23장 검증, 26장 테스트.
- ONNX 내보내기: SmallResidualCNN 구조, best_model.pt, 설치된 ONNX 패키지. 28장 export.
- ONNX 단독 데모: 원본 파일·val.csv·ONNX·export_config.json, 31장 셀. 앞의 학습 루프 실행 없이 추론.

### 이름별 자료형

- ROOT: Path. 프로젝트 폴더
- files: list[Path]. 한 환자 원본 경로
- ds: pydicom Dataset. 한 DICOM 파일
- hu: ndarray(H,W). HU 배열
- full quarter: list[dict]. 헤더 목록
- pairs: list[tuple]. 정상·저선량 짝
- pair_rows train_rows val_rows: list[dict]. 경로·환자 정보
- x y: Tensor(N,C,H,W). 모델 입력·비교 기준; Dataset 단일 반환은 C,H,W
- model: nn.Module. 메모리의 모델
- loss: 0차원 Tensor. 자동미분과 연결된 손실
- checkpoint: dict. 디스크에서 읽은 모델 상태·조건
- prediction: Tensor 또는 ndarray. 문맥에 따라 달라짐
- run_folder: Path. CNN 결과 폴더
- session demo_session: InferenceSession. ONNX 실행 객체

## 부록 B 자주 나는 오류와 해결 순서

### train_rows가 정의되지 않음

현재 커널에 학습 목록을 읽은 셀이 실행되지 않았다. 16장 또는 17장 목록 읽기 셀을 먼저 실행한다. CSV가 있는 것과 그 내용이 메모리 변수에 있는 것은 다르다.

### SmallResidualCNN이 정의되지 않음

19장에는 ResidualDenoiser가 있고 20장에 SmallResidualCNN이 정의된다. 20장의 정의 셀을 실행하되 fixed_x 준비가 먼저 필요하다. 최종 모델의 구조를 독립 정의하는 방법은 부록 실행 순서를 참고한다.

### run_folder가 정의되지 않음

22장 학습 셀에서 변수를 만들었지만 새 커널에서는 없다. ROOT를 준비한 뒤 run_folder = ROOT / "outputs" / "cnn_v1"로 직접 경로를 정의한다. 변수 하나를 복원했다고 다른 함수까지 생기는 것은 아니다.

### 패키지가 없다고 나옴

sys.executable로 현재 커널을 확인하고 해당 환경에 설치한다. GPU PyTorch는 장비·드라이버와 맞는 공식 설치 지침을 확인한다. 이미 사용하는 GPU 패키지를 임의로 CPU 패키지로 덮어쓰지 않는다.

### 파일이 없다고 나옴

ROOT와 CSV 안의 절대 경로를 확인한다. 다른 컴퓨터에서 원본 데이터를 옮긴 경우 QC·분할 경로 목록을 새 위치에서 재생성한다.

### 모델과 입력 장치가 다름

모델의 parameter.device와 입력 Tensor.device를 확인한다. 둘 다 같은 장치로 옮긴다. numpy 변환은 CPU에서 하며 학습 그래프가 있으면 detach().cpu().numpy()가 필요하다.

### ONNX 입력 shape 오류

현재 ONNX는 [1,1,512,512] float32 입력이다. [512,512]나 [4,1,128,128]은 현재 export 조건과 다르다.

### 데모가 계속 로딩됨

실행 중단 후 짧은 print 셀로 커널 응답을 확인한다. 최신 31장의 독립 일반 출력 셀로 확인하고 마지막 진행 문구를 살핀다. 저장된 이미지 표시와 모델 추론 시간을 구분한다.

### 학습을 반복했는데 결과가 달라짐

모델 생성·RNG 상태·샘플 호출·라이브러리와 장치 영향을 확인한다. 고정 패치 루프만 재실행하면 기존 파라미터를 이어 학습하며 전체 학습 셀은 새 모델을 만든다.

### 학습 Loss가 검증보다 큼

학습은 랜덤 패치, 검증은 배경을 포함한 전체 단면이고 환자도 다르다. 크기만 직접 비교해 오류나 과적합으로 단정하지 않는다.

### 첫 읽기 실습의 독립 복구 예제

```python
from pathlib import Path
import csv
import numpy as np
import pydicom

ROOT = Path(r"C:\Users\User\projects\ct-denoising")
with (ROOT / "data/splits/train.csv").open(encoding="utf-8-sig", newline="") as f:
    train_rows = list(csv.DictReader(f))

def read_hu(path):
    ds = pydicom.dcmread(path)
    return ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

HU_MIN, HU_MAX = -1000.0, 2000.0
def normalize_hu(image):
    return ((np.clip(image, HU_MIN, HU_MAX)-HU_MIN)/(HU_MAX-HU_MIN)).astype(np.float32)

row = train_rows[0]
image = read_hu(row["quarter_path"])
print(image.shape, image.dtype)
print(normalize_hu(image).min(), normalize_hu(image).max())

```

## 부록 C 실제 결과와 발표 설명

검증 L310 | MAE | 0.005310 | 0.003432
검증 L310 | PSNR | 42.708836 | 46.498366
검증 L310 | SSIM | 0.953715 | 0.982527
테스트 전체 | MAE | 0.004370 | 0.003052
테스트 전체 | PSNR | 43.932959 | 47.092443
테스트 전체 | SSIM | 0.965216 | 0.985398

왜 이 작은 모델을 선택했나?

데이터·학습·평가·ONNX 연결을 먼저 검증할 첫 기준 모델로 단순한 구조를 사용했다. 후보 모델보다 좋다는 비교 실험을 한 것은 아니다.

왜 L1을 사용했나?

픽셀 오차를 직접 최소화하는 첫 손실로 사용했다. L2와의 성능 비교는 후속 실험이다.

정상선량은 추론 입력인가?

학습·평가의 비교 기준이며 실제 모델 입력은 저선량 한 장이다.

어느 부분이 직접 수행한 일인가?

파일 품질검사, 짝짓기, 환자 분할, 모델 구현과 학습, 조건을 고정한 평가, ONNX 내보내기와 출력 검사, 데모 연결이다.

가장 중요한 한계는?

테스트 환자 수 2명, 단일 출처·조건, 짧은 단일 실행, ROI·진단 평가와 모델 비교가 없다는 점이다.

## 부록 D 용어 사전

- 배열: 같은 자료형의 숫자를 정해진 축 모양으로 담는 구조.
- Tensor: PyTorch의 다차원 숫자 배열. 장치 연산과 자동미분에 연결할 수 있다.
- 채널: 각 위치에서 보유한 값 또는 특징의 축. 흑백 입력은 1채널이다.
- 배치: 한 번의 모델 실행·업데이트에 함께 쓰는 샘플 묶음.
- epoch: 전체 학습 목록을 한 번 사용하는 단위.
- step: 한 배치에서 최적화 업데이트를 수행하는 단위.
- DICOM: 영상과 촬영 정보를 함께 담는 의료영상 표준 형식.
- HU: CT에서 감쇠 정도를 나타내는 Hounsfield Unit.
- 메타데이터: 영상 숫자 외의 위치·간격·두께·시리즈 같은 정보.
- 시리즈: 함께 구성되는 영상들의 그룹. SeriesInstanceUID로 식별한다.
- 정규화: 정한 규칙으로 숫자 범위를 변환하는 처리.
- clipping: 범위 밖 값을 최솟값 또는 최댓값으로 제한하는 처리.
- 윈도 표시: 특정 숫자 범위를 색상에 대응시키는 화면 표시 설정.
- 패치: 원본 영상에서 잘라낸 작은 영역.
- 합성곱: 주변 픽셀·특징에 필터를 적용하는 연산.
- ReLU: max(0,x)로 계산하는 활성화 함수.
- 잔차 연결: 입력 또는 중간값을 변환 결과에 더하는 연결.
- 파라미터: 학습으로 바꾸는 가중치와 편향.
- 하이퍼파라미터: 학습률·배치 크기 등 실험자가 정하는 조건.
- Loss: 예측과 비교 기준의 차이를 하나의 수로 표현한 학습 목적.
- 기울기: 파라미터 변화에 대한 손실 변화의 방향·크기 정보.
- 역전파: 계산 그래프를 거꾸로 따라 기울기를 계산하는 방법.
- Optimizer: 기울기를 이용해 파라미터를 변경하는 알고리즘.
- Adam: 기울기의 이동 평균 등을 이용하는 최적화 알고리즘.
- 학습 모드: 모델의 train 설정. 그 자체가 학습 루프는 아니다.
- 평가 모드: 모델의 eval 설정. 자동미분 비활성화와는 별개다.
- no_grad: 자동미분 기록을 생략하는 실행 컨텍스트.
- inference_mode: 추론용으로 자동미분과 추가 추적 부담을 줄이는 컨텍스트.
- Dataset: 샘플의 길이와 가져오는 처리 방법을 정의하는 객체.
- DataLoader: Dataset 샘플을 순회하고 배치로 묶는 도구.
- state_dict: 모델 파라미터·등록된 버퍼를 이름별로 담는 사전.
- checkpoint: 특정 시점의 모델 상태와 관련 정보를 저장한 파일.
- baseline: 개선 여부를 비교하기 위한 기준 방법.
- 검증: 개발 중 모델을 선택하고 설정을 확인하는 평가.
- 테스트: 선택된 모델의 성능을 분리된 데이터에서 확인하는 평가.
- MAE: 평균 절대 오차.
- MSE: 평균 제곱 오차.
- PSNR: 정해진 수치 범위와 MSE로 계산한 dB 지표.
- SSIM: 국소 밝기·대비·구조의 유사성을 사용하는 지표.
- ONNX: 모델 계산과 파라미터를 저장하는 교환 형식.
- Runtime: 저장한 모델을 실행하는 환경.
- Parity: 변환 전후 결과가 수치적으로 일치하는지 확인.
- Warmup: 측정 전 준비 실행.
- P95: 측정값의 약 95%가 해당 값 이하인 경계.
- Seed: 난수 생성기의 초기 상태를 정하는 값.
- float32: 32비트 부동소수점 자료형.
- float64: 64비트 부동소수점 자료형.
- contiguous: 연속적인 메모리 배치.
- NaN: 수치 계산에서 정상적인 숫자로 표현할 수 없는 값.
- inf: 무한대 표현.
- manifest: 원본 파일과 짝·조건의 위치를 기록한 목록.
- CSV: 행·열 형태의 텍스트 데이터 형식.
- JSON: 사전·목록 등 중첩 구조를 저장하는 텍스트 형식.
- kernel: 문맥에 따라 Jupyter 실행 엔진, CT 재구성 커널, CNN 필터를 뜻할 수 있어 구분해야 한다.

## 부록 E 원본 코드 수록 지도

- 1장 라이브러리와 프로젝트 경로: 코드 셀 1
- 2장 파일 경로 찾기: 코드 셀 2
- 3장 DICOM 파일 읽기: 코드 셀 3
- 4장 픽셀 배열 꺼내기: 코드 셀 4
- 5장 저장된 숫자를 HU로 바꾸기: 코드 셀 5
- 6장 첫 단면을 그림으로 보기: 코드 셀 6
- 7장 재사용할 HU 읽기 함수: 코드 셀 7
- 8장 환자의 DICOM 헤더 모으기: 코드 셀 8, 9
- 9장 같은 위치와 재구성 조건인지 비교하기: 코드 셀 10
- 10장 영상 짝 목록 만들기: 코드 셀 11, 12
- 11장 저선량·정상선량·차이 영상 비교: 코드 셀 13
- 12장 CSV 짝 목록과 검사 결과 저장: 코드 셀 14
- 13장 저장한 CSV에서 다시 읽어오기: 코드 셀 15
- 14장 전체 환자의 데이터 검사: 코드 셀 16, 17, 18, 19
- 15장 환자 단위 학습·검증·테스트 분리: 코드 셀 20, 21, 22, 23, 24, 25
- 16장 HU 정규화와 학습용 패치 만들기: 코드 셀 26, 27, 28, 29, 30, 31
- 17장 PyTorch Dataset과 DataLoader: 코드 셀 32, 33, 34, 35, 36, 37
- 18장 저선량 영상의 비교 기준 평가: 코드 셀 38, 39, 40, 41, 42
- 19장 작은 잔차 CNN 만들기: 코드 셀 43
- 20장 고정된 패치 4장으로 학습 검사: 코드 셀 44, 45, 46
- 21장 전체 학습 데이터 준비: 코드 셀 47
- 22장 전체 학습·검증 및 최적 모델 저장: 코드 셀 48, 49
- 23장 최적 모델의 검증 성능 평가: 코드 셀 50
- 24장 복원 영상 비교: 코드 셀 51
- 25장 오차가 큰 검증 단면 분석: 코드 셀 52
- 26장 최종 테스트 환자 평가: 코드 셀 53
- 27장 테스트 결과와 평가 조건 저장: 코드 셀 54, 55
- 28장 최적 모델을 ONNX로 변환: 코드 셀 56
- 29장 PyTorch와 ONNX 출력 일치 확인: 코드 셀 57
- 30장 CPU 추론 시간 비교: 코드 셀 58
- 31장 ONNX 기반 CT 복원 비교 데모: 코드 셀 59

## 자료 출처와 더 읽기

- 프로젝트 원본: 01_ct_read_and_pair.ipynb와 outputs/cnn_v1의 실제 결과 파일
- AAPM 데이터 공식 설명: https://www.aapm.org/GrandChallenge/LowDoseCT/
- 사용한 데이터 다운로드 폴더: https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/145241239366
- pydicom Dataset 기초: https://pydicom.github.io/pydicom/stable/tutorials/dataset_basics.html
- PyTorch Dataset와 DataLoader: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- PyTorch 자동미분: https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- Conv2d 공식 문서: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html
- L1Loss 공식 문서: https://docs.pytorch.org/docs/2.14/generated/torch.nn.L1Loss.html
- PyTorch ONNX 공식 문서: https://docs.pytorch.org/docs/2.14/onnx.html
- ONNX Runtime Python: https://onnxruntime.ai/docs/get-started/with-python.html
- ONNX Runtime 스레드 설정: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- NumPy isclose: https://numpy.org/doc/stable/reference/generated/numpy.isclose.html
- scikit-image structural_similarity: https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity
