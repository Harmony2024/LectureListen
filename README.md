# SmartLecture
> **2023 동계 AI SCI 부트캠프 – “인공지능을 통한 음성인식”**  
> 🏆 최우수상 수상 **(참가 8 팀 중 2 팀 선발)**

한국어 대학강의 음성(4800 h)을 이용해 **ESPnet**으로 STT 모델을 학습하고  
음성 ↔︎ ChatGPT 대화를 지원하는 서비스를 구축했습니다.

## 🎯 Goals
- 한국어 대학강의 도메인에서 **STT 성능(CER) 25 % 이상 개선**
- 오류 유형(이중 전사, 드문 음절, 속도 변동 등) 분석을 통한 **개선 포인트 도출**
- 학습된 모델을 **실시간 음성-ChatGPT 서비스**로 연결


## ⚙️ 환경

| 항목 | 사양                                 |
|------|------------------------------------|
| GPU | **RTX A6000 48 GB×4** (fp16, AMP)  |
| CUDA/cuDNN | 11.8 / 8.x                         |
| Driver | ≥ 525                              |
| Python | 3.10 (Anaconda)                    |
| PyTorch | 2.1.1+cu118                        |
| ESPnet | commit `76b318e` (2023-10 release) |
| sox / flac | latest                             |



## 📂 프로젝트 구조
```
code/espnet                    # espnet 코드
service/                       # FastAPI + Javascript
docs/                          # 발표 자료
README.md
```


## 1. 데이터 준비

AIHUB 한국어 대학강의 ▶ Training 1 712 897 utt / Validation 193 257 utt.

```bash
# 압축 해제 & 무결성
$ find source/Training -name '*.wav' |wc -l 
$ find label/Training  -name '*.json'|wc -l  
# 리스트 → wav.scp / text 생성
$ bash local/mkdataset.sh datasets/lists ./data   # 8:1:1 split
```


## 2. 레시피 설정

```bash
$ task=asr1
$ egs2/TEMPLATE/${task}/setup.sh egs2/aihub_kuniv/${task}
# ksponspeech→conf 복사 후 run.sh 수정 (token_type=char, --use_lm false)
```

- **모델** Conformer encoder-decoder (SpecAugment, encoder depth 12).
- **디코더** Beam 10, CTC/Attn α = 1.0 (LM off).


## 3. 학습 파이프라인

| Stage | 내용 | 시간 |
|-------|------|------|
| 3–5 | 데이터 ↔︎ flac, 길이 필터, token_list | 1 h |
| 10 | 통계 수집 | 1 h |
| 11 | **Conformer 35 epochs** (batch_bins 80 M, fp16) | **~44 h** |
| 12–13 | 디코딩(valid 480 h) & CER | 24 h |

명령어:
```bash
$ bash run.sh --stage 3 --ngpu 1 
```


## 4. 주요 실험 요약

| 변화점                                                     | 데이터 | CER (%) | 시간 |
|---------------------------------------------------------| --- | --- | --- |
| Baseline                                                | 1.5 M | 12.4 | 2 d |
| 이중전사 → 한글                                               | 1.5 M | 11.6 | +0 |
| HuggingFace KoCharELECTRA tokenizer                              | 0.2 M | 20.3 | 6 h |
| Speed Perturb                                           | 0.07 M | 30.7 | 6 h |
| LM dropout 0.3                                          | 1.5 M | 9.8 | +6 h |
| 최종 (이중 전사 -> 한글, KoCharELECTRA tokenizer, LM dropout 0.3) | 1.5 M | **9.3** | 2 d |

최종 **ERR 25 %↓** (12.4 → 9.3)


## 5. 서비스 (음성 ChatGPT)

- **/stt** 16 k wav → 텍스트 (Conformer 모델)
- **/chat** 텍스트 → GPT-3 응답
- **/voice-chat** 음성 업로드 ↔︎ STT ↔︎ GPT 대화 UI


## 6. 향후 과제

1. **LM 의 역효과** – 조사·어미 삭제 등으로 CER 악화; domain-specific LM 정제가 필요.
2. **Tokenizer** – 대규모 데이터에선 char 만으로도 성능 포화. 소규모·희귀 도메인에선 HF tokenizer 유용.


## demo
<div align="center"> <table> <tr> <td align="center"><strong>🖼️ 구성 </strong></td> <td align="center"><strong>📽️ STT 및 ChatGPT 응답</strong></td> </tr> <tr> <td> <img src="https://github.com/user-attachments/assets/5470e3cf-34b3-4431-a425-db7022c37587" width="400"/> </td> <td> <img src="https://github.com/cshooon/LectureListen/assets/113033780/88dc14a1-e47c-4591-bb5f-5cf2cc8bdef2" width="400"/> </td> </tr> </table> </div>


## Members

| 이름 | GitHub | 역할 |
|------|--------|------|
| 최승훈 | [github.com/cshooon](https://github.com/cshooon) | 모델 학습, 음성인식 서비스 |
| 강연주 | [github.com/sally7788](https://github.com/sally7788) | 발표, 모델 학습 |
| 진민찬 | [github.com/JinMinChan](https://github.com/JinMinChan) | 데이터 전처리, 음성인식 서비스 |
| 박율 | [github.com/Park-Yul](https://github.com/Park-Yul) | PPT 제작, 모델 학습 |


## 소감

**최승훈**  
인공지능 학습을 colab이 아닌 서버에서 할 수 있다는 점이 좋았다. A6000 4대를 활용할 수 있었고 다양한 실험을 할 수 있었다. 처음에 batch_bin을 작게 실험해서 학습을 더 많이 돌리지 못해 아쉽다. 그렇지만 이중 전사, tokenizer 등 여러 실험을 해보았고 최종 학습에서 소폭 성능 향상을 보였다. 겨울 방학 기간에 뜻깊은 시간을 보낸 것 같고 함께해준 팀원에게 고마웠다는 말을 전하고 싶다.

**강연주**  
실제 서버에 연결하여 학습을 진행할 수 있어서 실무와 비슷한 경험을 쌓을 수 있었다. 실제 데이터를 활용해 학습 방향을 찾고 평가하는 것은 처음이어서 부족한 점이 많았지만, 학습 결과를 토대로 좋은 성능을 낼 수 있어서 만족감이 컸다. 이번 부트캠프에서 가장 중요한 것은 학습 결과를 판단하고 추후 학습 방향을 정하는 것이라고 생각된다. 따라서 결과를 분석하는 힘을 길러야겠다고 다짐했다. 팀원들과 함께 했어서 뜻깊은 결과를 낼 수 있었다.

**진민찬**  
인공지능 학습이라는 큰 흐름과 이를 서비스화 하는 단계를 경험해 볼 수 있던 점이 가장 인상깊었다. 평소 인공지능에 관심이 많았지만 이론적인 내용만 알고 있어 크게 와닿지는 않았었다. 하지만 실제로 모델의 성능을 향상시키기 위해 데이터의 전처리는 어떻게 해야할지, 어떤 모델과 파라미터값을 정해야 할지, 또한 비록 성능은 향상되지만 시간이 너무 오래걸린다면 이를 최적화할 수 있는 방안이나 시간 대비 성능 향상이 적절한지 등을 고민해보며 성장할 수 있는 계기가 되었다.

**박율**  
인공지능 부트캠프를 통해 인공지능 모델을 최적화하는 과정을 경험해본 것이 가장 의미있었던 것 같다. 인공지능에 대해 아무것도 알고있지 못했지만 팀원들과 함께 연구를 진행하면서 인공지능에 대한 전체적인 틀을 알아가고, 알게 된 내용을 기반으로 리눅스 환경에서 데이터셋 전처리와 인공지능 레시피 모델링, 학습 진행과 결과 분석 등 인공지능을 활용하는 일련의 과정을 경험해볼 수 있었다. 비록 팀에 큰 도움은 되지 못했지만 이번 부트캠프를 기점으로, 관심 있었던 인공지능 반도체 관련 분야의 연구에 한 발자국 다가갈 수 있게 될 것 같다.

