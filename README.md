# Antitifake
Consolidated deepfake detection solution
<!--
[![eBPF Emerging Project](https://img.shields.io/badge/ebpf.io-Emerging--App-success)](https://ebpf.io/projects#loxilb) [![Go Report Card](https://goreportcard.com/badge/github.com/loxilb-io/loxilb)](https://goreportcard.com/report/github.com/loxilb-io/loxilb) ![build workflow](https://github.com/loxilb-io/loxilb/actions/workflows/docker-image.yml/badge.svg) ![sanity workflow](https://github.com/loxilb-io/loxilb/actions/workflows/basic-sanity.yml/badge.svg) ![apache](https://img.shields.io/badge/license-Apache-blue.svg) [![Info][docs-shield]][docs-url] [![Slack](https://img.shields.io/badge/community-join%20slack-blue)](https://www.loxilb.io/members)  
-->

## What is Antitifake

Antitifake is an open source deepfake detection tool for consolidated methods.

### 📦 Antitifake aims to provide the following :   
#### Detect
predict input image is deepfake, then return True and extract manipulated parts with image if the resulting value is higher than threshold(0.5).

#### true
- Facial Attributes: bangs, eyeglasses, beard, smiling, young
  
  ![attributes](./.asset/attribute_result.png)
  
- Facial Components: nose, eye, eyebrow, lip, hair
  
  ![components](./.asset/component_result.png)

#### False

  ![false](./.asset/false_result.png)

### Setting
#### 1. set trained model files(.pt)
- this source contain trained model
  - GPU: NVIDIA RTX A5000
  - BACKBONE: resnet50
  - ACC: facial_attributes=50.1550%, facial_components=49.3966%
- if you want to use your own model
  1. [SeqDeepFake](https://github.com/rshaojimmy/SeqDeepFake) ```sh train.sh &```
  2. put model file in each folder, [📁 facial_attributes](./SeqDeepFake/results/facial_attributes) & [📁 facial_components](./SeqDeepFake/results/facial_components)
  3. change files [seqdeepfake_.py](./seqdeepfake_.py)
     ```python
     # line 10
     attribute_checkpoint = './SeqDeepFake/results/facial_attributes/best_model_adaptive.pt'
     component_checkpoint = './SeqDeepFake/results/facial_components/best_model_adaptive.pt'
     # line 15
     model_config = Config('./SeqDeepFake/configs/r50.json')
     ```
  
#### 2. to run app.py
⚠️You must set **cuda compute capability**(TORCH_CUDA_ARCH_LIST=compute capability) before install requirements.txt
1. ```export TORCH_CUDA_ARCH_LIST=8.6```
2. ```pip3 install -r requirements.txt```

#### 3. set threshold
[app.py](app.py)

```python
# line 94
if __name__ == '__main__':
    threshold = 0.5
    demo.launch(share=True)
```

### 🧿 Antitifake is composed of:       
#### Model
- [SeqDeepFake](https://github.com/rshaojimmy/SeqDeepFake) : detect deepfake and manipulated part(attributes, components)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) : boxing the manipulated part on image
- [TruFor](https://github.com/grip-unina/TruFor) : Rely on the extraction of both high-level and low-level traces through a transformer-based fusion architecture that combines the RGB image and a learned noise-sensitive fingerprint.

#### Code
<code>python [app.py](app.py)</code>

### To-Do :       
- [x] Import SeqDeepFake to antitifake
  - ~~analyze gradio code~~
  - ~~integrate into app.py~~

- [x] Use grounding dino for visualize
  - ~~analyze grounding dino and import~~
- Import other baselines
    * [Candidates](https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection)
- Models to Be Tested
    * [DNA-Det](https://github.com/ICTMCG/DNA-Det) : a method for Deepfake Network Architecture Attribution to attribute fake images on architecture-level.



 
### 📚 Check Antitifake [Documentation](https:///) for more info.   

[docs-shield]: https://img.shields.io/badge/info-docs-blue
[docs-url]: https://loxilb-io.github.io/loxilbdocs/
[slack=shield]: https://img.shields.io/badge/Community-Join%20Slack-blue
[slack-url]: https://www.loxilb.io/members
