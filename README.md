# Antitifake
Consolidated deepfake detection solution
<!--
[![eBPF Emerging Project](https://img.shields.io/badge/ebpf.io-Emerging--App-success)](https://ebpf.io/projects#loxilb) [![Go Report Card](https://goreportcard.com/badge/github.com/loxilb-io/loxilb)](https://goreportcard.com/report/github.com/loxilb-io/loxilb) ![build workflow](https://github.com/loxilb-io/loxilb/actions/workflows/docker-image.yml/badge.svg) ![sanity workflow](https://github.com/loxilb-io/loxilb/actions/workflows/basic-sanity.yml/badge.svg) ![apache](https://img.shields.io/badge/license-Apache-blue.svg) [![Info][docs-shield]][docs-url] [![Slack](https://img.shields.io/badge/community-join%20slack-blue)](https://www.loxilb.io/members)  
-->

## What is Antitifake

Antitifake is an open source deepfake detection tool for consolidated methods.

![demo](https://github.com/riverallzero/antitifake/assets/93754504/c984f90c-799e-49f5-b3a9-b9057c34d4d5)

### 📦 Antitifake aims to provide the following :   
#### Detect
- Facial Attributes: bangs, eyeglasses, beard, smiling, young
- Facial Components: nose, eye, eyebrow, lip, hair


### 🧿 Antitifake is composed of:       
#### Model
- [SeqDeepFake](https://github.com/rshaojimmy/SeqDeepFake): detect deepfake and manipulated part(attributes, components)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO): boxing the manipulated part on image

#### Code
<code>python [app.py](https://github.com/riverallzero/antitifake/blob/main/app.py)</code>

### To-Do :       
- Import SeqDeepFake to antitifake
    * analyze gradio code
    * integrate into app.py
- Use grounding dino for visualize
    * analyze grounding dino and import
- Import other baelines
    * [Candidates](https://github.com/Daisy-Zhang/Awesome-Deepfakes-Detection)



 
### 📚 Check Antitifake [Documentation](https:///) for more info.   

[docs-shield]: https://img.shields.io/badge/info-docs-blue
[docs-url]: https://loxilb-io.github.io/loxilbdocs/
[slack=shield]: https://img.shields.io/badge/Community-Join%20Slack-blue
[slack-url]: https://www.loxilb.io/members
