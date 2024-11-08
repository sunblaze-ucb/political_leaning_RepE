# Representation Engineering (RepE) for Political Neutrality
In our paper (https://arxiv.org/pdf/2410.24190), we applied representation engineering (https://arxiv.org/abs/2310.01405) to mitigate LLM political leaning. 

## Installation

To install `repe` from the github repository main branch, run:

```bash
git clone https://github.com/sunblaze-ucb/political_leaning_RepE.git
cd representation-engineering
pip install -e .
```
## Quickstart

```python
from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines
repe_pipeline_registry()

# ... initializing model and tokenizer ....

rep_reading_pipeline =  pipeline("rep-reading", model=model, tokenizer=tokenizer)
rep_control_pipeline =  pipeline("rep-control", model=model, tokenizer=tokenizer, **control_kwargs)
```

## RepReading and RepControl Experiments For Political Neutrality
Check examples/political_bias

For Llama 3.1 8B, please run llama8_main.py or llama8_control.py

For Llama 3.1 70B, please run llama70.main.py

## Citation
If you find this useful in your research, please consider citing both:

```
@article{potter2024hidden,
  title={{Hidden Persuaders: LLMs' Political Leaning and Their Influence on Voters}}
  author={Potter, Yujin and Lai, Shiyang and Kim, Junsol and Evans, James and Song, Dawn},
  journal={arXiv preprint arXiv:2410.24190},
  year={2024}
}
```

```
@misc{zou2023transparency,
      title={{Representation Engineering: A Top-Down Approach to AI Transparency}}, 
      author={Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, Zico Kolter, Dan Hendrycks},
      year={2023},
      eprint={2310.01405},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
