# Leveraging SSL Speech Features and Mamba for Enhanced DeepFake Detection

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="mamba.png" width="15%" alt="spoofing_ssl_mamba" />
</div>
<div align="center" style="line-height: 1;">

  <br>
  <a href="https://www.isca-archive.org/interspeech_2025/tran25b_interspeech.html"><b>Paper Link</b></a>
</div>

<hr>

This Repository contains the code and pretrained model for the following INTERSPEECH 2025 paper:

* **Title** : Leveraging SSL Speech Features and Mamba for Enhanced DeepFake Detection
* **Autor** : Hoan My Tran, Damien Lolive, David Guennec, Aghilas Sini, Arnaud Delhay, Pierre-François Marteau


## Installation

```
conda create -n spoofing_ssl_mamba_env python=3.10
conda activate spoofing_ssl_mamba_env
pip install -r requirements.txt
```
## Training

To launch the training, modify the **config.yaml** file:
- Change *protocol_path* for both *train* and *val* in dataset
The protocole file path should be under the form without headers
*file_name,label*
which *file_name* is the path to the *file_name*.

Example:

```
../dataset/LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac,bonafide
../dataset/LA/ASVspoof2019_LA_train/flac/LA_T_1271820.flac,bonafide
```
- Train the model:
```
python main.py
```
## Evaluation

To evaluate, modify the **config.yaml** file:
- Change the *eval: true* in evaluation
- Change the *protocol_path* to the protocole file path.
- Provide the checkpoint for *checkpoint* in *model*

The protocole file path should be under the form
*file_name,label*

Example:

```
file_name,label
../dataset/ASVspoof2021_DF_eval/flac/DF_E_2000049.flac,spoof
../dataset/ASVspoof2021_DF_eval/flac/DF_E_2000053.flac,bonafide
```


which *file_name* is the path to the *file_name*.

Then run:
```
python main.py
```

Change the configuration in **config.yaml** file if necessary

## Checkpoint
The checkpoint is found [here](https://huggingface.co/hoanmyTran/ssl_spoofing_mamba).


## Citation
If you find this repository useful, please consider citing:
```
@inproceedings{tran25b_interspeech,
  title     = {{Leveraging SSL Speech Features and Mamba for Enhanced DeepFake Detection}},
  author    = {{Hoan My Tran and Damien Lolive and David Guennec and Aghilas Sini and Arnaud Delhay and Pierre-François Marteau}},
  year      = {{2025}},
  booktitle = {{Interspeech 2025}},
  pages     = {{5323--5327}},
  doi       = {{10.21437/Interspeech.2025-1703}},
  issn      = {{2958-1796}},
}
```

## Acknowledgements
Our implementations use the source code from the following repositories:

- [RawBoost-antispoofing](https://github.com/TakHemlata/RawBoost-antispoofing)
- [espnet](https://github.com/espnet/espnet)
- [asv-subtools](https://github.com/Snowdar/asv-subtools)
