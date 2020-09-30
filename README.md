Asymmetric Contextual Modulation
==============

MXNet/Gluon code for "Asymmetric Contextual Modulation for Infrared Small Target Detection" <>

What's in this repo so far:

 * Code and trained models for the SIRST (Single-frame InfraRed Small Target detection) dataset

The SIRST dataset is available at <https://github.com/YimianDai/sirst>
 
## Requirements
 
Install [MXNet](https://mxnet.apache.org/) and [Gluon-CV](https://gluon-cv.mxnet.io/):
  
```
pip install --upgrade mxnet-cu100 gluoncv
```

## Experiments 

The trained model params and training logs are in `./params`

The training commands / shell scripts are in `cmd_scripts.txt`

<img src=https://raw.githubusercontent.com/YimianDai/imgbed/master/github/acm/ACM-Tab-3.png width=100%>


<img src=https://raw.githubusercontent.com/YimianDai/imgbed/master/github/acm/ACM_ROC_all.png width=50%>



