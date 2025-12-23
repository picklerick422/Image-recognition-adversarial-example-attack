# Image-recognition-adversarial-example-attack
浙江大学控制科学与工程学院《信息安全导论》课程作业

## 文件组成

- ResNet.py 为图像识别程序文件、包含了攻击方法的实现
- attack.py 为攻击方法的实现文件
- defense_experiment.py 为防御实验程序文件
- picture 文件夹为实验图片文件夹
- adv_fgsm.png 为FGSM攻击生成的对抗样本
- adv_pgd.png 为PGD攻击生成的对抗样本
- adv_cw.png 为CW-L2攻击生成的对抗样本

## 需要的python库（我的python是3.12）
- torch
- torchvision
- torchattacks
- PIL
- matplotlib
- robustbench
- autoattack(下载：pip install git+https://github.com/fra31/auto-attack)

## 如何进行图像识别
- 只做正常预测：
```bash
python ResNet.py picture
```

## 如何进行攻击运行
- 只做正常预测：
```bash
python ResNet.py example.jpg --topk 5
```

- FGSM：
```bash
python ResNet.py example.jpg --attack fgsm --eps 0.031372549 --topk 5 --save_adv adv_fgsm.png
```
- PGD（更强，更容易成功）：
```bash
python ResNet.py example.jpg --attack pgd --eps 0.031372549 --alpha 0.007843137 --steps 10 --topk 5 --save_adv adv_pgd.png
``` 
- CW-L2（较慢，建议先用较小步数验证）：
```bash
python ResNet.py example.jpg --attack cw --cw_steps 10 --cw_lr 0.01 --cw_c 1.0 --topk 5 --save_adv adv_cw.png
```     
- 目标攻击（把模型推到指定类别）(这个是属于CW攻击方法的）：
```bash
python ResNet.py example.jpg --attack cw --target 805 --cw_steps 10 --cw_lr 0.01
``` 
- 防御实验（对比攻击前和攻击后的模型预测结果）：
```bash(例)
python defense_experiments.py --image example.jpg --attacks fgsm pgd --eps_list 0.015686275 0.031372549
```
得到结果：
```
attack=fgsm, eps=0.01569, attack_success=0.000, smooth_defense_acc=1.000, quant_defense_acc=1.000, detector_clean_pass_rate=1.000, detector_adv_flag_rate=1.000, detector_attack_success=0.000
attack=fgsm, eps=0.03137, attack_success=0.000, smooth_defense_acc=1.000, quant_defense_acc=1.000, detector_clean_pass_rate=1.000, detector_adv_flag_rate=0.000, detector_attack_success=0.000
attack=pgd, eps=0.01569, attack_success=1.000, smooth_defense_acc=1.000, quant_defense_acc=0.000, detector_clean_pass_rate=1.000, detector_adv_flag_rate=0.000, detector_attack_success=1.000
attack=pgd, eps=0.03137, attack_success=1.000, smooth_defense_acc=0.000, quant_defense_acc=0.000, detector_clean_pass_rate=1.000, detector_adv_flag_rate=0.000, detector_attack_success=1.000
```

## 如何进行防御实验
- 对比攻击前和攻击后的模型预测结果：
```bash
python defense_experiments.py `
  --model_type standard `
  --image_dir picture `
  --attacks fgsm pgd `
  --eps_list 0.015686275 0.031372549
```
```bash
python defense_experiments.py `
  --model_type robust `
  --image_dir picture `
  --attacks fgsm pgd `
  --eps_list 0.015686275 0.031372549
```