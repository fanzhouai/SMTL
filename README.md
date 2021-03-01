# Semantic Multi-task Learning Algorithm

Codes for AAAI 2021 paper 'Multi-task Learning by Leveraging the Semantic Information'


## 1. Download the dataset

To reproduce the experiments, please first download the official release of the dataset files into the specific folders. Please make sure the folder structure keeps the same with configure.py. If you want to use different folders, please modify the configure.py and dloader.py to point the path accordingly.


## 2. To run the experiments on digits dataset:
	cd Exp_digits
	python3 train_digits.py --re_weighting 1 --initial_lr 1e-3 --train_samples 3000 # You can change number of training samples as you wish (3000, 5000, or 8000)
	  
## 3. To run the experiments on vision dataset (PACS for example):
	cd Exp_vision
	python3 train.py --dataset pacs --initial_lr 2e-4 --ratio 0.1 # you can modify the ratio as you wish to run the experiments, you can replace pacs with office or office-home etc.. to run the exps.	


## 4. To run the experiments under label shift: (results on Figure 3)
 	cd Exp_vision
	python3 train.py --dataset office31 --initial_lr 2e-4 --ratio 0.2 --drift_ratio 0.1 # You can modify the drift_ratio as you wish (we report the results of drift ratio from 0.1 to 0.8)
	python3 train.py --dataset office_home --initial_lr 2e-4 --ratio 0.2 --drift_ratio 0.5 # You can modify the drift_ratio as you wish (we report the results of drift ratio from 0.1 to 0.8)
