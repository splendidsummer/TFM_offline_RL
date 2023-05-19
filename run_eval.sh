cd ~/Projects/rrc_2022_offlineRL/

python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=aug --probs=0.001
python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=raw --probs=0.001

python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=aug --probs=0.01
python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=raw --probs=0.01

python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=aug --probs=0.05
python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=raw --probs=0.05

python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=aug --probs=0.1
python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=raw --probs=0.1

python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=aug --probs=0.15
python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=raw --probs=0.15

python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=aug --probs=0.2
python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=raw --probs=0.2

python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=aug --probs=0.25
python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=raw --probs=0.25

python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=aug --probs=0.3
python evaluate_policy.py --task=push --algorithm=bc --n_epochs=10 --aug=raw --probs=0.3