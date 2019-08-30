# ManifoldMixup
Unofficial implementation of [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) (Proceedings of ICML 19) in PyTorch with support for [Interpolated Adversarial](https://arxiv.org/pdf/1906.06784.pdf) training. 

## Results of adversarial training on MNIST

|  Dataset  | Adversarial training  | Normal training |
|  -------  | --------------------- | --------------- |
|  MNIST  | 0.6779  | 0.0069  |

Adversarial training done using Interpolated adversarial training framework with FGSM attack (eps=0.3) and MixupManifold
## Usage
### ManifoldMixup training
```python
from manifold_mixup import ManifoldMixupDataset, ManifoldMixupModel, ManifoldMixupLoss
"""
Wrap your dataset, model and loss in ManifoldMixup classes that's it!
"""
mixup_ds = ManifoldMixupDataset(trn_ds)
mixup_model = ManifoldMixupModel(model, alpha=0.2)
mixup_criterion = ManifoldMixupLoss(criterion)
"""
Now train as usual using mixup dataset, model and loss
"""
```
### Interpolated Adversarial training with ManifoldMixup
```python
from manifold_mixup import ManifoldMixupDataset, ManifoldMixupModel
from adversarial_attacks import FGSM
from interpolated_adversarial import InterpolatedAdversarialLoss

mixup_ds = ManifoldMixupDataset(trn_ds)
mixup_model = ManifoldMixupModel(model, alpha=0.2, interpolated_adv=True)

"""
To define loss for interpolated adversarial training, you need to pass attack and
your original loss function
"""
adv_loss = nn.CrossEntropyLoss()
model_loss = nn.CrossEntropyLoss()
fgsm = FGSM(adv_loss)
adv_criterion = InterpolatedAdversarialLoss(model_loss, fgsm)
"""
Now train as usual with new model, dataset and loss
"""
```
### Creating custom adversarial attack
```python
from adversarial_attacks import BlackBoxAdversarialAttack
"""
BlackBoxAdversarialAttack automatically handles your model's parameter's requirement for grad
and creates a callable compatible with InterpolatedAdversarialLoss
"""
class MyNewAttack(BlackBoxAdversarialAttack):
  def run(self, x, y, model):
    """
    Logic for attack
    """
```
