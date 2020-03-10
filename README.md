# ManifoldMixup
Unofficial implementation of [ManifoldMixup](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) (Proceedings of ICML 19) in PyTorch with support for [Interpolated Adversarial](https://arxiv.org/pdf/1906.06784.pdf) training. 

## Results of adversarial training on MNIST

|  Dataset  | Adversarial training  | Normal training |
|  -------  | --------------------- | --------------- |
|  MNIST  | 0.732  | 0.0069  |

Adversarial training done using Interpolated adversarial training framework with FGSM attack (eps=0.3) and MixupManifold

## Update
Following [discussion with author](https://github.com/vikasverma1077/manifold_mixup/issues/8) the following features are added:</br>
1. Module level control for user which allows deciding specifically which layers to consider for mixup
2. Warning is raised if a module is called more than once in forward pass, and mixup is done at first instance only
3. If for some reason, you cannot pre-decide which modules to use for mixup, pass ```mixup_all=True``` while creating ```ManifoldMixupModel``` instance 
## Usage
### ManifoldMixup training
```python
from manifold_mixup import ManifoldMixupDataset, ManifoldMixupModel, ManifoldMixupLoss, MixupModule
"""
(optional)
Wrap modules which you want to use for mixup using MixupModule
Example:
class Model(nn.Module):
    def __init__(self, in_dims, hid_dims, out_dims):
        super(Model, self).__init__()
        self.m = nn.Sequential(Flatten(),
                              LinearLayer(in_dims, hid_dims, use_bn=True),
                              MixupModule(LinearLayer(hid_dims, hid_dims, use_bn=True)),
                              MixupModule(LinearLayer(hid_dims, hid_dims, use_bn=True)),
                              nn.Linear(hid_dims, out_dims))
    def forward(self, x):
        return self.m(x)
"""
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
#### Fastai implementation by @nestordemeure
https://github.com/nestordemeure/ManifoldMixup
