# EP-DNN-testing

Research Artifact of ISSTA 2024 Paper: *See the Forest, not Trees: Unveiling and Escaping the Pitfalls of Error-Triggering Inputs in Neural Network Testing*


## Transformation

The following transformations are implemented in `transformation.py`.

### Previous Transformations

- [x] Pixel-based: 1) adding noise, 2) changing brightness, 3) changing contrast, 4) blurring, 5) translation, 6) scale, 7) rotation, 8) shear, 9) reflection
- [x] Style-based (weather): 10) cloudy, 11) foggy, 12) snowy, 13) rainy

Artistic-style transfer is implemented as `Stylize` in `transformation.py`.

### Perception-Based Transformation

Perception-based transformation is implemented using `Perceptual` in `transformation.py`.

### Generative Models

We use the [official implementation](https://github.com/ajbrock/BigGAN-PyTorch) of BigGAN for generating perception-based transformations for ImageNet. We have extracted some code and configurations from the official BigGAN repo to ease the import; please refer to the files in the `BigGAN` folder (which need to be put into the root directory of the official BigGAN repo). 

The StyleGANs are provided by [genforce](https://github.com/genforce/genforce), which are used to generate perception-based transformations for several specialized datasets. We have revised the original `synthesize.py` in the genforce repo to ease the import; our revised script is provided as `genforce/synthesize.py`.

To generate perception-based transformation, set up the above repos and run `generate_T.py`.

## Testing Methods

We consider 1) objective-guided testing, 2) test input prioritization, and 3) random testing.

### Objectives

The following objectives are implemented in `objectives.py`.

- [x] Neuron Coverage (NC)
- [x] Top-K Neuron Coverage (TKNC)
- [x] NeuraL Coverage (NLC)
- [x] Likelihood Surprise Adequacy/Coverage (LSA/LSC)
- [x] Entropy-based black-box objective (Entropy)

For an objective, e.g., NC, simply calling `NC.gain(x)` to check
whether `x` increases the coverage.

🌟 Check [this repo](https://github.com/Yuanyuan-Yuan/NeuraL-Coverage) for implementations of more objectives. 🌟

### Prioritization Metrics

The following prioritization metrics are implemented in `objectives.py`.

- [x] Kullback-Leibler divergence (KL)
- [x] Jensen-Shannon divergence (JS)

For a metric, e.g., KL, calling `KL.priority(x, mutated_x)` to compute
the priority if of `mutate_x`.

### Random Testing

Two random testing schems, namely *ARand* and *PRand* are implemented
in `Test.py`.

## Tested Models

The five ImageNet DNNs (i.e., ResNet, VGG, MobileNet, DenseNet, Inception) are officially provided by Pytoch; see `prepare_model.py` to how to use them.

FaceNet, AlexNet, and EfficientNet are implemented in `models/face.py`, `models/animal.py`,
and `models/car.py`, respectively. 

## Experiments

### Generating Perception-Based Transformations

Run `python generate_T.py` to generate transformations.

### RQ1

ETI related experiments: run `python Test_multi.py --mode input` and `python Test.py --mode input` for ImageNet-trained DNNs and other DNNs, respectively.

EP related experiments: run `python Test_multi.py --mode property` and `python Test_multi.py --mode input` for ImageNet-trained DNNs and other DNNs, respectively.

### RQ2

For ETI- and EP-oriented repairing, run `python Repair.py`.

## Acknowledgements

- The `jacobian.py` is adapted from the [resefa repo](https://github.com/zhujiapeng/resefa).

- The `style_operator.py` is adapted from the [Stylized-ImageNet repo](https://github.com/rgeirhos/Stylized-ImageNet).

We sincerely thank the authors for sharing their code.
