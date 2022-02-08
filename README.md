# Siren: Implicit Neural Representations with Periodic Activation Functions

The unofficial Tensorflow 2 implementation of the paper Implicit Neural Representations with Periodic Activation Functions. Please note that, this repo tested with image fitting experiments.

### [Paper](https://arxiv.org/abs/2006.09661) | [Official PyTorch Implementation](https://github.com/vsitzmann/siren)


## Get Started
To start working with this repository please install Python packages using:

```sh
conda env create --file setup/environment.yaml
```

## Training
```sh
python main.py --train --input_image samples/durham_mcs.jpg --model_name siren --output_dir results/durham_mcs,
```

## Testing

```sh
python main.py --input_image samples/durham_mcs.jpg --model_name siren --output_dir results/durham_mcs/
```

## Results

|   |   |
|---|---|
| ![](/results/istanbul_airport.png) | ![](/results/leaves.png)    |
| ![](/results/face.png)   |![](/results/stone_nsm.png)   |



## References

- [Siren Pytorch Official Repository](https://github.com/vsitzmann/siren/)
- [Siren Tensorflow Repository](https://github.com/titu1994/tf_SIREN)


