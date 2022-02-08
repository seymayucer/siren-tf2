# Siren TF : Implicit Neural Representations with Periodic Activation Functions

The unofficial Tensorflow 2 implementation of the paper Implicit Neural Representations with Periodic Activation Functions. Please note that, this repo tested with image fitting experiments.

### [Paper](https://arxiv.org/abs/2006.09661) | [Official PyTorch Implementation](https://github.com/vsitzmann/siren)


## Get Started
To start working with this repository please install Python packages using:

```sh
conda env create --file setup/environment.yaml
```

## Training
```sh
python main.py --is_train --input_image samples/durham_mcs.jpg  --output_dir results/ 
```

## Testing

```sh
python main.py --input_image samples/durham_mcs.jpg --output_dir results/ 
```

## Results

![example 1](/results/istanbul_airport.png)
![example 2](/results/leaves.png)

## References

- [Siren Pytorch Official Repository](https://github.com/vsitzmann/siren/)
- [Siren Tensorflow Repository](https://github.com/titu1994/tf_SIREN)


