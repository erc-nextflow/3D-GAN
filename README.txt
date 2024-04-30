DATA AND CODE AVAILABILITY RELATED TO:
"3D generative adversarial networks for turbulent flow estimation from wall measurements"
by Antonio Cuéllar, Alejandro Güemes, Andrea Ianiro, Óscar Flores, Ricardo Vinuesa and Stefano Discetti
published in the Journal of Fluid Mechanics

----------------------------------------------------------------------------------------------------------------

GitHub: https://github.com/erc-nextflow/3D-GAN

-> FOLDER 'python codes' contains:
	-> codes for each case to train the network using tensorflow
	-> codes for each case to estimate the flow using a trained model
	-> code 'matfileconverter.py' to conver the estimation output (.npy) into matlab files (.mat)

-> FOLDER 'channel coordinates' contains:
	-> files 'coordX.npy', 'coordY.npy' an 'coordZ.npy'. These files may be useful to reconstruct the geometry of the channel.

-> FILE 'uv2.m' is a matlab script used to develop the analysis found in section 3.4 of the article.
	To run this code it is necessary to convert the output of the network into .mat files, using 'matfileconverter.py'

----------------------------------------------------------------------------------------------------------------

ZENODO REPOSITORY: https://doi.org/10.5281/zenodo.11090713

-> FOLDER 'models' with the trained models of the network, contain the weights of each layer.
	-> CASE A: case *A03
	-> CASE B: case *B04
	-> CASE C: case *C04
	-> CASE D: case *D04

	-> CASE A with 16 residual blocks (FIG 12 a): case *A01
	-> CASE C with 48 residual blocks (FIG 12 b): case *C05
	-> CASE C with 56 residual blocks (FIG 12 b): case *C06

-> FOLDER 'tfrecords' contains: (THE CODES ARE PREPARED TO READ THE DATASETS IN THIS FORMAT)
	-> 'scaling.npz' file, needed for training and testing
	-> 'train' folder with training/validation dataset. 10 files are included in this repository due to storage restrictions. If more samples were needed, data can be shared upon request.
	-> 'test' folder with testing dataset. 4000 samples. 
	