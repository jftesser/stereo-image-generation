## Stereo (Side by Side) Image Generation in cython

### Setup 

1) Download the model weights [model-f45da743.pt](https://drive.google.com/file/d/1l_w6Jny_erNQpgc8-nzBa_adh4bBDaFw/view?usp=sharing) and place the
file in the root folder.

2) Set up dependencies: 

    ```shell
	pip install pytorch  
	pip install torchvision
	pip install opencv_python
	```

   The code was tested with Cuda 10.1, Python 3.6.6, PyTorch 1.6.0, Torchvision 0.7.0 and OpenCV 3.4.0.12.

    
### Usage

 Run the model:
	
```shell
python demo_cam_cython.py
```

### Acknowledgments

Our code builds upon Intel [MiDaS](https://github.com/intel-isl/MiDaS)


