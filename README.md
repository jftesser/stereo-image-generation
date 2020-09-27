## Stereo (Side by Side) Image Generator from Single Image

This repository contains code to generate stereo (Side by side) image from a single image.
                       
![image](https://github.com/m5823779/Stereo-Side-by-Side-Image-Generator-from-Single-Image/blob/master/doc/Stereo_image_demo.gif)

Fig) Input image(Left) / Output(Middle) / 3D effect(Right)

### Introduce

Have you ever wondered why people and many animals have two eyes? By using two eyes we can have a perception of depth. If you could have two pictures of your both eyes and you try to overlap these pictures you will see that there are big displacements in the nearest objects to your eyes while the displacements in the objects that are far away will be very small. These observed displacements are translated to the depth of the objects. 
	
	
So far, many deep learning algorithms have made a huge success in monocular depth estimation. Using a convolution neural network, we can predict the depth value of each pixel, given only a single RGB image as input. According to the previous introduction, for pixels with farther depth, we make a larger offset for the pixels corresponding to the original image. for pixels with closer depth, we make a smaller offset. Then using some image inpainting algorithms. So that we can get an image from different perspectives. Combine with the original input we can get a stereo image. Finally, through a special display, we can let this stereo (side by side) image produce a 3D effect.
	
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

1) Place input images in the folder `example`.

2) Run the model:
   
   (Generate depth map from image)

    ```shell
    python depth_estimate_image.py
    ```
	
	(Generate depth map from camera)
	
	```shell
    python depth_estimate_cam.py
    ```
	
	(Generate stereo image from image)
	
	```shell
    python SBS_generate_image.py
    ```
	
	(Generate stereo image from camera)
	
	```shell
    python SBS_generate_cam.py
    ```

3) The resulting depth maps are written to the `depth` folder.

	The resulting stereo image are written to the `stereo` folder.




### Acknowledgments

Our code builds upon Intel [MiDaS](https://github.com/intel-isl/MiDaS)


