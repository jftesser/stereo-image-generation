# DUPLICATE, ORIGINAL REPO WAS [HERE](https://github.com/m5823779/Stereo-Side-by-Side-Image-Generator-from-Single-Image/)
## Stereo (Side by Side) Image Generation from Single Image

Utilizes AI to instantly fully convert 2D content into stereo 3D
                       
![image](https://github.com/m5823779/Stereo-Side-by-Side-Image-Generator-from-Single-Image/blob/master/doc/Stereo_image_demo.gif)

Fig) Input image(Left) / Output(Middle) / 3D effect(Right)

### Introduce

With the rapid of autostereoscopic 3D monitors, through the specialized optical lens and eye-tracking technology delivers users have an entirely new stereoscopic 3D visualization experience. However, it only works with 3D content inputs (Stereoscopic Images). Such as side-by-side images. But most image or video on the internet is 2D single view content. Making the technology difficult to popularize. In order to solve this problem, this project utilizes "Deep Learning" and "Computer Vision" to enable conversion of 2D content into stereo 3D content.

### Changelog

* [Aug 2020] Release [C++](https://github.com/m5823779/stereo_image_generator_from_single_image/tree/master/c%2B%2B) and [cython](https://github.com/m5823779/stereo_image_generator_from_single_image/tree/master/cython) version
* [Aug 2020] Initial release of stereo image generation base on MiDaS v2.0

### Setup 

1) Download the model weights [model-f45da743.pt](https://drive.google.com/file/d/1J6x7ea_lRd14A_dXD1lkHJujQDe7215e/view?usp=sharing) and place the
file in the root folder.

2) Set up dependencies: 

    ```shell
	pip install torch  
	pip install torchvision
	pip install opencv_python
    pip install tqdm
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


