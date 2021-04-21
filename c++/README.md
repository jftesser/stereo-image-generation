## Stereo (Side by Side) Image Generation by using LibTorch in C++

### Setup 

1) Download the model weights [model.pt](https://drive.google.com/file/d/1G3XQuxpSn5Mh0si5GD6N_hG3cy0n4J3V/view?usp=sharing) and place the
file in the root folder.

2) Install Opencv and libtorch to C:\

3) Build
	
	3.1 Go to root folder
		```
		mkdir build && cd build
		```
	
	3.2 cmake
	
	>For Release：
		```
		cmake -DCMAKE_PREFIX_PATH=path_to_opencv\build\x64\vc15\lib;path_to_lib_torch -DCMAKE_BUILD_TYPE=Release -G"Visual Studio 15 Win64" ..
		```
	
	>For Debug：
		```
		cmake -DCMAKE_PREFIX_PATH=path_to_opencv\build\x64\vc15\lib;path_to_lib_torch -DCMAKE_BUILD_TYPE=Debug -G"Visual Studio 15 Win64" ..
		```
	
	>Example：
		```
		cmake -DCMAKE_PREFIX_PATH=C:\opencv\build\x64\vc15\lib;C:\libtorch -DCMAKE_BUILD_TYPE=Release -G"Visual Studio 15 Win64" ..
		```

	(Be Sure if you want to cmake again. Please go to `\build` and delete `CMakeCache.txt)
	
	
	3.3 Generate exe file or put `dll` file to `\build\Release` or `\build\Debug`
	
	>For Release：
		```
		cmake --build . --config Release
		```
	
	>For Debug：
		```
		cmake --build . --config Debug
		```
	
	3.4 Edit opencv to system enviroment variable
	

3) Convert model (Optional)
	```
    python convert.py
	```
	and put converted model to `\build\Release` or `\build\Debug`
	
	(or download [model.pt](https://drive.google.com/file/d/1G3XQuxpSn5Mh0si5GD6N_hG3cy0n4J3V/view?usp=sharing) to `\build\Release` )
    
### Usage

Run the model:
   
   
   Go to `\build\Release`  or `\build\Debug`

   ```shell
   ./Midas.exe
   ```


### Acknowledgments

Our code builds upon Intel [MiDaS](https://github.com/intel-isl/MiDaS)


