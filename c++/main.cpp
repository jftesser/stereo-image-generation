#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <string>
#include <ctime>
#include<cuda.h>
#include<cmath>


using namespace std;
using namespace cv;


int main(int argc, const char* argv[]) {
	// Setting device
	torch::Device device = torch::kCPU;
	if (torch::cuda::is_available()) {
		cout << "CUDA is available!" << endl;
		device = torch::kCUDA;
	}
	else {
		cout << "CUDA is unavailable!" << endl;
	}

	// Cam setting
	cv::VideoCapture cap(0, cv::CAP_DSHOW);

	int input_h = 1080;
	int input_w = 1920;
	cap.set(CV_CAP_PROP_FRAME_WIDTH, input_w);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, input_h);
	cout << "Camera Resolution: " << input_w << "x" << input_h << endl;

	// Loading model
	string model_path = "model.pt";
	torch::NoGradGuard no_grad;
	torch::jit::script::Module module;
	module = torch::jit::load(model_path);
	module.to(device);
	cout << "Success loading model..." << endl;

	Mat left_img, input, depth_map;
	Mat right_img(input_h, input_w, CV_8UC3);
	Mat anaglyph(input_h, input_w, CV_8UC3);

	float IPD = 6.25;
	float max_val = (pow(2, 16)) - 1;
	float minVal, maxVal;
	float deviation_cm = IPD * 0.12;
	float deviation = deviation_cm * 38.5 * (float)input_w / 1920.0;
	double duration;
	cout << "IPD: " << IPD << "cm" << endl;
	
	clock_t start, finish;

	while (1)
	{
		// ------------------------------------------------------------------------------------------- //
		
		
		start = clock();

		// Read image
		cap.read(left_img);

		// Image pre processing
		cvtColor(left_img, input, cv::COLOR_BGR2RGB);
		input.convertTo(input, CV_32F, 1.0 / 255);
		resize(input, input, cv::Size(384, 224), INTER_CUBIC);

		int h = input.rows;
		int w = input.cols;

		// To tensor
		auto img_tensor = torch::from_blob(input.data, { 1, h, w, 3 }).to(device);
		img_tensor = img_tensor.permute({ 0, 3, 1, 2 });

		// Normalize
		img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
		img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
		img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);
		auto img_var = torch::autograd::make_variable(img_tensor, false);


		vector<torch::jit::IValue> inputs;
		inputs.emplace_back(img_var);

		// Depth estimation
		at::Tensor output = module.forward(inputs).toTensor();

		namespace F = torch::nn::functional;
		output = F::interpolate(output.unsqueeze(1), F::InterpolateFuncOptions().size(vector<int64_t>({ input_h, input_w })).align_corners(false).mode(torch::kBicubic));

		output = output.squeeze().to(torch::kCPU);
		maxVal = output.max().item<float>();
		minVal = output.min().item<float>();
		
		Mat tempMat(output.size(0), output.size(1), CV_32FC1, (void*)output.data_ptr<float_t>());
		tempMat = (tempMat - minVal) / (maxVal - minVal);
		tempMat = 1 - tempMat;
		
		/*tempMat = max_val * (tempMat - minVal) / (maxVal - minVal);
		tempMat = max_val - tempMat;

		tempMat.convertTo(depth_map, CV_16UC1);*/


		// ------------------------------------------------------------------------------------------- //


		// Shifting pixel
		right_img = false;

		for (int row = 0; row < left_img.rows; row++)
		{
			for (int col = 0; col < left_img.cols; col++)
			{
				int dis = (int)((1.0 - tempMat.ptr<float>(row)[col]) * deviation);
				int col_r = col - dis;
				if (col_r > 0)
				{
					right_img.ptr<Vec3b>(row)[col_r] = left_img.ptr<Vec3b>(row)[col];
				}
			}
		}


		// ------------------------------------------------------------------------------------------- //


		// Image inpainting
		
		vector<int> zero_rows;
		vector<int> zero_cols;

		for (int row = 0; row < left_img.rows; row++)
		{
			for (int col = 0; col < left_img.cols; col++)
			{	
				if (right_img.ptr<Vec3b>(row)[col] == Vec3b(0, 0, 0))
				{
					zero_rows.push_back(row);
					zero_cols.push_back(col);
				}
			}
		}
		
		
		// ------------------------------------------------------------------------------------------- //
		

		vector<int>::iterator r_begin = zero_rows.begin();
		vector<int>::iterator r_end = zero_rows.end();
		vector<int>::iterator r;

		vector<int>::iterator c_begin = zero_cols.begin();
		vector<int>::iterator c_end = zero_cols.end();
		vector<int>::iterator c;
		
		for (r = r_begin, c = c_begin; r != r_end, c != c_end; r++, c++) 
		{	
			for (int offset = 1; offset < (int)deviation; offset++)
			{		
				int r_offset = *c + offset;
				int l_offset = *c - offset;
				
				if (r_offset <= right_img.cols && right_img.ptr<Vec3b>(*r)[r_offset] != Vec3b(0, 0, 0))
				{	
					right_img.ptr<Vec3b>(*r)[*c] = right_img.ptr<Vec3b>(*r)[r_offset];
					break;
				}
				if (l_offset >= 0  && right_img.ptr<Vec3b>(*r)[l_offset] != Vec3b(0, 0, 0))
				{
					right_img.ptr<Vec3b>(*r)[*c] = right_img.ptr<Vec3b>(*r)[l_offset];
					break;
				}
			}
		}


		// ------------------------------------------------------------------------------------------- //


		// Overlap
		anaglyph = 0;
		for (int row = 0; row < right_img.rows; row++)
		{
			for (int col = 0; col < right_img.cols; col++)
			{
				anaglyph.ptr<Vec3b>(row)[col][2] = left_img.ptr<Vec3b>(row)[col][2];
			}
		}
		for (int row = 0; row < right_img.rows; row++)
		{
			for (int col = 0; col < right_img.cols; col++)
			{
				anaglyph.ptr<Vec3b>(row)[col][1] = right_img.ptr<Vec3b>(row)[col][1];
				anaglyph.ptr<Vec3b>(row)[col][0] = right_img.ptr<Vec3b>(row)[col][0];
			}
		}

		imshow("Right Image", anaglyph);

		finish = clock();
		duration = (double)(finish - start) / CLOCKS_PER_SEC;
		cout << "Framerate: " << 1.0 / duration << " \tfps" << "\r";
		
		if (waitKey(1) >= 0)
			break;

	}
}
