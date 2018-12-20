// ObjectSfM - Object Based Structure-from-Motion.
// Copyright (C) 2018  Ohio State University, CEGE, GDA group
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "basic_structs.h"
#include "sfm_incremental.h"

#include <opencv2/opencv.hpp>

using namespace std;


int Video2Img(std::string file_video)
{
	std::string fold_out = "F:\\Database\\GoPro\\11-20\\1114_HERO7_data\\11142018_HERO_7_front_19GB\\GX010005\\";

	cv::VideoCapture capture;
	capture.open(file_video);
	if (!capture.isOpened())  // check if we succeeded
		return -1;

	int frame_width = (int)capture.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = (int)capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	float frame_fps = capture.get(CV_CAP_PROP_FPS);
	int frame_number = capture.get(CV_CAP_PROP_FRAME_COUNT); 
	cout << "frame_width is " << frame_width << endl;
	cout << "frame_height is " << frame_height << endl;
	cout << "frame_fps is " << frame_fps << endl;

	for (size_t i = 0; i < frame_number; i++)
	{
		cv::Mat frame;
		bool ok = capture.read(frame);
		if (!ok) {
			cout << "error" << endl;
			break;
		}
		std::string file_i = fold_out + std::string(10000 - std::to_string(i).length(), '0') + std::to_string(i) + ".jpg";
		cv::imwrite(file_i, frame);
	}

	return 1;
}

void main(void)
{
	std::string file_video = "F:\\Database\\GoPro\\11-20\\1114_HERO7_data\\11142018_HERO_7_front_19GB\\GX010005.MP4";
	Video2Img(file_video);


	objectsfm::IncrementalSfM incremental_sfm;

	std::string mode = "WEB"; // WEB, UAV 
	incremental_sfm.options_.input_fold  = "F:\\DevelopCenter\\3DReconstruction\\3DModelingProject.git\\trunk\\data_sfm\\15\\image";
	incremental_sfm.options_.output_fold = "F:\\DevelopCenter\\3DReconstruction\\3DModelingProject.git\\trunk\\data_sfm\\15\\result";
	incremental_sfm.options_.use_bow = true;
	incremental_sfm.options_.num_image_voc = 1000;
	incremental_sfm.options_.resize_image = 1600*1200;
	incremental_sfm.options_.th_sim = 0.01;
	incremental_sfm.options_.matching_graph_algorithm = "WordQuerying";  // FeatureDescriptor InvertedFile WordQuerying
	incremental_sfm.options_.sim_graph_type = "WordNumber";  // BoWDistance WordNumber
	incremental_sfm.options_.th_max_iteration_full_bundle = 100;
	incremental_sfm.options_.th_max_iteration_partial_bundle = 100;
	incremental_sfm.options_.minimizer_progress_to_stdout = true;
	if (mode == "WEB")
	{
		incremental_sfm.options_.all_match = false;
		incremental_sfm.options_.feature_type = "CUDASIFT"; // VLSIFT CUDASIFT CUDAASIFT
		incremental_sfm.options_.use_same_camera = false;
		incremental_sfm.options_.th_max_new_add_pts = 300;
		incremental_sfm.options_.th_mse_localization = 5.0;
		incremental_sfm.options_.th_mse_reprojection = 3.0;
		incremental_sfm.options_.th_mse_outliers = 1.0;
	}
	else
	{
		incremental_sfm.options_.all_match = true;
		incremental_sfm.options_.feature_type = "VLSIFT"; // VLSIFT CUDASIFT CUDAASIFT
		incremental_sfm.options_.use_same_camera = true;
		incremental_sfm.options_.th_max_new_add_pts = 20000;
		incremental_sfm.options_.th_mse_localization = 7.0;
		incremental_sfm.options_.th_mse_reprojection = 7.0;
		incremental_sfm.options_.th_mse_outliers = 5.0;
	}

	// step1: initialize the system, which includes:
	// (1) feature extration (2) bow generation (3) bow similarity calculation
	// (4) feature matching  (5) graph generation
	incremental_sfm.InitializeSystem();

	// step2: run the incremental sfm pipeline, which includes:
	// (1) find seed pair and negerate the initial camera and 3D points
	// (2) incrementally add new images into the model
	// (3) partial bundle adjustment and full bundle adjustment
	incremental_sfm.Run();
}