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
#include "basic_structs.h"
#include "sfm_incremental.h"

void main(void)
{
	objectsfm::IncrementalSfM incremental_sfm;

	std::string mode = "UAV"; // WEB, UAV 
	incremental_sfm.options_.input_fold  = "C:\\Users\\lu.2037\\Downloads\\geotagged-images\\image";
	incremental_sfm.options_.output_fold = "C:\\Users\\lu.2037\\Downloads\\geotagged-images\\result";
	incremental_sfm.options_.use_bow = true;
	incremental_sfm.options_.num_image_voc = 500;
	incremental_sfm.options_.size_image = 4000*3000;
	incremental_sfm.options_.th_sim = 0.01;
	incremental_sfm.options_.matching_graph_algorithm = "WordQuerying";  // FeatureDescriptor InvertedFile WordQuerying
	incremental_sfm.options_.sim_graph_type = "WordNumber";  // BoWDistance WordNumber
	incremental_sfm.options_.th_max_iteration_full_bundle = 100;
	incremental_sfm.options_.th_max_iteration_partial_bundle = 100;
	incremental_sfm.options_.minimizer_progress_to_stdout = true;
	if (mode == "WEB")
	{
		incremental_sfm.options_.resize = false;
		incremental_sfm.options_.feature_type = "CUDASIFT"; // VLSIFT CUDASIFT CUDAASIFT
		incremental_sfm.options_.use_same_camera = false;
		incremental_sfm.options_.th_max_new_add_pts = 1000;
		incremental_sfm.options_.th_mse_localization = 7.0;
		incremental_sfm.options_.th_mse_reprojection = 3.0;
		incremental_sfm.options_.th_mse_outliers = 1.0;
	}
	else
	{
		incremental_sfm.options_.matching_type = "all";
		incremental_sfm.options_.resize = false;
		incremental_sfm.options_.feature_type = "VLSIFT"; // VLSIFT CUDASIFT CUDAASIFT
		incremental_sfm.options_.use_same_camera = true;
		incremental_sfm.options_.th_max_new_add_pts = 20000;
		incremental_sfm.options_.th_mse_localization = 7.0;
		incremental_sfm.options_.th_mse_reprojection = 7.0;
		incremental_sfm.options_.th_mse_outliers = 3.0;
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