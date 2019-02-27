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

#ifndef OBJECTSFM_BASIC_STRUCTS_H_
#define OBJECTSFM_BASIC_STRUCTS_H_

#ifndef MAX_
#define MAX_(a,b) ( ((a)>(b)) ? (a):(b) )
#endif // !MAX_

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN_

#include <vector>
#include <string>
#include <map>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "utils/nanoflann_utils.h"
#include "utils/nanoflann.hpp"

namespace objectsfm 
{
	struct ImageInfo
	{
		int rows, cols;
		float zoom_ratio = 1.0;
		float f_mm = 0.0, f_pixel = 0.0;
		float gps_latitude, gps_longitude, gps_attitude;
		std::string cam_maker, cam_model;
	};

	struct CameraModel
	{
	public:
		CameraModel() {};
		CameraModel(int id, int h, int w, double f_mm, double f, std::string cam_maker, std::string cam_model )
		{
			f_mm_ = f_mm;
			f_ = f;
			f_hyp_ = MAX_(w, h)*1.2;
			w_ = w, h_ = h;
			px_ = w_ / 2.0, py_ = h_ / 2.0;
			k1_ = 0.0, k2_ = 0.0, dcx_ = 0.0, dcy_ = 0.0;
			id_ = id;
			cam_maker_ = cam_maker;
			cam_model_ = cam_model;
			num_cams_ = 0;
			is_mutable_ = true;

			UpdateDataFromModel();
		}

		void SetIntrisicParas(double f, double px, double py)
		{
			f_ = f;
			px_ = px;
			py_ = py;
			UpdateDataFromModel();
		}

		void SetFocalLength(double f)
		{
			f_ = f;
			UpdateDataFromModel();
		}

		void UpdateDataFromModel()
		{
			data[0] = f_;
			data[1] = k1_;
			data[2] = k2_;
			data[3] = dcx_;
			data[4] = dcy_;
		}

		void UpdataModelFromData()
		{
			f_ = data[0];
			k1_ = data[1];
			k2_ = data[2];
			dcx_ = data[3];
			dcy_ = data[4];

			px_ += dcx_;
			py_ += dcy_;
		}

		void AddCamera(int idx)
		{
			idx_cams_.push_back(idx);
			num_cams_++;
		}

		void SetImmutable()
		{
			is_mutable_ = false;
		}

		int id_;
		std::string cam_maker_, cam_model_;
		int w_, h_;       // cols and rows
		double f_mm_, f_, f_hyp_, px_, py_;  // focal length and principle point
		double k1_, k2_, dcx_, dcy_;   // distrotion paremeters
		double data[5];    // for bundle adjustment, {f_, k1_, k2_, dcx_, dcy_}, respectively
		int num_cams_;
		std::vector<int> idx_cams_;
		bool is_mutable_;
	};

	struct RTPoseRelative
	{
		Eigen::Matrix3d R;
		Eigen::Vector3d t;
	};

	// [Xc 1] = [R|t]*[Xw 1], converts a 3D point in the world-3d coordinate system
	// into the camera-3d coordinate system
	struct RTPose
	{
		Eigen::Matrix3d R;
		Eigen::Vector3d t;
	};

	// Another form of camera pose designed for bundle adjustment
	struct ACPose
	{
		Eigen::Vector3d a;  // R = Rodrigues(a)
		Eigen::Vector3d c;  // Xc = R*(Xw-c) since Xc = R*Xw+t, we have t = -R*c and c = -R'*t
	};

	struct IncrementalSfMOptions
	{
		// image reading
		std::string input_fold;
		std::string output_fold;

		// data base
		int num_image_voc = 500;
		int size_image = 2000*1500;
		std::string feature_type = "VLSIFT"; // VLSIFT CUDASIFT CUDAASIFT
		bool resize = false;

		// graph
		std::string matching_graph_algorithm = "FeatureDescriptor"; // BoWSimilarity InvertedFile WordMatching
		std::string sim_graph_type = "WordNumber";  // BoWDistance WordNumber
		bool use_bow = false;
		bool use_gpu = false;
		float th_sim = 0.01;

		// 
		bool use_same_camera = false;

		// id_pt_global = id_pt_image + id_image * label_max_per_image
		// to generat the global index of the keypoint on each image
		int idx_max_per_image = 1000000;

		// localization
		int th_seedpair_structures = 20;
		int th_max_new_add_pts = 200;
		int th_max_failure_localization = 5;
		int th_min_2d3d_corres = 20;

		// bundle
		bool minimizer_progress_to_stdout = false;
		int th_max_iteration_full_bundle = 100;
		int th_max_iteration_partial_bundle = 100;
		int th_step_full_bundle_adjustment = 5;

		// outliers
		double th_mse_localization = 5.0;
		double th_mse_reprojection = 3.0;
		double th_mse_outliers = 1.0;

		double th_angle_small = 3.0 / 180.0*3.1415;
		double th_angle_large = 5.0 / 180.0*3.1415;

		// matching graph
		std::string matching_type = "all";
		std::string priori_type = "llt";
		std::string priori_file;
	};

	//
	struct DatabaseOptions
	{
		// voc
		int num_image_voc = 500;
		int fbow_k = 10;  // number of children for each node
		int fbow_l = 6;     // number of levels of the voc-tree

		bool resize = false;
		float size_image = 2000*1500;
		std::string feature_type = "VLSIFT"; // VLSIFT CUDASIFT CUDAASIFT
		bool extract_gist = false;
	};

	//
	struct GraphOptions
	{
		std::string matching_graph_algorithm = "BoWSimilarity"; // BoWSimilarity InvertedFile WordExpending
		bool use_bow = false;
		bool use_gpu = false;
		float th_sim = 0.01;
		std::string sim_graph_type = "WordNumber";  // BoWDistance WordNumber
		std::string matching_type = "feature"; // feature  all
		std::string priori_type = "llt"; // llt xyz xy order
		std::string priori_file;
		int ellipsoid_id_ = 23; // WGS-84
		std::string zone_id_ = "17N";
		int knn = 50;
	};

	// 
	struct BundleAdjustOptions
	{
		int max_num_iterations = 200;
		bool minimizer_progress_to_stdout = true;
		int num_threads = 1;
	};

	//
	struct DenseOptions
	{
		int disp_size = 128;
		float uniqueness = 0.96;
	};

	//
	struct ListKeyPoint
	{
		std::vector<cv::KeyPoint> pts;
	};

	struct ListFloat
	{
		std::vector<float> data;
	};

	struct QuaryResult
	{
		int ids[10];
	};

	typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, SiftList<float> >, SiftList<float>, 128/*dim*/ > my_kd_tree_t;

}
#endif //OBJECTSFM_BASIC_STRUCTS_H_