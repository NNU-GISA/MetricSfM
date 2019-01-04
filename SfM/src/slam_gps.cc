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

#ifndef MAX_
#define MAX_(a,b) ( ((a)>(b)) ? (a):(b) )
#endif // !MAX

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN_

#include "slam_gps.h"

#include <fstream>
#include <filesystem>
#include <iomanip>

#include <Eigen/Core>
#include "ceres/ceres.h"
#include <opencv2/opencv.hpp>

#include "utils/basic_funcs.h"
#include "utils/converter_utm_latlon.h"
#include "utils/ellipsoid_utm_info.h"
#include "utils/reprojection_error_pose_cam_xyz.h"
#include "utils/reprojection_error_pose_xyz.h"
#include "utils/gps_error_pose.h"
#include "utils/geo_verification.h"

#include "flann/flann.h"

namespace objectsfm {

	SLAMGPS::SLAMGPS()
	{
		rows = 1500;
		cols = 2000;
		k1 = 0.0;
		k2 = 0.0;
		p1 = 0.0;
		p2 = 0.0;
		k3 = 0.0;
	}

	SLAMGPS::~SLAMGPS()
	{
	}

	void SLAMGPS::Run(std::string fold)
	{
		// read in slam information
		std::string file_slam = fold + "\\slam.txt";
		ReadinSLAM(file_slam);

		// read in gps
		std::string file_gps = fold + "\\gps.txt";
		std::map<int, cv::Point2d> gps_info;
		ReadinGPS(file_gps, gps_info);

		// accosiation
		std::string file_rgb = fold + "\\rgb.txt";
		AssociateCameraGPS(file_rgb, gps_info);

		//std::vector<int> img_ids = { 1,2,3,4,5,6,7,8,9 };
		//DrawPts(img_ids, fold + "\\undistort_image");

		GrawGPS(fold + "\\gps_pos.bmp");
		GrawSLAM(fold + "\\slam_pos.bmp");

		// feature extraction
		FeatureExtraction(fold);

		// feature matching with pose information
		if (0) {
			FeatureMatching(fold);
		}
		std::cout << 1 << std::endl;
		Triangulation(fold);

		// do adjustment
		//FullBundleAdjustment();
		std::cout << 2 << std::endl;
		std::string file_accu = fold + "\\accuracy.txt";
		GetAccuracy(file_accu, cam_models_, cams_, pts_);

		// save
		SaveForSure(fold);

		SaveforOpenMVS(fold);

		SaveforCMVS(fold);

		// write out
		WriteCameraPointsOut(fold + "\\slam_cam_pts.txt");
		GrawSLAM(fold + "\\slam_pos2.bmp");
	}

	void SLAMGPS::ReadinSLAM(std::string file_slam)
	{
		std::ifstream ff(file_slam);

		// cameras
		int n_frames = 0;
		ff >> n_frames;
		cams_.resize(n_frames);
		int id;
		double tt;
		Eigen::Vector3d t;
		Eigen::Matrix3d R;
		std::map<int, int> cam_index;
		for (size_t i = 0; i < n_frames; i++)
		{
			cams_[i] = new Camera();
			ff >> id >> tt >> fx >> fy >> cx >> cy;
			for (size_t m = 0; m < 3; m++) {
				ff >> t(m);
			}
			for (size_t m = 0; m < 3; m++) {
				for (size_t n = 0; n < 3; n++)
				{
					ff >> R(m,n);
				}
			}
			cams_[i]->SetRTPose(R, t);
			cams_[i]->SetID(id);

			cam_index.insert(std::pair<int, int>(id, i));
		}

		// points
		int n_pts = 0;
		ff >> n_pts;
		pts_.resize(n_pts);
		double x, y, z;
		for (size_t i = 0; i < n_pts; i++)
		{
			pts_[i] = new Point3D();
 			ff >> x >> y >> z;
			pts_[i]->data[0] = x;
			pts_[i]->data[1] = y;
			pts_[i]->data[2] = z;

			int n_obs = 0;
			ff >> n_obs;
			int id_cam;
			double px, py;
			for (size_t j = 0; j < n_obs; j++)
			{
				ff >> id_cam >> px >> py;

				std::map<int, int >::iterator iter_i_c = cam_index.find(id_cam);
				if (iter_i_c == cam_index.end())
				{
					continue;
				}
				int idx_cam = iter_i_c->second;

				pts_[i]->AddObservation(cams_[idx_cam], px - cx, py - cy, j);
			}
		}

		// camera models
		cam_models_.resize(1);
		cam_models_[0] = new CameraModel(cam_models_.size(), rows, cols, (fx+fy)/2.0, 0.0, "lu", "lu");
		cam_models_[0]->SetIntrisicParas((fx + fy) / 2.0, cx, cy);
		for (size_t i = 0; i < cams_.size(); i++)
		{
			cams_[i]->AssociateCamereModel(cam_models_[0]);
		}
	}

	void SLAMGPS::ReadinGPS(std::string file_gps, std::map<int, cv::Point2d> &gps_info )
	{
		int ellipsoid_id = 23; // WGS-84
		std::string zone_id = "17N";

		std::ifstream ff(file_gps);
		int id;
		double lat, lon, alt;
		double x, y;
		while (!ff.eof())
		{
			ff >> id >> lat >> lon >> alt;
			LLtoUTM(ellipsoid_id, lat, lon, y, x, (char*)zone_id.c_str());
			gps_info.insert(std::pair<int, cv::Point2d>(id, cv::Point2d(x, y)));
		}
		ff.close();
	}

	void SLAMGPS::AssociateCameraGPS(std::string file_rgb, std::map<int, cv::Point2d>& gps_info)
	{
		std::ifstream ff(file_rgb);
		std::string s0;
		std::getline(ff, s0);
		std::getline(ff, s0);
		std::getline(ff, s0);


		std::vector<cv::Point2d> gps_all_frame;
		std::vector<std::string> names_all_frame;
		while (!ff.eof())
		{
			std::string s;
			std::getline(ff, s);
			size_t loc1 = s.find_last_of('/');
			size_t loc2 = s.find_last_of('.');
			std::string name = s.substr(loc1 + 1, loc2 - loc1 - 1);
			int id = std::stoi(name);

			std::map<int, cv::Point2d>::iterator iter = gps_info.find(id);
			if (iter == gps_info.end())
			{
				continue;
			}
			gps_all_frame.push_back(iter->second);
			names_all_frame.push_back(name);
		}

		// gps of keyframes
		cams_gps_.resize(cams_.size());
		cams_name_.resize(cams_.size());
		for (size_t i = 0; i < cams_.size(); i++)
		{
			int id_frame = cams_[i]->id_;
			cams_gps_[i] = gps_all_frame[id_frame];
			cams_name_[i] = names_all_frame[id_frame];
		}
	}

	void SLAMGPS::FeatureExtraction(std::string fold)
	{
		std::string fold_image = fold + "\\rgb";
		std::string fold_feature = fold + "\\feature";
		if (!std::experimental::filesystem::exists(fold_feature)) {
			std::experimental::filesystem::create_directory(fold_feature);
		}

		// feature extraction
		std::vector<std::string> image_paths(cams_name_.size());
		for (size_t i = 0; i < cams_name_.size(); i++)
		{
			image_paths[i] = fold_image + "\\" + cams_name_[i] + ".jpg";
		}
		db_.output_fold_ = fold_feature;
		db_.options.resize = false;
		db_.options.feature_type = options_.feature_type;
		if (!db_.FeatureExtraction(image_paths)) {
			std::cout << "Error with the database" << std::endl;
		}
	}

	void SLAMGPS::FeatureMatching(std::string fold)
	{
		int win_size = 10;
		int th_same_pts = 20;
		float th_epipolar = 2.0;
		float th_distance = 5.0;
		float th_ratio_f = 0.75;
		float th_h_f_ratio = 0.90;
		float th_second_first_ratio = 0.80;
		std::string fold_feature = fold + "\\feature";

		// step1: matching graph, priori F and H
		std::map<int, int> cams_info;
		for (size_t i = 0; i < cams_.size(); i++) {
			cams_info.insert(std::pair<int, int>(cams_[i]->id_, i));
		}

		std::vector<std::vector<int>> cam_pts(cams_.size());
		for (size_t i = 0; i < pts_.size(); i++)
		{
			std::map<int, Camera*>::iterator iter_cams = pts_[i]->cams_.begin();
			while (iter_cams != pts_[i]->cams_.end())
			{
				int id_cam = iter_cams->second->id_;
				std::map<int, int >::iterator iter = cams_info.find(id_cam);
				cam_pts[iter->second].push_back(i);
				iter_cams++;
			}
		}

		std::vector<std::vector<int>> ids(cams_.size());
		std::vector<std::vector<cv::Mat>> Fs(cams_.size());
		std::vector<std::vector<cv::Mat>> Hs(cams_.size());
		std::string file_prior = fold + "\\feature\\prior.txt";
		if (!std::experimental::filesystem::exists(file_prior))
		{
			for (int i = 0; i < cams_.size(); i++)
			{
				std::cout << "image " << i << std::endl;
				int id_s = MAX_(i - win_size, 0);
				int id_e = MIN_(i + win_size, cams_.size());
				for (int j = id_s; j < id_e; j++)
				{
					if (j == i) {
						continue;
					}

					// query the same pts
					std::vector<int> same_pts;
					math::same_in_vectors(cam_pts[i], cam_pts[j], same_pts);
					if (same_pts.size() < th_same_pts) {
						continue;
					}
					// 
					std::vector<cv::Point2f> pts1, pts2;
					for (int m = 0; m < same_pts.size(); m++)
					{
						int id_pt = same_pts[m];
						auto it1 = pts_[id_pt]->pts2d_.begin();
						auto it2 = pts_[id_pt]->cams_.begin();
						while (it2 != pts_[id_pt]->cams_.end())
						{
							int id_cam = it2->second->id_;
							std::map<int, int >::iterator iter = cams_info.find(id_cam);
							if (iter->second == i) {
								pts1.push_back(cv::Point2f(it1->second(0), it1->second(1)));
							}
							if (iter->second == j) {
								pts2.push_back(cv::Point2f(it1->second(0), it1->second(1)));
							}
							it1++;  it2++;
						}
					}

					// F matrix
					std::vector<uchar> status_f(pts1.size());
					cv::Mat F = cv::findFundamentalMat(pts1, pts2, status_f, cv::FM_RANSAC, th_epipolar);
					int count_inlier_f = 0;
					for (int m = 0; m < status_f.size(); m++) {
						if (status_f[m]) {
							count_inlier_f++;
						}
					}
					if (count_inlier_f < pts1.size()*th_ratio_f) {
						continue;
					}

					// H matrix
					std::vector<uchar> status_h(pts1.size());
					cv::Mat H = cv::findHomography(pts1, pts2, status_h, cv::FM_RANSAC, th_distance);
					int count_inlier_h = 0;
					for (int m = 0; m < status_h.size(); m++) {
						if (status_h[m]) {
							count_inlier_h++;
						}
					}
					if (count_inlier_h > count_inlier_f * th_h_f_ratio) {
						continue;
					}

					ids[i].push_back(j);
					Fs[i].push_back(F);
					Hs[i].push_back(H);
				}
			}
			WriteOutPriorInfo(file_prior, ids, Fs, Hs);
		}
		else
		{
			ReadinPriorInfo(file_prior, ids, Fs, Hs);
		}

		// step2: matching with priori F and H
		std::vector<std::vector<int>> match_graph;
		match_graph.resize(cams_.size());
		for (size_t i = 0; i < cams_.size(); i++) {
			match_graph[i].resize(cams_.size());
		}

		for (int i = 0; i < cams_.size(); i++)
		{
			int id1 = i;
			db_.ReadinImageFeatures(id1);

			// generate kd-tree for idx1
			struct FLANNParameters p;
			p = DEFAULT_FLANN_PARAMETERS;
			p.algorithm = FLANN_INDEX_KDTREE;
			p.trees = 8;
			p.log_level = FLANN_LOG_INFO;
			p.checks = 64;
			float speedup;

			float* data_idx1 = (float*)db_.descriptors_[id1]->data;
			flann_index_t kdtree_idx1 = flann_build_index(data_idx1, db_.descriptors_[id1]->rows, db_.descriptors_[id1]->cols, &speedup, &p);

			// matching
			std::vector<std::vector<std::pair<int, int>>> matches(ids[i].size());
			for (int j = 0; j < ids[i].size(); j++)
			{
				int id2 = ids[i][j];
				db_.ReadinImageFeatures(id2);

				std::cout << "  -------" << id1 << " " << id2 << " " << db_.descriptors_[id1]->rows
					<< " " << db_.descriptors_[id2]->rows << std::endl;

				// do matching
				int count = db_.keypoints_[id2]->pts.size();
				int* knn_id = new int[count * 2];
				float* knn_dis = new float[count * 2];
				flann_find_nearest_neighbors_index(kdtree_idx1, (float*)db_.descriptors_[id2]->data, count, knn_id, knn_dis, 2, &p);

				int* ptr_id = knn_id;
				float* ptr_dis = knn_dis;
				int count1 = 0, count2 = 0, count3 = 0;
				std::vector<cv::Point2f> pts1_init, pts2_init;
				for (int m = 0; m < count; m++)
				{
					float ratio = ptr_dis[2 * m + 0] / ptr_dis[2 * m + 1];

					// check1: second first ratio
					if (ratio > th_second_first_ratio) {
						count1++;
						continue;
					}

					// check2: fundamental
					cv::Mat pt1 = (cv::Mat_<double>(3, 1) << db_.keypoints_[id1]->pts[ptr_id[2 * m + 0]].pt.x,
						db_.keypoints_[id1]->pts[ptr_id[2 * m + 0]].pt.y,
						1);
					cv::Mat pt2 = (cv::Mat_<double>(3, 1) << db_.keypoints_[id2]->pts[m].pt.x,
						db_.keypoints_[id2]->pts[m].pt.y,
						1);
					cv::Mat l2 = Fs[i][j] * pt1;
					double epi_dis = abs(l2.dot(pt2)) / sqrt(pow(l2.at<double>(0, 0), 2) + pow(l2.at<double>(1, 0), 2));
					if (epi_dis > th_epipolar) {
						count2++;
						continue;
					}

					// check3: homography
					cv::Mat pt22 = Hs[i][j] * pt1;
					pt22 *= 1.0 / pt22.at<double>(2, 0);
					double dx = pt2.at<double>(0, 0) - pt22.at<double>(0, 0);
					double dy = pt2.at<double>(1, 0) - pt22.at<double>(1, 0);
					double homo_dis = sqrt(dx*dx + dy * dy);
					if (homo_dis > 20 * th_distance) {
						count3++;
						continue;
					}

					pts1_init.push_back(db_.keypoints_[id1]->pts[ptr_id[2 * m + 0]].pt);
					pts2_init.push_back(db_.keypoints_[id2]->pts[m].pt);
					matches[j].push_back(std::pair<int, int>(ptr_id[2 * m + 0], m));
				}
				//std::cout << count << " " << count1 << " " << count2 << " " << count3 << " " << matches[j].size() << " ";

				// do geo-verification
				std::vector<int> inliers;
				bool isOK = GeoVerification::GeoVerificationFundamental(pts1_init, pts2_init, inliers);
				std::vector<std::pair<int, int>> match_inliers;
				for (size_t m = 0; m < inliers.size(); m++) {
					match_inliers.push_back(matches[j][inliers[m]]);
				}
				matches[j] = match_inliers;
				std::cout << matches[j].size() << std::endl;

				// draw
				if (0)
				{
					cv::Mat image1 = cv::imread(db_.image_paths_[id1]);
					cv::Mat image2 = cv::imread(db_.image_paths_[id2]);
					for (size_t m = 0; m < matches[j].size(); m++)
					{
						int id_pt1_local = matches[j][m].first;
						int id_pt2_local = matches[j][m].second;
						cv::Point2f offset1(image1.cols / 2.0, image1.rows / 2.0);
						cv::Point2f offset2(image2.cols / 2.0, image2.rows / 2.0);
						cv::line(image1, db_.keypoints_[id1]->pts[id_pt1_local].pt + offset1,
							db_.keypoints_[id2]->pts[id_pt2_local].pt + offset2, cv::Scalar(0, 0, 255), 1);
					}
					std::string path = "F:\\" + std::to_string(id1 ) + "_" + std::to_string(id2) + "_match.jpg";
					cv::imwrite(path, image1);
				}

				//
				match_graph[id1][id2] = matches[j].size();
				WriteOutMatches(id1, id2, matches[j]);

				//
				db_.ReleaseImageFeatures(id2);
				db_.ReleaseImageKeyPoints(id2);
				delete[] knn_id;
				delete[] knn_dis;
			}
			flann_free_index(kdtree_idx1, &p);	
			db_.ReleaseImageFeatures(id1);
			db_.ReleaseImageKeyPoints(id1);
		}

		WriteOutMatchGraph(match_graph);
	}

	void SLAMGPS::Triangulation(std::string fold)
	{
		int n_img = cams_.size();

		graph_.AssociateDatabase(&db_);
		graph_.ReadinMatchingGraph();

		// data association
		std::map<int, int> pts_points_map;
		for (size_t i = 0; i < n_img; i++)
		{
			std::cout << i << " " << pts_new_.size() << std::endl;
			std::vector<int> ids;
			for (size_t j = 0; j < n_img; j++) {
				if (graph_.match_graph_[i*n_img + j] > 0) {
					ids.push_back(j);
				}
			}

			int id_img1 = i;
			db_.ReadinImageFeatures(id_img1);
			for (size_t j = 0; j < ids.size(); j++)
			{
				int id_img2 = ids[j];
				db_.ReadinImageFeatures(id_img2);
				std::vector<std::pair<int, int>> matches;
				graph_.QueryMatch(id_img1, id_img2, matches);

				for (size_t m = 0; m < matches.size(); m++)
				{
					int id_pt1_local = matches[m].first;
					int id_pt2_local = matches[m].second;

					int id_pt1_global = id_pt1_local + id_img1 * options_.idx_max_per_image;
					int id_pt2_global = id_pt2_local + id_img2 * options_.idx_max_per_image;
					std::map<int, int >::iterator iter1 = pts_points_map.find(id_pt1_global);
					std::map<int, int >::iterator iter2 = pts_points_map.find(id_pt2_global);
					if (iter1 != pts_points_map.end()) // add new obs to existing pt
					{
						int id_pt = iter1->second;
						pts_new_[id_pt]->AddObservation(cams_[id_img2],
							db_.keypoints_[id_img2]->pts[id_pt2_local].pt.x, 
							db_.keypoints_[id_img2]->pts[id_pt2_local].pt.y,
							id_img2);
						pts_points_map.insert(std::pair<int, int>(id_pt2_global, id_pt));
					}
					else if (iter2 != pts_points_map.end()) // add new obs to existing pt
					{
						int id_pt = iter2->second;
						pts_new_[id_pt]->AddObservation(cams_[id_img1],
							db_.keypoints_[id_img1]->pts[id_pt1_local].pt.x,
							db_.keypoints_[id_img1]->pts[id_pt1_local].pt.y,
							id_img1);
						pts_points_map.insert(std::pair<int, int>(id_pt1_global, id_pt));
					}
					else  // creat a new pt
					{
						Point3D *pt_temp = new Point3D;
						pt_temp->AddObservation(cams_[id_img1],
							db_.keypoints_[id_img1]->pts[id_pt1_local].pt.x,
							db_.keypoints_[id_img1]->pts[id_pt1_local].pt.y,
							id_img1);
						pt_temp->AddObservation(cams_[id_img2],
							db_.keypoints_[id_img2]->pts[id_pt2_local].pt.x,
							db_.keypoints_[id_img2]->pts[id_pt2_local].pt.y,
							id_img2);

						pts_new_.push_back(pt_temp);
						pts_points_map.insert(std::pair<int, int>(id_pt1_global, pts_new_.size() - 1));
						pts_points_map.insert(std::pair<int, int>(id_pt2_global, pts_new_.size() - 1));
					}
				}
				db_.ReleaseImageFeatures(id_img2);
				db_.ReleaseImageKeyPoints(id_img2);
			}

			db_.ReleaseImageFeatures(id_img1);
			db_.ReleaseImageKeyPoints(id_img1);
		}

		// triangulation
		double th_mse_reprojection = 3.0;
		double th_tri_angle = 5.0 / 180.0*CV_PI;
		int count_bad = 0;
		for (size_t i = 0; i < pts_new_.size(); i++)
		{
			bool is_ok = pts_new_[i]->Trianglate2(th_mse_reprojection, th_tri_angle);
			if (!is_ok || pts_new_[i]->cams_.size() < 3) {
				pts_new_[i]->is_bad_estimated_ = true;
				count_bad++;
			}
		}
		std::cout << "count_bad " << count_bad << " count_good " << pts_new_.size() - count_bad << std::endl;
		
		// accuracy analysis
		pts_ = pts_new_;
		std::string file_accu = fold + "\\accuracy.txt";
		GetAccuracy(file_accu, cam_models_, cams_, pts_);

		WriteCameraPointsOut(fold + "\\slam_cam_pts2.txt");
	}

	void SLAMGPS::FullBundleAdjustment()
	{
		ceres::Problem problem;
		ceres::Solver::Options options;
		ceres::Solver::Summary summary;

		options.max_num_iterations = 50;
		options.minimizer_progress_to_stdout = true;
		options.num_threads = 1;
		options.linear_solver_type = ceres::DENSE_SCHUR;

		// add reprojection error
		int count1 = 0;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			std::map<int, Camera*>::iterator iter_cams = pts_[i]->cams_.begin();
			std::map<int, Eigen::Vector2d>::iterator iter_pts = pts_[i]->pts2d_.begin();
			while (iter_cams != pts_[i]->cams_.end())
			{
				ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
				ceres::CostFunction* cost_function;

				//cost_function = ReprojectionErrorPoseCamXYZ::Create(iter_pts->second(0), iter_pts->second(1), pts_[i]->weight);
				//problem.AddResidualBlock(cost_function, loss_function, iter_cams->second->data, iter_cams->second->cam_model_->data, pts_[i]->data);

				cost_function = ReprojectionErrorPoseXYZ::Create(iter_pts->second(0), iter_pts->second(1), iter_cams->second->cam_model_->data, pts_[i]->weight);
				problem.AddResidualBlock(cost_function, loss_function, iter_cams->second->data, pts_[i]->data);

				iter_cams++;
				iter_pts++;

				count1++;
			}
		}

		// add gps-topology error
		int count2 = 0;
		if (1)
		{
			int step = 5;
			double weight = 20;
			for (int i = 30; i < cams_.size(); i++)
			{
				int id1 = i;
				int k = (i - 6) / 3;
				for (int j = 0; j < k; j++)
				{
					int id2 = j;
					int id3 = k + j;

					cv::Point2d v12 = cams_gps_[id2] - cams_gps_[id1];
					double l12 = sqrt(v12.x*v12.x + v12.y*v12.y);

					cv::Point2d v13 = cams_gps_[id3] - cams_gps_[id1];
					double l13 = sqrt(v13.x*v13.x + v13.y*v13.y);

					cv::Point2d v23 = cams_gps_[id3] - cams_gps_[id2];
					double l23 = sqrt(v23.x*v23.x + v23.y*v23.y);

					if (l12 == 0 || l13 == 0 || l23 == 0)
					{
						continue;
					}

					double angle1 = acos((v12.x*v13.x + v12.y*v13.y) / l12 / l13);
					double angle2 = acos((-v12.x*v23.x - v12.y*v23.y) / l12 / l23);
					double angle3 = acos((v13.x*v23.x + v13.y*v23.y) / l13 / l23);

					ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
					ceres::CostFunction* cost_function;
					cost_function = GPSErrorPose::Create(angle1, angle2, angle3, weight);
					problem.AddResidualBlock(cost_function, loss_function, cams_[id1]->data, cams_[id2]->data, cams_[id3]->data);

					count2++;
				}
			}
		}
		std::cout << "count1 " << count1 << " count2 " << count2 << std::endl;

		std::cout << cams_[0]->cam_model_->f_ << " ";
		std::cout << cams_[0]->cam_model_->k1_ << " ";
		std::cout << cams_[0]->cam_model_->k2_ << std::endl;

		ceres::Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << "\n";

		// update parameters
		for (size_t i = 0; i < cams_.size(); i++)
		{
			cams_[i]->UpdatePoseFromData();
		}
		for (size_t i = 0; i < cam_models_.size(); i++)
		{
			cam_models_[i]->UpdataModelFromData();
		}

		std::cout << cams_[0]->cam_model_->f_ << " ";
		std::cout << cams_[0]->cam_model_->k1_ << " ";
		std::cout << cams_[0]->cam_model_->k2_ << std::endl;
	}

	void SLAMGPS::SaveModel()
	{
		std::string fold = "";
		if (!std::experimental::filesystem::exists(fold))
		{
			std::experimental::filesystem::create_directory(fold);
		}

		// write out point cloud
		std::string path_ptcloud = fold + "//pts_cams.txt";
		WriteCameraPointsOut(path_ptcloud);

		// write out points
		std::string path_pt3d = fold + "//pts.txt";
		std::ofstream ofs_pt(path_pt3d);
		ofs_pt << pts_.size() << std::endl;
		ofs_pt.precision(20);
		for (size_t i = 0; i < pts_.size(); i++)
		{
			ofs_pt << pts_[i]->data[0] << " " << pts_[i]->data[1] << " " << pts_[i]->data[2] << std::endl;
		}
		ofs_pt.close();

		// write out cameras
		std::string path_cam = fold + "//cams.txt";
		std::ofstream ofs_cam(path_cam);
		ofs_cam << cams_.size() << std::endl;
		ofs_cam.precision(20);
		for (size_t i = 0; i < cams_.size(); i++)
		{
			ofs_cam << cams_[i]->id_img_ << " ";
			ofs_cam << cams_[i]->data[0] << " " << cams_[i]->data[1] << " " << cams_[i]->data[2] << " ";    // angle-axis
			ofs_cam << cams_[i]->pos_ac_.c(0) << " " << cams_[i]->pos_ac_.c(1) << " " << cams_[i]->pos_ac_.c(2) << " ";  // camera center
			ofs_cam << cams_[i]->cam_model_->data[0] << " " << cams_[i]->cam_model_->data[1] << " " << cams_[i]->cam_model_->data[2] << std::endl;    // f, k1, k2
		}
		ofs_cam.close();

		// write out points of each camera
		for (size_t i = 0; i < cams_.size(); i++)
		{
			std::string path_cam_pts = fold + "//point_cam" + std::to_string(i) + ".txt";
			std::ofstream ofs_cam_pts(path_cam_pts);
			ofs_cam_pts.precision(20);

			for (auto iter = cams_[i]->pts_.begin(); iter != cams_[i]->pts_.end(); iter++)
			{
				ofs_cam_pts << iter->second->data[0] << " " << iter->second->data[1] << " " << iter->second->data[2] << std::endl;
			}
			ofs_cam_pts.close();
		}

		// clearn current results
		for (size_t i = 0; i < pts_.size(); i++)
		{
			delete pts_[i];
		}
		pts_.clear();

		for (size_t i = 0; i < cams_.size(); i++)
		{
			delete cams_[i];
		}
		cams_.clear();

		for (size_t i = 0; i < cam_models_.size(); i++)
		{
			delete cam_models_[i];
		}
		cam_models_.clear();
	}

	void SLAMGPS::WriteCameraPointsOut(std::string path)
	{
		std::ofstream ofs(path);
		if (!ofs.is_open())
		{
			return;
		}
		ofs << std::fixed << std::setprecision(8);

		float span = 100;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (abs(pts_[i]->data[0]) > span || abs(pts_[i]->data[1]) > span || abs(pts_[i]->data[2]) > span)
			{
				continue;
			}

			ofs << pts_[i]->data[0] << " " << pts_[i]->data[1] << " " << pts_[i]->data[2] 
				<< " " << 0 << " " << 0 << " " << 0 << std::endl;
		}

		double scale = (cams_[1]->pos_ac_.c- cams_[0]->pos_ac_.c).squaredNorm() / 10;
		//double scale = 0.01;
		for (size_t i = 0; i < cams_.size(); i++)
		{
			std::vector<Eigen::Vector3d> axis(3);
			axis[0] = cams_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(1, 0, 0);
			axis[1] = cams_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(0, 1, 0);
			axis[2] = cams_[i]->pos_rt_.R.inverse() * Eigen::Vector3d(0, 0, 1);

			std::vector<Eigen::Vector3d> cam_pts;
			GenerateCamera3D(cams_[i]->pos_ac_.c, axis, cams_[i]->cam_model_->f_,
				cams_[i]->cam_model_->w_, cams_[i]->cam_model_->h_, scale, cam_pts);

			for (size_t j = 0; j < cam_pts.size(); j++)
			{
				if (abs(cam_pts[j](0)) > span || abs(cam_pts[j](1)) > span || abs(cam_pts[j](2)) > span)
				{
					continue;
				}

				ofs << cam_pts[j](0) << " " << cam_pts[j](1) << " " << cam_pts[j](2)
					<< " " << 255 << " " << 0 << " " << 0 << std::endl;
			}
		}
		ofs.close();
	}

	void SLAMGPS::GrawGPS(std::string path)
	{
		double xmin = 1000000000000;
		double xmax = -xmin;
		double ymin = xmin;
		double ymax = -ymin;
		for (size_t i = 0; i < cams_gps_.size(); i++)
		{
			double x = cams_gps_[i].x;
			double y = cams_gps_[i].y;
			if (x < xmin) xmin = x;
			if (x > xmax) xmax = x;
			if (y < ymin) ymin = y;
			if (y > ymax) ymax = y;
		}

		double xspan = xmax - xmin;
		double yspan = ymax - ymin;
		double span = MAX(xspan, yspan);
		double step = (span + 1) / 2000;
		int h = yspan / step;
		int w = xspan / step;
		cv::Mat img(h, w, CV_8UC3, cv::Scalar(255, 255, 255));
	
		for (size_t i = 0; i < cams_gps_.size(); i++)
		{
			double x = cams_gps_[i].x;
			double y = cams_gps_[i].y;
			int xx = (x - xmin) / step;
			int yy = (y - ymin) / step;
			cv::circle(img, cv::Point(xx, yy), 5, cv::Scalar(0, 0, 255));
			cv::putText(img, std::to_string(i), cv::Point(xx, yy), 1, 1, cv::Scalar(255, 0, 0));
		}
		cv::imwrite(path, img);
	}

	void SLAMGPS::GrawSLAM(std::string path)
	{
		double xmin = 1000000000000;
		double xmax = -xmin;
		double ymin = xmin;
		double ymax = -ymin;
		for (size_t i = 0; i < cams_.size(); i++)
		{
			double x = cams_[i]->pos_ac_.c(0);
			double y = cams_[i]->pos_ac_.c(2);
			if (x < xmin) xmin = x;
			if (x > xmax) xmax = x;
			if (y < ymin) ymin = y;
			if (y > ymax) ymax = y;
		}

		double xspan = xmax - xmin;
		double yspan = ymax - ymin;
		double span = MAX(xspan, yspan);
		double step = (span + 0.1) / 2000;
		int h = yspan / step;
		int w = xspan / step;
		cv::Mat img(h, w, CV_8UC3, cv::Scalar(255, 255, 255));

		for (size_t i = 0; i < cams_gps_.size(); i++)
		{
			double x = cams_[i]->pos_ac_.c(0);
			double y = cams_[i]->pos_ac_.c(2);
			int xx = (x - xmin) / step;
			int yy = (y - ymin) / step;
			cv::circle(img, cv::Point(xx, yy), 5, cv::Scalar(0, 0, 255));
			cv::putText(img, std::to_string(i), cv::Point(xx, yy), 1, 1, cv::Scalar(255, 0, 0));
		}
		cv::imwrite(path, img);
	}

	void SLAMGPS::SaveUndistortedImage(std::string fold)
	{
		// undistortion
		cv::Mat K(3, 3, CV_64FC1);
		K.at<double>(0, 0) = 450.495, K.at<double>(0, 1) = 0.0, K.at<double>(0, 2) = 499.215;
		K.at<double>(1, 0) = 0.0, K.at<double>(1, 1) = 450.495, K.at<double>(1, 2) = 380.510;
		K.at<double>(2, 0) = 0.0, K.at<double>(2, 1) = 0.0, K.at<double>(2, 2) = 1.0;

		cv::Mat dist(1, 5, CV_64FC1);
		dist.at<double>(0, 0) = -0.25653475791443974829;
		dist.at<double>(0, 1) = 0.08229711989891387580;
		dist.at<double>(0, 2) = -0.00071314261865646465;
		dist.at<double>(0, 3) = 0.00006466208069485206;
		dist.at<double>(0, 4) = -0.01320155290268222939;

		//
		std::string fold_rgb = fold + "\\rgb";
		std::string fold_image = fold + "\\undistort_image";
		if (!std::experimental::filesystem::exists(fold_image)) {
			std::experimental::filesystem::create_directory(fold_image);
		}

		for (size_t i = 0; i < cams_.size(); i++)
		{
			// write out image
			std::string path_in = fold_rgb + "\\" + cams_name_[i] + ".jpg";
			std::string path_out = fold_image + "\\" + cams_name_[i] + ".jpg";
			cv::Mat img = cv::imread(path_in);
			cv::Mat img_undistort;
			cv::undistort(img, img_undistort, K, dist);
			cv::imwrite(path_out, img_undistort);
		}
	}

	void SLAMGPS::SaveForSure(std::string fold)
	{
		// undistortion
		cv::Mat K(3, 3, CV_64FC1);
		K.at<double>(0, 0) = 450.495, K.at<double>(0, 1) = 0.0, K.at<double>(0, 2) = 499.215;
		K.at<double>(1, 0) = 0.0, K.at<double>(1, 1) = 450.495, K.at<double>(1, 2) = 380.510;
		K.at<double>(2, 0) = 0.0, K.at<double>(2, 1) = 0.0, K.at<double>(2, 2) = 1.0;

		cv::Mat dist(1, 5, CV_64FC1);
		dist.at<double>(0, 0) = -0.25653475791443974829;
		dist.at<double>(0, 1) = 0.08229711989891387580;
		dist.at<double>(0, 2) = -0.00071314261865646465;
		dist.at<double>(0, 3) = 0.00006466208069485206;
		dist.at<double>(0, 4) = -0.01320155290268222939;

		//
		std::string file_para = fold + "\\sfm_sure.txt";

		FILE * fp;
		fp = fopen(file_para.c_str(), "w+");
		fprintf(fp, "%s\n", "fileName imageWidth imageHeight");
		fprintf(fp, "%s\n", "camera matrix K [3x3]");
		fprintf(fp, "%s\n", "radial distortion [3x1]");
		fprintf(fp, "%s\n", "tangential distortion [2x1]");
		fprintf(fp, "%s\n", "camera position t [3x1]");
		fprintf(fp, "%s\n", "camera rotation R [3x3]");
		fprintf(fp, "%s\n\n", "camera model P = K [R|-Rt] X");


		for (size_t i = 0; i < cams_.size(); i++)
		{
			fprintf(fp, "%s %d %d\n", cams_name_[i] + ".jpg", cols, rows);
			fprintf(fp, "%.8lf %.8lf %.8lf\n", fx, 0, cx);
			fprintf(fp, "%.8lf %.8lf %.8lf\n", 0, fy, cy);
			fprintf(fp, "%d %d %d\n", 0, 0, 1);
			fprintf(fp, "%.8lf %.8lf %.8lf\n", k1, k2, k3);
			fprintf(fp, "%.8lf %.8lf\n", p1, p2);
			fprintf(fp, "%.8lf %.8lf %.8lf\n", cams_[i]->pos_rt_.t(0), cams_[i]->pos_rt_.t(1), cams_[i]->pos_rt_.t(2));
			fprintf(fp, "%.8lf %.8lf %.8lf\n", cams_[i]->pos_rt_.R(0, 0), cams_[i]->pos_rt_.R(0, 1), cams_[i]->pos_rt_.R(0, 2));
			fprintf(fp, "%.8lf %.8lf %.8lf\n", cams_[i]->pos_rt_.R(1, 0), cams_[i]->pos_rt_.R(1, 1), cams_[i]->pos_rt_.R(1, 2));
			fprintf(fp, "%.8lf %.8lf %.8lf\n", cams_[i]->pos_rt_.R(2, 0), cams_[i]->pos_rt_.R(2, 1), cams_[i]->pos_rt_.R(2, 2));
		}

		fclose(fp);
	}

	void SLAMGPS::SaveforOpenMVS(std::string fold)
	{
		std::string file_para = fold + "\\sfm_openmvs.txt";

		std::ofstream ff(file_para);
		ff << std::fixed << std::setprecision(8);

		// write out cams
		ff << cams_.size() << std::endl;
		for (size_t i = 0; i < cams_.size(); i++)
		{
			ff << cams_name_[i] + ".jpg" << std::endl;
			ff << cams_[i]->cam_model_->f_ << std::endl;
			ff << cams_[i]->pos_rt_.R(0, 0) << " " << cams_[i]->pos_rt_.R(0, 1) << " " << cams_[i]->pos_rt_.R(0, 2) << " "
				<< cams_[i]->pos_rt_.R(1, 0) << " " << cams_[i]->pos_rt_.R(1, 1) << " " << cams_[i]->pos_rt_.R(1, 2) << " "
				<< cams_[i]->pos_rt_.R(2, 0) << " " << cams_[i]->pos_rt_.R(2, 1) << " " << cams_[i]->pos_rt_.R(2, 2) << std::endl;
			ff << cams_[i]->pos_rt_.t(0) << " " << cams_[i]->pos_rt_.t(1) << " " << cams_[i]->pos_rt_.t(2) << std::endl;
		}

		//
		std::map<int, int> cams_info;
		for (size_t i = 0; i < cams_.size(); i++) {
			cams_info.insert(std::pair<int, int>(cams_[i]->id_, i));
		}

		// write out points
		int count_good = pts_.size();
		std::vector<int> num_goods(pts_.size(), 0);
		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (pts_[i]->is_bad_estimated_)
			{
				count_good--;
				continue;
			}

			auto it1 = pts_[i]->pts2d_.begin();
			int count_t = pts_[i]->pts2d_.size();
			while (it1 != pts_[i]->pts2d_.end())
			{
				int x = it1->second(0) + cx;
				int y = it1->second(1) + cy;
				if (x<0 || x>=cols || y<0 || y >= rows)
				{
					count_t--;
				}
				it1++;
			}

			num_goods[i] = count_t;
			if (count_t < 2)
			{
				count_good--;
			}
		}

		ff << count_good << std::endl;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (num_goods[i] < 2 || pts_[i]->is_bad_estimated_)
			{
				continue;
			}

			ff << pts_[i]->data[0] << " " << pts_[i]->data[1] << " " << pts_[i]->data[2] << " ";
			ff << 255 << " " << 255 << " " << 255 << " ";
			ff << num_goods[i] << std::endl;

			auto it1 = pts_[i]->pts2d_.begin();
			auto it2 = pts_[i]->cams_.begin();
			while (it1 != pts_[i]->pts2d_.end())
			{
				int id_cam = it2->second->id_;
				std::map<int, int >::iterator iter = cams_info.find(id_cam);

				int x = it1->second(0) + cx;
				int y = it1->second(1) + cy;
				if (!(x < 0 || x >= cols || y < 0 || y >= rows))
				{
					ff << iter->second << " " << x << " " << y << std::endl;
				}				
				it1++;  it2++;
			}
		}
		ff.close();
	}

	void SLAMGPS::SaveforCMVS(std::string fold)
	{
		std::string fold_cmvs = fold + "\\cmvs";
		if (!std::experimental::filesystem::exists(fold_cmvs)) {
			std::experimental::filesystem::create_directory(fold_cmvs);
		}
		std::string fold_img = fold_cmvs + "\\visualize";
		if (!std::experimental::filesystem::exists(fold_img)) {
			std::experimental::filesystem::create_directory(fold_img);
		}
		std::string fold_txt = fold_cmvs + "\\txt";
		if (!std::experimental::filesystem::exists(fold_txt)) {
			std::experimental::filesystem::create_directory(fold_txt);
		}
		std::string fold_rgb = fold + "\\rgb";

		// step1: save bundle
		std::string file_para = fold_cmvs + "\\bundle.rd.out";
		int count_good = pts_.size();
		std::vector<int> num_goods(pts_.size(), 0);
		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (pts_[i]->is_bad_estimated_)
			{
				count_good--;
			}
		}

		//
		std::ofstream ff(file_para);
		ff << std::fixed << std::setprecision(8);

		// cams
		ff << "# Bundle file v0.3" << std::endl;
		ff << cams_.size() << " " << count_good << std::endl;
		for (size_t i = 0; i < cams_.size(); i++)
		{
			ff << (fx + fy) / 2.0 << " " << 0.0 << " " << 0.0 << std::endl;
			ff << cams_[i]->pos_rt_.R(0, 0) << " " << cams_[i]->pos_rt_.R(0, 1) << " " << cams_[i]->pos_rt_.R(0, 2) << " "
				<< cams_[i]->pos_rt_.R(1, 0) << " " << cams_[i]->pos_rt_.R(1, 1) << " " << cams_[i]->pos_rt_.R(1, 2) << " "
				<< cams_[i]->pos_rt_.R(2, 0) << " " << cams_[i]->pos_rt_.R(2, 1) << " " << cams_[i]->pos_rt_.R(2, 2) << std::endl;
			ff << cams_[i]->pos_rt_.t(0) << " " << cams_[i]->pos_rt_.t(1) << " " << cams_[i]->pos_rt_.t(2) << std::endl;
		}

		// points
		std::map<int, int> cams_info;
		for (size_t i = 0; i < cams_.size(); i++) {
			cams_info.insert(std::pair<int, int>(cams_[i]->id_, i));
		}

		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (pts_[i]->is_bad_estimated_)
			{
				continue;
			}

			ff << pts_[i]->data[0] << " " << pts_[i]->data[1] << " " << pts_[i]->data[2] << " ";
			ff << 255 << " " << 255 << " " << 255 << " ";
			ff << pts_[i]->pts2d_.size() << std::endl;

			auto it1 = pts_[i]->pts2d_.begin();
			auto it2 = pts_[i]->cams_.begin();
			while (it1 != pts_[i]->pts2d_.end())
			{
				int idx_pt = it2->first;
				int id_cam = it2->second->id_;
				std::map<int, int >::iterator iter = cams_info.find(id_cam);

				int x = it1->second(0);
				int y = it1->second(1);
				ff << iter->second << " " << idx_pt << " " << float(x) << " " << float(y) << std::endl;
				it1++;  it2++;
			}
		}
		ff.close();

		// step2: save txt and image
		for (size_t i = 0; i < cams_.size(); i++)
		{
			std::string name = std::to_string(i);
			name = std::string(8 - name.length(), '0') + name;

			// txt
			Eigen::Matrix3d R = cams_[i]->pos_rt_.R;
			Eigen::Vector3d t = cams_[i]->pos_rt_.t;
			cv::Mat M = (cv::Mat_<double>(3, 4) << R(0, 0), R(0, 1), R(0, 2), t(0),
				                                   R(1, 0), R(1, 1), R(1, 2), t(1),
				                                   R(2, 0), R(2, 1), R(2, 2), t(2));
			cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
				0, fy, cy,
				0, 0, 1);
			cv::Mat P = K * M;

			std::ofstream ff(fold_txt + "\\" + name + ".txt");
			ff << std::fixed << std::setprecision(8);
			ff << "CONTOUR" << std::endl;
			for (size_t m = 0; m < 3; m++) {
				for (size_t n = 0; n < 4; n++) {
					ff << P.at<double>(m, n) << " ";
				}
				ff << std::endl;
			}
			ff.close();

			// image
			std::string file_img_in = fold_rgb + "\\" + cams_name_[i] + ".jpg";
			std::string file_img_out = fold_img + "\\" + name + ".jpg";
			cv::Mat img = cv::imread(file_img_in);
			cv::imwrite(file_img_out, img);
		}
	}

	void SLAMGPS::GetAccuracy(std::string file, std::vector<CameraModel*> cam_models, std::vector<Camera*> cams, std::vector<Point3D*> pts)
	{
		accuracer_ = new AccuracyAssessment();
		accuracer_->SetData(cam_models, cams, pts);

		int n_obs;
		double e_avg, e_mse;
		std::vector<double> errors;
		accuracer_->ErrorReprojectionPts(e_avg, e_mse, n_obs, errors);
		std::cout << "e_avg " << e_avg << " e_mse " << e_mse << " n_obs " << n_obs << std::endl;

		int count_outliers = 0;
		for (size_t i = 0; i < errors.size(); i++)
		{
			if (errors[i] > th_outlier)
			{
				pts_[i]->is_bad_estimated_ = true;
				count_outliers++;
			}
		}
		std::cout << "outliers " << count_outliers << " inliers " << errors.size() - count_outliers << std::endl;
	}

	void SLAMGPS::DrawPts(std::vector<int> img_ids, std::string fold)
	{
		// get all the images
		std::vector<cv::Mat> imgs(img_ids.size());
		for (size_t i = 0; i < img_ids.size(); i++)
		{
			int id = img_ids[i];
			std::string img_name = fold + "\\" + cams_name_[id] + ".jpg";
			imgs[i] = cv::imread(img_name);
		}

		std::map<int, int> cams_info;
		for (size_t i = 0; i < cams_.size(); i++) {
			cams_info.insert(std::pair<int, int>(cams_[i]->id_, i));
		}

		// 
		for (size_t i = 0; i < pts_.size(); i++)
		{
			auto it1 = pts_[i]->pts2d_.begin();
			auto it2 = pts_[i]->cams_.begin();
			while (it1 != pts_[i]->pts2d_.end())
			{
				int idx_pt = it2->first;
				int id_cam = it2->second->id_;
				std::map<int, int >::iterator iter = cams_info.find(id_cam);
				bool is_in = false;
				int idx_in = 0;
				for (size_t j = 0; j < img_ids.size(); j++) {
					if (iter->second == img_ids[j]) {
						is_in = true;
						idx_in = j;
						break;
					}
				}
				if (is_in) {
					int x = it1->second(0) + cx;
					int y = it1->second(1) + cy;
					cv::circle(imgs[idx_in], cv::Point(x, y), 2, cv::Scalar(0, 0, 255), 2);
					cv::putText(imgs[idx_in], std::to_string(i), cv::Point(x, y), 1, 1, cv::Scalar(255, 0, 0));
				}
				it1++;  it2++;
			}
		}

		for (size_t i = 0; i < imgs.size(); i++)
		{
			int id = img_ids[i];
			std::string file_out = "F:\\" + cams_name_[id] + ".jpg";
			cv::imwrite(file_out, imgs[i]);
		}
	}

	void SLAMGPS::WriteOutMatches(int idx1, int idx2, std::vector<std::pair<int, int>>& matches)
	{
		int num_match = matches.size();
		if (!num_match)
		{
			return;
		}

		int *tempi = new int[num_match * 2];
		for (size_t m = 0; m < num_match; m++)
		{
			tempi[2 * m + 0] = matches[m].first;
			tempi[2 * m + 1] = matches[m].second;
		}

		// file i
		std::ofstream ofsi;
		std::string match_file_i = db_.output_fold_ + "//" + std::to_string(idx1) + "_match";
		ofsi.open(match_file_i, std::ios::out | std::ios::app | std::ios::binary);
		ofsi.write((const char*)(&idx2), sizeof(int));
		ofsi.write((const char*)(&num_match), sizeof(int));
		ofsi.write((const char*)(tempi), num_match * 2 * sizeof(int));
		ofsi.close();

		delete[] tempi;
	}

	void SLAMGPS::WriteOutMatchGraph(std::vector<std::vector<int>> &match_graph)
	{
		std::string path = db_.output_fold_ + "//" + "graph_matching.txt";
		std::ofstream ofs(path, std::ios::binary);

		if (!ofs.is_open()) {
			return;
		}

		for (size_t i = 0; i < match_graph.size(); i++) {
			for (size_t j = 0; j < match_graph[i].size(); j++) {
				ofs << match_graph[i][j] << " ";
			}
			ofs << std::endl;
		}
		ofs.close();
	}

	void SLAMGPS::WriteOutPriorInfo(std::string file, std::vector<std::vector<int>>& ids, 
		std::vector<std::vector<cv::Mat>>& Fs, std::vector<std::vector<cv::Mat>>& Hs)
	{
		std::ofstream ff(file);
		ff << std::fixed << std::setprecision(12);

		ff << ids.size() << std::endl;
		for (size_t i = 0; i < ids.size(); i++)
		{
			ff << ids[i].size() << std::endl;
			for (size_t j = 0; j < ids[i].size(); j++)
			{
				ff << ids[i][j] << " ";
				for (size_t m = 0; m < 3; m++)
				{
					for (size_t n = 0; n < 3; n++)
					{
						ff << Fs[i][j].at<double>(m, n) << " ";
						ff << Hs[i][j].at<double>(m, n) << " ";
					}
				}
				ff << std::endl;
			}
		}
		ff.close();
	}

	void SLAMGPS::ReadinPriorInfo(std::string file, std::vector<std::vector<int>>& ids, 
		std::vector<std::vector<cv::Mat>>& Fs, std::vector<std::vector<cv::Mat>>& Hs)
	{
		std::ifstream ff(file);
		int num = 0;
		ff >> num;
		ids.resize(num);
		Fs.resize(num);
		Hs.resize(num);

		for (size_t i = 0; i < num; i++)
		{
			int n_cams = 0;
			ff >> n_cams;
			ids[i].resize(n_cams);
			Fs[i].resize(n_cams);
			Hs[i].resize(n_cams);

			for (size_t j = 0; j < n_cams; j++)
			{
				ff >> ids[i][j];

				Fs[i][j] = cv::Mat(3, 3, CV_64FC1);
				Hs[i][j] = cv::Mat(3, 3, CV_64FC1);
				for (size_t m = 0; m < 3; m++)
				{
					for (size_t n = 0; n < 3; n++)
					{
						ff >> Fs[i][j].at<double>(m, n);
						ff >> Hs[i][j].at<double>(m, n);
					}
				}
			}
		}
		ff.close();
	}

}  // namespace objectsfm
