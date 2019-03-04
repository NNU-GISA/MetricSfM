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
#include "utils/gps_error_pose_raletive_angle.h"
#include "utils/gps_error_pose_raletive_dis.h"
#include "utils/gps_error_pose_absolute.h"
#include "utils/geo_verification.h"
#include "utils/transformation.h"

#include "flann/flann.h"

namespace objectsfm {

	SLAMGPS::SLAMGPS()
	{
		rows = 3000;
		cols = 4000;
		th_outlier = 3.0;
		resize_ratio = 0.5;	
		use_slam_pt_ = false;
	}

	SLAMGPS::~SLAMGPS()
	{
	}

	void SLAMGPS::Run(std::string fold)
	{
		std::cout << "begin..." << std::endl;
		//Convert2GPSDense(fold);
		//Convert2GPS(fold);
		//exit(0);

		/*************************/
		/* Part 1: pre-processing*/
		/*************************/

		// read in slam information
		std::string file_slam = fold + "\\KeyFramePts.txt";
		ReadinSLAM(file_slam);
		WriteCameraPointsOut(fold + "\\cam_pts1.txt");

		// read in gps
		std::string file_gps = fold + "\\pos.txt";
		std::map<int, cv::Point3d> gps_info;
		ReadinGPS(file_gps, gps_info);

		// accosiation
		std::string file_rgb = fold + "\\rgb.txt";
		AssociateCameraGPS(file_rgb, gps_info);


		/**********************************************/
		/* Part 2: matching, triangulation, and bundle*/
		/**********************************************/

		// feature extraction
		fold_image_ = fold + "\\original";
		FeatureExtraction(fold);

		// convert to the gps world coordinates
		AbsoluteOrientationWithGPSGlobal();
		WriteGPSPose(fold + "\\gps_pos.txt");
		WriteOffset(fold + "\\offset.txt");

		// feature matching with pose information
		if (0) {
			FeatureMatching(fold);
		}

		// triangulation
		Triangulation(fold);
		WriteCameraPointsOut(fold + "\\cam_pts2.txt");


		GPSRegistration2(fold);
		WriteCameraPointsOut(fold + "\\cam_pts3.txt");

		// do adjustment
		FullBundleAdjustment();
		std::cout << 2 << std::endl;
		std::string file_accu = fold + "\\accuracy.txt";
		GetAccuracy(file_accu, cam_models_, cams_, pts_);
		WriteCameraPointsOut(fold + "\\cam_pts4.txt");

		Convert2GPS(fold);

		/**************************/
		/* Part 3: write out*/
		/**************************/

		// save
		std::cout << "saving output..." << std::endl;
		SaveUndistortedImage(fold);

		SaveforCMVS(fold);

		SaveforOpenMVS(fold);

		SaveforMSP(fold);
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

				pts_[i]->AddObservation(cams_[idx_cam], (px - cx) / resize_ratio, (py - cy) / resize_ratio, j);
			}
		}

		// camera models
		fx /= resize_ratio;
		fy /= resize_ratio;
		cx /= resize_ratio;
		cy /= resize_ratio;

		cam_models_.resize(1);
		cam_models_[0] = new CameraModel(cam_models_.size(), rows, cols, (fx+fy)/2.0, 0.0, "lu", "lu");
		cam_models_[0]->SetIntrisicParas((fx + fy) / 2.0, cx, cy);
		for (size_t i = 0; i < cams_.size(); i++)
		{
			cams_[i]->AssociateCamereModel(cam_models_[0]);
		}
	}

	void SLAMGPS::ReadinGPS(std::string file_gps, std::map<int, cv::Point3d> &gps_info )
	{
		int ellipsoid_id = 23; // WGS-84
		std::string zone_id = "17N";

		std::ifstream ff(file_gps);
		int id;
		double lat, lon, alt;
		double x, y;
		std::vector<int> gps_ids;
		std::vector<cv::Point3d> gps_pts;
		double alt_avg = 0.0;
		while (!ff.eof())
		{
			ff >> id >> lat >> lon >> alt;
			LLtoUTM(ellipsoid_id, lat, lon, y, x, (char*)zone_id.c_str());
			gps_pts.push_back(cv::Point3d(x, y, alt));
			gps_ids.push_back(id);
			alt_avg += alt;
		}
		ff.close();

		alt_avg /= gps_pts.size();
		for (size_t i = 0; i < gps_pts.size(); i++)
		{
			gps_info.insert(std::pair<int, cv::Point3d>(gps_ids[i], cv::Point3d(gps_pts[i].x, gps_pts[i].y, alt_avg)));
		}
	}

	void SLAMGPS::AssociateCameraGPS(std::string file_rgb, std::map<int, cv::Point3d>& gps_info)
	{
		std::ifstream ff(file_rgb);
		std::string s0;
		std::getline(ff, s0);
		std::getline(ff, s0);
		std::getline(ff, s0);


		std::vector<cv::Point3d> gps_all_frame;
		std::vector<std::string> names_all_frame;
		while (!ff.eof())
		{
			std::string s;
			std::getline(ff, s);
			if (s == "") {
				break;
			}
			size_t loc1 = s.find_last_of('/');
			size_t loc2 = s.find_last_of('.');
			std::string name = s.substr(loc1 + 1, loc2 - loc1 - 1);
			int id = std::stoi(name);

			std::map<int, cv::Point3d>::iterator iter = gps_info.find(id);
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
		std::string fold_image = fold_image_;
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
		int win_size = 5;
		int th_same_pts = 20;
		float th_epipolar = 2.0 / resize_ratio;
		float th_distance = 5.0 / resize_ratio;
		float th_ratio_f = 0.5;
		float th_h_f_ratio = 0.90;
		float th_first_second_ratio = 0.80;
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
		std::cout << "win_size " << win_size << std::endl;
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
					if (count_inlier_f < pts1.size()*th_ratio_f || count_inlier_f < 30) {
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
					if (ratio > th_first_second_ratio) {
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
					if (homo_dis > 40 * th_distance) {
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
				cv::Mat F;
				bool isOK = GeoVerification::GeoVerificationFundamental(pts1_init, pts2_init, inliers, F);
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
		for (size_t i = 0; i <n_img; i++)
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
		double th_tri_angle = 3.0 / 180.0*CV_PI;
		int count_bad = 0;
		for (size_t i = 0; i < pts_new_.size(); i++)
		{
			bool is_ok = pts_new_[i]->Trianglate2(th_outlier, th_tri_angle);
			if (!is_ok || pts_new_[i]->cams_.size() < 3) {
			//if (!is_ok) {
				pts_new_[i]->is_bad_estimated_ = true;
				count_bad++;
			}
		}
		std::cout << "count_bad " << count_bad << " count_good " << pts_new_.size() - count_bad << std::endl;
		
		// accuracy analysis
		std::cout << "pts slam " << pts_.size() << std::endl;
		std::cout << "pts sfm " << pts_new_.size() << std::endl;
		if (use_slam_pt_) {
			for (size_t i = 0; i < pts_new_.size(); i++) {
				pts_.push_back(pts_new_[i]);
			}
		}
		else {
			pts_ = pts_new_;
		}
		std::cout << "pts all " << pts_.size() << std::endl;

		std::string file_accu = fold + "\\accuracy.txt";
		GetAccuracy(file_accu, cam_models_, cams_, pts_);

		//WriteCameraPointsOut(fold + "\\slam_cam_pts2.txt");
	}

	void SLAMGPS::Dilution(std::string fold)
	{

	}

	void SLAMGPS::FullBundleAdjustment()
	{
		ceres::Problem problem;
		ceres::Solver::Options options;
		ceres::Solver::Summary summary;

		options.max_num_iterations = 200;
		options.minimizer_progress_to_stdout = true;
		options.num_threads = 8;
		options.linear_solver_type = ceres::DENSE_SCHUR;

		// add reprojection error
		int count1 = 0;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (pts_[i]->is_bad_estimated_) {
				continue;
			}

			std::map<int, Camera*>::iterator iter_cams = pts_[i]->cams_.begin();
			std::map<int, Eigen::Vector2d>::iterator iter_pts = pts_[i]->pts2d_.begin();
			while (iter_cams != pts_[i]->cams_.end())
			{
				ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
				ceres::CostFunction* cost_function;

				cost_function = ReprojectionErrorPoseCamXYZ::Create(iter_pts->second(0), iter_pts->second(1), pts_[i]->weight);
				problem.AddResidualBlock(cost_function, loss_function, iter_cams->second->data, iter_cams->second->cam_model_->data, pts_[i]->data);

				//cost_function = ReprojectionErrorPoseXYZ::Create(iter_pts->second(0), iter_pts->second(1), iter_cams->second->cam_model_->data, pts_[i]->weight);
				//problem.AddResidualBlock(cost_function, loss_function, iter_cams->second->data, pts_[i]->data);

				iter_cams++;
				iter_pts++;

				count1++;
			}
		}

		// add gps-topology error
		int count2 = 0, count3 = 0;
		if (1)
		{
			bool use_relative_gps_angle = false;
			bool use_relative_gps_dis = false;
			bool use_absolute_gps = true;
			if (use_relative_gps_angle)
			{
				double ddx = cams_[0]->pos_ac_.c[0] - cams_[10]->pos_ac_.c[0];
				double ddy = cams_[0]->pos_ac_.c[1] - cams_[10]->pos_ac_.c[1];
				double ddz = cams_[0]->pos_ac_.c[2] - cams_[10]->pos_ac_.c[2];
				double len_baseline = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
				len_baseline /= 2.0;
				float th_triangle_len = 10.0 * len_baseline;

				int step = 2;
				double weight = 1000.0;
				for (int i = 6; i < cams_.size() - 10; i++)
				{
					int id1 = i;
					int k = (i - 6) / 2;
					for (int j = 0; j < k; j++)
					{
						int id2 = j;
						int id3 = k + j;

						cv::Point3d v12 = cams_gps_[id2] - cams_gps_[id1];
						double l12 = sqrt(v12.x*v12.x + v12.y*v12.y);

						cv::Point3d v13 = cams_gps_[id3] - cams_gps_[id1];
						double l13 = sqrt(v13.x*v13.x + v13.y*v13.y);

						cv::Point3d v23 = cams_gps_[id3] - cams_gps_[id2];
						double l23 = sqrt(v23.x*v23.x + v23.y*v23.y);

						if (l12 < th_triangle_len || l13 < th_triangle_len || l23 < th_triangle_len)
						{
							continue;
						}

						double angle1 = acos((v12.x*v13.x + v12.y*v13.y) / l12 / l13);
						double angle2 = acos((-v12.x*v23.x - v12.y*v23.y) / l12 / l23);
						double angle3 = acos((v13.x*v23.x + v13.y*v23.y) / l13 / l23);
						if (angle1 != angle1 || angle2 != angle2 || angle3 != angle3)
						{
							continue;
						}

						ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
						ceres::CostFunction* cost_function;
						cost_function = GPSErrorPoseRelativeAngle::Create(angle1, angle2, angle3, weight);
						problem.AddResidualBlock(cost_function, loss_function, cams_[id1]->data, cams_[id2]->data, cams_[id3]->data);

						count2++;
					}
				}
			}
			if (use_relative_gps_dis)
			{
				double ddx = cams_[0]->pos_ac_.c[0] - cams_[2]->pos_ac_.c[0];
				double ddy = cams_[0]->pos_ac_.c[1] - cams_[2]->pos_ac_.c[1];
				double ddz = cams_[0]->pos_ac_.c[2] - cams_[2]->pos_ac_.c[2];
				double len_baseline = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
				len_baseline /= 2.0;
				float th_triangle_len = 50.0 * len_baseline;

				int step = 2;
				double weight = 10;
				for (int i = 6; i < cams_.size(); i++)
				{
					int id1 = i;
					int k = (i - 6) / 2;
					for (int j = 0; j < k; j++)
					{
						int id2 = j;
						int id3 = k + j;

						cv::Point3d v1 = cams_gps_[id2] - cams_gps_[id1];
						double l1 = sqrt(v1.x*v1.x + v1.y*v1.y);

						cv::Point3d v2 = cams_gps_[id3] - cams_gps_[id2];
						double l2 = sqrt(v2.x*v2.x + v2.y*v2.y);

						cv::Point3d v3 = cams_gps_[id1] - cams_gps_[id3];
						double l3 = sqrt(v3.x*v3.x + v3.y*v3.y);

						if (l1 < th_triangle_len || l2 < th_triangle_len || l3 < th_triangle_len)
						{
							continue;
						}

						double ratio12 = l1 / l2;
						double ratio23 = l2 / l3;

						ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
						ceres::CostFunction* cost_function;
						cost_function = GPSErrorPoseRelativeDis::Create(ratio12, ratio23, weight);
						problem.AddResidualBlock(cost_function, loss_function, cams_[id1]->data, cams_[id2]->data, cams_[id3]->data);

						count2++;
					}
				}
			}
			if (use_absolute_gps)
			{
				for (int i = 0; i < cams_.size(); i++)
				{
					ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
					ceres::CostFunction* cost_function;
					double weight = count1 / cams_.size();
					cost_function = GPSErrorPoseAbsolute::Create(cams_gps_[i].x, cams_gps_[i].y , cams_gps_[i].z, weight);
					problem.AddResidualBlock(cost_function, loss_function, cams_[i]->data);

					count3++;
				}
			}
			
		}
		std::cout << "count1 " << count1 << " count2 " << count2 << " count3 " << count3 << std::endl;

		std::cout << cams_[0]->cam_model_->f_ << " ";
		std::cout << cams_[0]->cam_model_->k1_ << " ";
		std::cout << cams_[0]->cam_model_->k2_ << " ";
		std::cout << cams_[0]->cam_model_->dcx_ << " ";
		std::cout << cams_[0]->cam_model_->dcy_ << std::endl;

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
		//for (size_t i = 0; i < pts_.size(); i++)
		//{
		//	pts_[i]->UpdateFromData();
		//}

		std::cout << cams_[0]->cam_model_->f_ << " ";
		std::cout << cams_[0]->cam_model_->k1_ << " ";
		std::cout << cams_[0]->cam_model_->k2_ << " ";
		std::cout << cams_[0]->cam_model_->dcx_ << " ";
		std::cout << cams_[0]->cam_model_->dcy_ << std::endl;
	}

	void SLAMGPS::GPSRegistration(std::string fold)
	{
		// calculate the transformation of cam to gps
		std::vector<Eigen::Matrix3d> Rs_local;
		std::vector<Eigen::Vector3d> ts_local;
		AbsoluteOrientationWithGPSLocal(50, Rs_local, ts_local);

		// transform all the 3d points via cam tramsform
		std::map<int, int> cams_info;
		for (size_t i = 0; i < cams_.size(); i++) {
			cams_info.insert(std::pair<int, int>(cams_[i]->id_, i));
		}

		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (i%1000 == 0)
			{
				std::cout << i << std::endl;
			}

			if (pts_[i]->is_bad_estimated_) {
				continue;
			}

			auto it = pts_[i]->cams_.begin();
			Eigen::Vector3d pt_avg(0.0, 0.0, 0.0);
			std::vector<Eigen::Vector3d> pts_new;
			while (it != pts_[i]->cams_.end())
			{
				int id_temp = it->second->id_;
				std::map<int, int >::iterator iter = cams_info.find(id_temp);
				int id_cam = iter->second;

				// new point after transformation
				Eigen::Vector3d pt(pts_[i]->data[0], pts_[i]->data[1], pts_[i]->data[2]);
				pt = Rs_local[id_cam] * pt + ts_local[id_cam];
				pts_new.push_back(pt);
				pt_avg += pt;
				//break;

				it++;
			}

			// averaging as the final result
			pts_[i]->data[0] = pt_avg[0] / pts_new.size();
			pts_[i]->data[1] = pt_avg[1] / pts_new.size();
			pts_[i]->data[2] = pt_avg[2] / pts_new.size();
		}
		//WriteCameraPointsOut(fold + "\\cam_pts4.txt");
	}


	void SLAMGPS::GPSRegistration2(std::string fold)
	{
		// calculate the offset of each cam to the gps
		std::vector<cv::Point3d> cam_offset(cams_.size());
		for (size_t i = 0; i < cams_.size(); i++) {
			cam_offset[i] = cams_gps_[i] - cv::Point3d(cams_[i]->pos_ac_.c[0], cams_[i]->pos_ac_.c[1], cams_[i]->pos_ac_.c[2]);
			//cams_[i]->SetACPose(cams_[i]->pos_ac_.a, Eigen::Vector3d(cams_gps_[i].x, cams_gps_[i].y, cams_gps_[i].z));
		}
		

		// transform all the 3d points via cam offset
		std::map<int, int> cams_info;
		for (size_t i = 0; i < cams_.size(); i++) {
			cams_info.insert(std::pair<int, int>(cams_[i]->id_, i));
		}

		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (i % 1000 == 0)
			{
				std::cout << i << std::endl;
			}

			if (pts_[i]->is_bad_estimated_) {
				continue;
			}

			cv::Point3d offset_i(0.0, 0.0, 0.0);
			double weight_i = 0.0;
			auto it = pts_[i]->cams_.begin();
			while (it != pts_[i]->cams_.end())
			{
				int id_temp = it->second->id_;
				std::map<int, int >::iterator iter = cams_info.find(id_temp);
				int id_cam = iter->second;

				// new point after transformation
				double dx = pts_[i]->data[0] - cams_[id_cam]->pos_ac_.c(0);
				double dy = pts_[i]->data[1] - cams_[id_cam]->pos_ac_.c(1);
				double dz = pts_[i]->data[2] - cams_[id_cam]->pos_ac_.c(2);
				double dis = sqrt(dx * dx + dy * dy + dz * dz);
				double w = 1.0 / (sqrt(dis) + 5.0);
				weight_i += w;
				offset_i += w * cam_offset[id_cam];

				if (i % 1000 == 0)
				{
					std::cout << "weight_i " << weight_i << std::endl;
					std::cout << "offset_i " << offset_i << std::endl;
				}

				it++;
			}
			offset_i.x /= weight_i;
			offset_i.y /= weight_i;
			offset_i.z /= weight_i;

			// averaging as the final result
			pts_[i]->data[0] += offset_i.x;
			pts_[i]->data[1] += offset_i.y;
			pts_[i]->data[2] += offset_i.z;
		}
		
		for (size_t i = 0; i < cams_.size(); i++) {
			cams_[i]->SetACPose(cams_[i]->pos_ac_.a, Eigen::Vector3d(cams_gps_[i].x, cams_gps_[i].y, cams_gps_[i].z));
		}
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

		//float span = 1000;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (pts_[i]->is_bad_estimated_) {
				continue;
			}
			//if (abs(pts_[i]->data[0]) > span || abs(pts_[i]->data[1]) > span || abs(pts_[i]->data[2]) > span) {
			//	continue;
			//}

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
				//if (abs(cam_pts[j](0)) > span || abs(cam_pts[j](1)) > span || abs(cam_pts[j](2)) > span)
				//{
				//	continue;
				//}

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

	void SLAMGPS::GrawSLAMGPS(std::string path)
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
		double step = (span + 0.1) / 2000;
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

	void SLAMGPS::SaveUndistortedImage(std::string fold)
	{
		// undistortion
		cv::Mat K(3, 3, CV_64FC1);
		K.at<double>(0, 0) = cams_[0]->cam_model_->f_,  K.at<double>(0, 1) = 0.0, K.at<double>(0, 2) = cams_[0]->cam_model_->px_;
		K.at<double>(1, 0) = 0.0, K.at<double>(1, 1) = cams_[0]->cam_model_->f_,  K.at<double>(1, 2) = cams_[0]->cam_model_->py_;
		K.at<double>(2, 0) = 0.0, K.at<double>(2, 1) = 0.0, K.at<double>(2, 2) = 1.0;

		cv::Mat dist(1, 5, CV_64FC1);
		dist.at<double>(0, 0) = cams_[0]->cam_model_->k1_;
		dist.at<double>(0, 1) = cams_[0]->cam_model_->k2_;
		dist.at<double>(0, 2) = 0.0;
		dist.at<double>(0, 3) = 0.0;
		dist.at<double>(0, 4) = 0.0;

		//
		std::string fold_rgb = fold_image_;
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
			fprintf(fp, "%.8lf %.8lf %.8lf\n", 0, 0, 0);
			fprintf(fp, "%.8lf %.8lf\n", 0, 0);
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
		int ncluster = cams_.size() / 500;
		int nstep = cams_.size() / ncluster;

		std::map<int, int> cams_info;
		for (size_t i = 0; i < cams_.size(); i++) {
			cams_info.insert(std::pair<int, int>(cams_[i]->id_, i));
		}

		for (size_t k = 0; k < ncluster; k++)
		{
			int cam_ids = k * nstep;
			int cam_ide = (k + 1) * nstep;
			if (cam_ide > cams_.size()) {
				cam_ide = cams_.size();
			}

			std::string fold_cmvs = fold + "\\cmvs" + std::to_string(k);

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
			std::string fold_rgb = fold + "\\undistort_image";

			// step0: find pts belonging to the clusters
			std::vector<int> pt_ids;
			for (size_t i = 0; i < pts_.size(); i++)
			{
				if (pts_[i]->is_bad_estimated_) {
					continue;
				}

				auto it = pts_[i]->cams_.begin();
				while (it != pts_[i]->cams_.end())
				{
					int id_cam = it->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide) {
						pt_ids.push_back(i);
						break;
					}
					it++;
				}
			}


			// step1: save bundle
			std::string file_para = fold_cmvs + "\\bundle.rd.out";

			//
			std::ofstream ff(file_para);
			ff << std::fixed << std::setprecision(8);

			// cams
			std::cout << 111 << std::endl;
			ff << "# Bundle file v0.3" << std::endl;
			ff << cam_ide - cam_ids << " " << pt_ids.size() << std::endl;
			for (size_t i = cam_ids; i < cam_ide; i++)
			{
				ff << cams_[0]->cam_model_->f_ << " " << cams_[0]->cam_model_->k1_ << " " << cams_[0]->cam_model_->k2_ << std::endl;
				ff << cams_[i]->pos_rt_.R(0, 0) << " " << cams_[i]->pos_rt_.R(0, 1) << " " << cams_[i]->pos_rt_.R(0, 2) << " "
					<< cams_[i]->pos_rt_.R(1, 0) << " " << cams_[i]->pos_rt_.R(1, 1) << " " << cams_[i]->pos_rt_.R(1, 2) << " "
					<< cams_[i]->pos_rt_.R(2, 0) << " " << cams_[i]->pos_rt_.R(2, 1) << " " << cams_[i]->pos_rt_.R(2, 2) << std::endl;
				ff << cams_[i]->pos_rt_.t(0) << " " << cams_[i]->pos_rt_.t(1) << " " << cams_[i]->pos_rt_.t(2) << std::endl;
			}

			std::cout << 222 << std::endl;
			// points
			for (size_t i = 0; i < pt_ids.size(); i++)
			{
				int pt_id = pt_ids[i];
				ff << pts_[pt_id]->data[0] << " " << pts_[pt_id]->data[1] << " " << pts_[pt_id]->data[2] << " ";
				ff << 255 << " " << 255 << " " << 255 << " ";
				int count_cams = 0;
				auto it = pts_[pt_id]->cams_.begin();
				while (it != pts_[pt_id]->cams_.end())
				{
					int id_cam = it->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide) {
						count_cams++;
					}
					it++;
				}
				ff << count_cams << std::endl;

				auto it1 = pts_[pt_id]->pts2d_.begin();
				auto it2 = pts_[pt_id]->cams_.begin();
				while (it1 != pts_[pt_id]->pts2d_.end())
				{
					int idx_pt = it1->first;
					int id_cam = it2->second->id_;
					std::map<int, int >::iterator iter = cams_info.find(id_cam);
					if (iter->second >= cam_ids && iter->second < cam_ide) {
						int x = it1->second(0);
						int y = it1->second(1);
						ff << iter->second - cam_ids << " " << idx_pt << " " << float(x) << " " << float(y) << std::endl;
					}
					it1++;  it2++;
				}
			}
			ff.close();
			std::cout << 333 << std::endl;

			// step2: save txt and image
			for (size_t i = cam_ids; i < cam_ide; i++)
			{
				std::string name = std::to_string(i- cam_ids);
				name = std::string(8 - name.length(), '0') + name;

				// txt
				Eigen::Matrix3d R = cams_[i]->pos_rt_.R;
				Eigen::Vector3d t = cams_[i]->pos_rt_.t;
				cv::Mat M = (cv::Mat_<double>(3, 4) << R(0, 0), R(0, 1), R(0, 2), t(0),
					R(1, 0), R(1, 1), R(1, 2), t(1),
					R(2, 0), R(2, 1), R(2, 2), t(2));
				cv::Mat K = (cv::Mat_<double>(3, 3) << cams_[0]->cam_model_->f_, 0, cams_[0]->cam_model_->px_,
					0, cams_[0]->cam_model_->f_, cams_[0]->cam_model_->py_,
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
		
	}

	void SLAMGPS::SaveforMSP(std::string fold)
	{
		std::string fold_msp = fold + "\\msp";
		if (!std::experimental::filesystem::exists(fold_msp)) {
			std::experimental::filesystem::create_directory(fold_msp);
		}

		std::string file_msp = fold_msp + "\\pose.qin";

		std::ofstream ff(file_msp);
		ff << std::fixed << std::setprecision(12);

		ff << cams_.size() << std::endl;
		double pixel_mm = 0.005;
		ff << cams_[0]->cam_model_->f_ * pixel_mm << " " 
		   << cams_[0]->cam_model_->dcx_ * pixel_mm << " " 
		   << cams_[0]->cam_model_->dcy_ * pixel_mm << " " 
		   << pixel_mm << " " << pixel_mm << " " << cols << " " << rows << std::endl;

		Eigen::Matrix3d R_cv2ph;
		R_cv2ph << 1.0, 0.0, 0.0,
			0.0, cos(CV_PI), -sin(CV_PI),
			0.0, sin(CV_PI), cos(CV_PI);

		for (size_t i = 0; i < cams_.size(); i++)
		{
			ff << cams_name_[i] + ".jpg" << " " << cams_[i]->pos_ac_.c(0) << " " << cams_[i]->pos_ac_.c(1) << " " << cams_[i]->pos_ac_.c(2) << " ";

			// convert cv coordinate system to photogrammetry system by rotating around x-axis for pi
			Eigen::Matrix3d Rph = R_cv2ph * cams_[i]->pos_rt_.R;
			double rx = 0.0, ry = 0.0, rz = 0.0;
			rotation::RotationMatrixToEulerAngles(Rph, rx, ry, rz);
			ff << rx << " " << ry << " " << rz;
			if (i < cams_.size() - 1) {
				ff << std::endl;
			}
		}
		ff.close();
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

	void SLAMGPS::AbsoluteOrientationWithGPSGlobal()
	{
		std::vector<Eigen::Vector3d> pts_cam(cams_.size());
		std::vector<Eigen::Vector3d> pts_gps(cams_.size());
		std::vector<double> weight(cams_.size());
		for (size_t i = 0; i < cams_.size(); i++)
		{
			pts_cam[i] = Eigen::Vector3d(cams_[i]->pos_ac_.c);
			pts_gps[i] = Eigen::Vector3d(cams_gps_[i].x, cams_gps_[i].y, cams_gps_[i].z);

			// calculate the weights
			int ids = i - 20; 
			if (ids < 0) {
				ids = 0;
			}
			int ide = i + 20;
			if (ide > cams_.size() - 1) {
				ide = cams_.size() - 1;
			}
			double dxs = pts_gps[ids][0] - pts_gps[i][0];
			double dys = pts_gps[ids][1] - pts_gps[i][1];
			double dxe = pts_gps[ide][0] - pts_gps[i][0];
			double dye = pts_gps[ide][1] - pts_gps[i][1];
			double angle = acos((dxs * dxe + dys * dye) / sqrt(dxs * dxs + dys * dys + 0.1) / sqrt(dxe * dxe + dye * dye + 0.1));
			angle = abs(angle - CV_PI);
			if (angle >= CV_PI * 80.0 / 180.0) {
				angle = CV_PI * 80.0 / 180.0;
			}
			weight[i] = tan(angle);
			std::cout << weight[i] << std::endl;
		}

		Eigen::Matrix3d R;
		Eigen::Vector3d t;
		double scale = 0.0;
		double err = 0.0;
		SimilarityTransformation(pts_cam, pts_gps, weight, R, t, scale, err);
		//std::cout << R << std::endl;
		//std::cout << t << std::endl;
		//std::cout << scale << std::endl;
		//std::cout << err << std::endl;

		// convert the coordinate of all the cams and points
		for (size_t i = 0; i < cams_.size(); i++) {
			cams_[i]->Transformation(R, t, scale);
		}
		for (size_t i = 0; i < pts_.size(); i++) {
			//pts_[i]->Transformation(R, t, scale);
		}
		for (size_t i = 0; i < pts_new_.size(); i++)
		{
			//pts_new_[i]->Transformation(R, t, scale);
		}

		// subtract the offset
		gps_offset_ = Eigen::Vector3d(0.0, 0.0, 0.0);
		for (size_t i = 0; i < cams_.size(); i++) {
			gps_offset_ += cams_[i]->pos_ac_.c;
		}
		gps_offset_ /= cams_.size();

		Eigen::Matrix3d R_eye = Eigen::MatrixXd::Identity(3, 3);
		for (size_t i = 0; i < cams_.size(); i++) {
			cams_[i]->Transformation(R_eye, -gps_offset_, 1.0);
		}
		for (size_t i = 0; i < pts_.size(); i++) {
			//pts_[i]->Transformation(R_eye, -gps_offset_, 1.0);
		}
		for (size_t i = 0; i < pts_new_.size(); i++)
		{
			//pts_new_[i]->Transformation(R_eye, -gps_offset_, 1.0);
		}
		for (size_t i = 0; i < cams_gps_.size(); i++)
		{
			cams_gps_[i].x -= gps_offset_[0];
			cams_gps_[i].y -= gps_offset_[1];
			cams_gps_[i].z -= gps_offset_[2];
		}
	}

	void SLAMGPS::AbsoluteOrientationWithGPSLocal(int range, std::vector<Eigen::Matrix3d> &Rs_local, std::vector<Eigen::Vector3d> &ts_local)
	{
		//std::vector<Eigen::Matrix3d> Rs_local;
		//std::vector<Eigen::Vector3d> ts_local;
		double scale = 1.0;
		for (size_t k = 0; k < cams_.size(); k++)
		{
			//std::cout << k << std::endl;
			std::vector<Eigen::Vector3d> pts_cam_local;
			std::vector<Eigen::Vector3d> pts_gps_local;
			std::vector<double> weight_local;
			Eigen::Vector3d pt_cam_ref = scale * cams_[k]->pos_ac_.c;
			int step = range;
			int idxs = std::max(0, int(k - step));
			int idxe = std::min(int(cams_.size()), int(k + step));

			for (size_t i = idxs; i < idxe; i++)
			{
				pts_cam_local.push_back(scale * Eigen::Vector3d(cams_[i]->pos_ac_.c));
				pts_gps_local.push_back(Eigen::Vector3d(cams_gps_[i].x, cams_gps_[i].y, cams_gps_[i].z));

				double dis = std::abs(int(i - k));
				weight_local.push_back(1.0 / (dis + 1.0));
				//weight_local[i] = 1.0;
			}

			for (size_t i = 0; i < cams_.size(); i++)
			{
				pts_cam_local.push_back(scale * Eigen::Vector3d(cams_[i]->pos_ac_.c));
				pts_gps_local.push_back(Eigen::Vector3d(cams_gps_[i].x, cams_gps_[i].y, cams_gps_[i].z));

				weight_local.push_back(0.001);
			}

			Eigen::Matrix3d R_local;
			Eigen::Vector3d t_local;
			double err_local = 0.0;
			double scale_local = 0.0;
			RigidTransformation(pts_cam_local, pts_gps_local, weight_local, R_local, t_local, err_local);
			//SimilarityTransformation(pts_cam_local, pts_gps_local, weight_local, R_local, t_local, scale_local, err_local);

			// convert the coordinate of all the cams and points
			Rs_local.push_back(R_local);
			ts_local.push_back(t_local);
		}
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

	void SLAMGPS::WriteGPSPose(std::string file)
	{
		std::ofstream ff(file);
		ff << std::fixed << std::setprecision(8);
		for (size_t i = 0; i < cams_gps_.size(); i++)
		{
			double x = cams_gps_[i].x;
			double y = cams_gps_[i].y;
			double z = cams_gps_[i].z;
			ff << x << " " << y << " " << z << " 0 0 255" << std::endl;
		}

		for (size_t i = 0; i < cams_.size(); i++)
		{
			double x = cams_[i]->pos_ac_.c[0];
			double y = cams_[i]->pos_ac_.c[1];
			double z = cams_[i]->pos_ac_.c[2];
			ff << x << " " << y << " " << z << " 255 0 0" << std::endl;
		}
		ff.close();
	}

	void SLAMGPS::WriteOffset(std::string file)
	{
		std::ofstream ff(file);
		ff << std::fixed << std::setprecision(8);
		ff << gps_offset_[0] << " " << gps_offset_[1] << " " << gps_offset_[2] << std::endl;
		ff.close();
	}

	void SLAMGPS::Convert2GPS(std::string fold)
	{
		std::string file_pts = fold + "\\cam_pts4.txt";
		std::vector<cv::Point3d> pts;
		std::vector<cv::Point3i> colors;
		std::ifstream ff(file_pts);
		double x, y, z;
		int r, g, b;
		while (!ff.eof())
		{
			ff >> x >> y >> z >> r >> g >> b;
			pts.push_back(cv::Point3d(x, y, z));
			colors.push_back(cv::Point3i(r, g, b));
		}
		ff.close();

		//
		std::string file_offset = fold + "\\offset.txt";
		double offx, offy, offz;
		std::ifstream fff(file_offset);
		fff >> offx >> offy >> offz;

		// write out
		std::string file_pts_gps = fold + "\\cam_pts5.txt";
		std::ofstream fout(file_pts_gps);
		fout << std::fixed << std::setprecision(8);
		for (size_t i = 0; i < pts.size(); i++)
		{
			fout << pts[i].x + offx << " " << pts[i].y + offy << " " << pts[i].z + offz << " ";
			fout << colors[i].x << " " << colors[i].y << " " << colors[i].z << std::endl;
		}
		fout.close();

	}

	void SLAMGPS::Convert2GPSDense(std::string fold)
	{
		std::string file_pts = fold + "\\dense_all.txt";
		std::vector<cv::Point3d> pts;
		std::vector<cv::Point3i> colors;
		std::vector<cv::Point3d> normals;
		std::ifstream ff(file_pts);
		double x, y, z;
		int r, g, b;
		double nx, ny, nz;
		while (!ff.eof())
		{
			ff >> x >> y >> z >> r >> g >> b >> nx >> ny >> nz;
			pts.push_back(cv::Point3d(x, y, z));
			colors.push_back(cv::Point3i(r, g, b));
			normals.push_back(cv::Point3d(nx, ny, nz));
		}
		ff.close();

		//
		std::string file_offset = fold + "\\offset.txt";
		double offx, offy, offz;
		std::ifstream fff(file_offset);
		fff >> offx >> offy >> offz;

		// write out
		std::string file_pts_gps = fold + "\\dense_all_gps.txt";
		std::ofstream fout(file_pts_gps);
		fout << std::fixed << std::setprecision(8);
		for (size_t i = 0; i < pts.size(); i++)
		{
			fout << pts[i].x + offx << " " << pts[i].y + offy << " " << pts[i].z + offz << " ";
			fout << colors[i].x << " " << colors[i].y << " " << colors[i].z << " ";
			fout << normals[i].x << " " << normals[i].y << " " << normals[i].z << std::endl;
		}
		fout.close();
	}

}  // namespace objectsfm
