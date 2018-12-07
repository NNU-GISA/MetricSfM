
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

#include "sfm_incremental.h"

#include <fstream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "utils/basic_funcs.h"
#include "utils/find_polynomial_roots_companion_matrix.h"
#include "orientation/relative_pose_estimation.h"
#include "orientation/absolute_pose_estimation.h"

namespace objectsfm {

	IncrementalSfM::IncrementalSfM()
	{
		found_seed_ = false;
		num_reconstructions_ = 0;
	}

	IncrementalSfM::~IncrementalSfM()
	{
	}

	void IncrementalSfM::InitializeSystem()
	{
		// database
		db_.input_fold_ = options_.input_fold;
		db_.output_fold_ = options_.output_fold;
		db_.options.feature_type = options_.feature_type;
		db_.options.resize_image = options_.resize_image;
		db_.options.feature_type = options_.feature_type;
		if (!db_.FeatureExtraction()) {
			std::cout << "Error with the database" << std::endl;
		}
		is_img_processed_ = std::vector<bool>(db_.num_imgs_, false);
		localize_fail_times_ = std::vector<int>(db_.num_imgs_, 0);

		// graph
		graph_.options_.matching_type = options_.matching_type;
		graph_.options_.priori_type = options_.priori_type;
		graph_.options_.priori_file = options_.priori_file;
		graph_.AssociateDatabase(&db_);
		if (!graph_.BuildGraph()) {
			std::cout << "Error with the graph" << std::endl;
		}

		bundle_full_options_.max_num_iterations = options_.th_max_iteration_full_bundle;
		bundle_full_options_.minimizer_progress_to_stdout = options_.minimizer_progress_to_stdout;
		bundle_partial_options_.max_num_iterations = options_.th_max_iteration_partial_bundle;
		bundle_partial_options_.minimizer_progress_to_stdout = options_.minimizer_progress_to_stdout;
	}

	void IncrementalSfM::Run()
	{
		bool use_temp = false;
		bool is_temp_used = false;

		// load matching graph
		graph_.ReadinMatchingGraph();

		// Try to add as many views as possible to the reconstruction until no more views can be localized.
		while (true)
		{
			int images_added_total = 0;
			found_seed_ = false;
			if (use_temp && !is_temp_used)
			{
				images_added_total = 31;
				std::string path_temp_in = db_.output_fold_ + "\\temp_result" + std::to_string(images_added_total) + ".txt";
				ReadTempResultIn(path_temp_in);
				found_seed_ = true;
				is_temp_used = true;
			}

			// find seed pair
			if (!found_seed_)
			{
				if (!FindSeedPairThenReconstruct())
				{
					break;
				}
			}

			// try to find as many views as possible to localize
			while (true)
			{
				std::cout << std::endl << std::endl;

				std::vector<int> image_ids;
				std::vector<std::vector<std::pair<int, int>>> corres_2d3d;
				std::vector<std::vector<int>> visible_cams;
				FindImageToLocalize(image_ids, corres_2d3d, visible_cams);
				if (!image_ids.size()) {
					break;
				}
				found_seed_ = true;

				// try to localize a new image
				bool localize_successful = false;
				int i = 0;
				for (i = 0; i < image_ids.size(); ++i)
				{
					if (corres_2d3d[i].size() < options_.th_min_2d3d_corres)
					{
						continue;
					}
					// only localize with the stable 2d-3d correspndences
					if (!LocalizeImage(image_ids[i], corres_2d3d[i], visible_cams[i]))
					{
						continue;
					}
					localize_successful = true;
					break;
				}
				if (!localize_successful)
				{
					std::cout << "LocalizeImage Fail" << std::endl;
					break;
				}

				// generate new 3d points before optimization
				GenerateNew3DPoints();

				//std::string path1 = db_.output_fold_ + "//pts_test.txt";
				//WriteCameraPointsOut(path1);

				// do partial bundle adjustment for the added camera and all its visible cameras
				int idx_new_cam = cams_.size() - 1;
				//PartialBundleAdjustment(idx_new_cam);
				is_img_processed_[cams_[idx_new_cam]->id_img_] = true;
				img_cam_map_.insert(std::pair<int, int>(cams_[idx_new_cam]->id_img_, idx_new_cam));

				// do full bundle adjustment if several images has been added
				images_added_total++;
				if (images_added_total % options_.th_step_full_bundle_adjustment == 0)
				{
					FullBundleAdjustment();
				}
				std::cout << cams_[0]->cam_model_->f_ << std::endl;
				std::cout << cams_[0]->cam_model_->k1_ << std::endl;
				std::cout << cams_[0]->cam_model_->k2_ << std::endl;

				// remove new added outliers
				RemovePointOutliers();

				// update graph information
				for (size_t i = 0; i < cams_.size(); i++)
				{
					is_img_processed_[cams_[i]->id_img_] = 1;
				}
				std::cout << "Cameras " << cams_.size() << "   Models " << cam_models_.size() << std::endl;

				// write out
				std::string path = db_.output_fold_ + "//pts" + std::to_string(images_added_total) + ".txt";
				WriteCameraPointsOut(path);

				if (images_added_total%10 == 0)
				//if (images_added_total >= 31)
				{
					std::string path_temp = db_.output_fold_ + "//temp_result" + std::to_string(images_added_total) + ".txt";
					WriteTempResultOut(path_temp);
					//exit(0);
				}
			}
			
			// save the model
			SaveModel();
		}
		std::cout << "----------Finish Reconstruction" << std::endl;
	}

	bool IncrementalSfM::FindSeedPairThenReconstruct()
	{
		// find seed pair hypotheses
		std::vector<std::pair<int, int>> seed_pair_hyps;
		SortImagePairs(seed_pair_hyps);
		if (!seed_pair_hyps.size())
		{
			return false;
		}

		// try to validate each hypothesis
		for (size_t i = 0; i < seed_pair_hyps.size(); i++)
		{
			int id_img1 = seed_pair_hyps[i].first;
			int id_img2 = seed_pair_hyps[i].second;

			//id_img1 = 142; // 16, 17    7 8    124 317
			//id_img2 = 143; // 52, 53

			std::cout << id_img1 << " " << id_img2 << std::endl;

			// load image features
			db_.ReadinImageFeatures(id_img1);
			db_.ReadinImageFeatures(id_img2);

			// load correspondences
			std::vector<std::pair<int, int>> matches;
			graph_.QueryMatch(id_img1, id_img2, matches);

			// initial camera
			cams_.resize(2);
			cams_[0] = new Camera;
			cams_[1] = new Camera;
			cams_[0]->AssociateImage(id_img1);
			cams_[1]->AssociateImage(id_img2);
			if (!CameraAssociateCameraModel(cams_[0]))
			{
				CameraModel* cam_model_new = new CameraModel(0,
					db_.image_infos_[id_img1]->rows,
					db_.image_infos_[id_img1]->cols,
					db_.image_infos_[id_img1]->f_mm,
					//db_.image_infos_[id_img1].f_pixel,
					0.0,
					db_.image_infos_[id_img1]->cam_maker,
					db_.image_infos_[id_img1]->cam_model);
				cams_[0]->AssociateCamereModel(cam_model_new);
				cam_models_.push_back(cam_model_new);
			}
			if (!CameraAssociateCameraModel(cams_[1]))
			{
				CameraModel* cam_model_new = new CameraModel(1,
					db_.image_infos_[id_img2]->rows,
					db_.image_infos_[id_img2]->cols,
					db_.image_infos_[id_img2]->f_mm,
					//db_.image_infos_[id_img2].f_pixel,
					0.0,
					db_.image_infos_[id_img2]->cam_maker,
					db_.image_infos_[id_img2]->cam_model);
				cams_[1]->AssociateCamereModel(cam_model_new);
				cam_models_.push_back(cam_model_new);
			}
			cams_[0]->cam_model_->AddCamera(0);
			cams_[1]->cam_model_->AddCamera(1);

			// set the pose of the first camera as [I|0]
			cams_[0]->SetRTPose(Eigen::Matrix3d::Identity(3, 3), Eigen::Vector3d::Zero());

			// recover the pose of the second camera
			std::vector<Eigen::Vector2d> pts1, pts2;
			for (int j = 0; j < matches.size(); ++j)
			{
				int id_pt1 = matches[j].first;
				int id_pt2 = matches[j].second;
				double x1 = db_.keypoints_[id_img1]->pts[id_pt1].pt.x;
				double y1 = db_.keypoints_[id_img1]->pts[id_pt1].pt.y;
				double x2 = db_.keypoints_[id_img2]->pts[id_pt2].pt.x;
				double y2 = db_.keypoints_[id_img2]->pts[id_pt2].pt.y;
				pts1.push_back(Eigen::Vector2d(x1, y1));
				pts2.push_back(Eigen::Vector2d(x2, y2));
			}

			RTPoseRelative rt_pose_21;
			if (cams_[0]->cam_model_->f_ && cams_[1]->cam_model_->f_)
			{
				if (!RelativePoseEstimation::RelativePoseWithFocalLength(pts1, pts2,
					cams_[0]->cam_model_->f_, cams_[1]->cam_model_->f_,
					rt_pose_21))
				{
					continue;
				}
			}
			else
			{
				double f1 = 0.0, f2 = 0.0;
				if (!RelativePoseEstimation::RelativePoseWithoutFocalLength(pts1, pts2, f1, f2, rt_pose_21))
				{
					continue;
				}

				if (cams_[0]->cam_model_->id_ == cams_[1]->cam_model_->id_)
				{
					cams_[0]->SetFocalLength((f1 + f2) / 2.0);
				}
				else
				{
					cams_[0]->SetFocalLength(f1);
					cams_[1]->SetFocalLength(f2);
				}
			}
			cams_[1]->SetRTPose(cams_[0]->pos_rt_, rt_pose_21);

			// draw
			//cv::Mat image1 = cv::imread(db_.image_paths_[id_img1]);
			//cv::Mat image2 = cv::imread(db_.image_paths_[id_img2]);
			//int pitch = 128;
			//cv::resize(image1, image1, cv::Size((image1.cols / pitch + 1) * pitch, (image1.rows / pitch + 1) * pitch));
			//cv::resize(image2, image2, cv::Size((image2.cols / pitch + 1) * pitch, (image2.rows / pitch + 1) * pitch));

			// initial 3D structure via triangulation
			for (size_t j = 0; j < matches.size(); j++)
			{
				int id_pt1_local = matches[j].first;
				int id_pt2_local = matches[j].second;

				int id_pt1_global = id_pt1_local + id_img1 * options_.idx_max_per_image;
				int id_pt2_global = id_pt2_local + id_img2 * options_.idx_max_per_image;

				Point3D *pt_temp = new Point3D;
				pt_temp->SetID(pts_.size());
				pt_temp->AddObservation(cams_[0],
					db_.keypoints_[id_img1]->pts[id_pt1_local].pt.x,
					db_.keypoints_[id_img1]->pts[id_pt1_local].pt.y,
					id_pt1_global);
				pt_temp->AddObservation(cams_[1],
					db_.keypoints_[id_img2]->pts[id_pt2_local].pt.x,
					db_.keypoints_[id_img2]->pts[id_pt2_local].pt.y,
					id_pt2_global);

				//cv::Point2f offset1(db_.image_infos_[id_img1]->cols / 2.0, db_.image_infos_[id_img1]->rows / 2.0);
				//cv::Point2f offset2(db_.image_infos_[id_img2]->cols / 2.0, db_.image_infos_[id_img2]->rows / 2.0);
				//cv::line(image1, db_.keypoints_[id_img1]->pts[id_pt1_local].pt + offset1,
				//	db_.keypoints_[id_img2]->pts[id_pt2_local].pt + offset2, cv::Scalar(0,0,255), 1);

				if (pt_temp->Trianglate2(options_.th_mse_reprojection, options_.th_angle_small))
				{
					pts_.push_back(pt_temp);
					cams_[0]->AddPoints(pt_temp, id_pt1_global);
					cams_[1]->AddPoints(pt_temp, id_pt2_global);
				}
			}
			//cv::imwrite("F:\\result.jpg", image1);
			cv::Mat img_match;
			Drawmatch(id_img1, id_img2, img_match);
			cv::imwrite("F:\\img_match_" + std::to_string(id_img1) + "_" + std::to_string(id_img2) + ".jpg", img_match);

			db_.ReleaseImageFeatures(id_img1);
			db_.ReleaseImageFeatures(id_img2);

			if (pts_.size() < options_.th_seedpair_structures
				|| pts_.size() < matches.size() / 5)
			{
				memory::ReleasePointerVector(pts_);
				memory::ReleasePointerVector(cams_);
				memory::ReleasePointerVector(cam_models_);
				pts_.clear();
				cams_.clear();
				cam_models_.clear();
				continue;
			}

			//std::cout << cams_[0]->id_img_ << std::endl;
			//std::cout << cams_[1]->id_img_ << std::endl;
			//exit(0);

			// perform fully bundle adjustment for the seed pair
			FullBundleAdjustment();

			// remove new added outliers
			RemovePointOutliers();

			std::string path_seed = db_.output_fold_ + "//cams_pts_seed.txt";
			WriteCameraPointsOut(path_seed);

			cams_[0]->visible_cams_.push_back(0);
			cams_[0]->visible_cams_.push_back(1);
			cams_[1]->visible_cams_.push_back(1);
			cams_[1]->visible_cams_.push_back(0);
			is_img_processed_[cams_[0]->id_img_] = true;
			is_img_processed_[cams_[1]->id_img_] = true;
			img_cam_map_.insert(std::pair<int, int>(cams_[0]->id_img_, 0));
			img_cam_map_.insert(std::pair<int, int>(cams_[1]->id_img_, 1));

			return true;

		}

		return false;
	}

	void IncrementalSfM::FindImageToLocalize(std::vector<int> &image_ids, std::vector<std::vector<std::pair<int, int>>> &corres_2d3d,
		std::vector<std::vector<int>> &visible_cams)
	{
		int num_imgs = db_.num_imgs_;

		// find the images that have matches with current cameras
		for (int i = 0; i < cams_.size(); ++i)
		{
			int id_img = cams_[i]->id_img_;
			for (int j = 0; j < num_imgs; ++j)
			{
				if (graph_.match_graph_[id_img*num_imgs+j] > 0 && !is_img_processed_[j] && localize_fail_times_[j] < options_.th_max_failure_localization)
				{
					image_ids.push_back(j);
				}
			}
		}
		if (image_ids.empty())
		{
			return;
		}
		math::unique_vector(image_ids);

		// find the 2d-3d correspondences
		std::vector<int> no_corres(image_ids.size(), 0);
		corres_2d3d.resize(image_ids.size());
		visible_cams.resize(image_ids.size());
		for (size_t i = 0; i < image_ids.size(); i++)
		{
			int id_img_i = image_ids[i];

			std::vector<int> visible_cams_i;
			std::map<int, int> corres_2d3d_i;    // first, idx of 2d keypoints; second, idx of 3d points
			std::map<int, double> corres_2d3d_info_i;

			for (int j = 0; j < num_imgs; ++j)
			{
				int id_img_j = j;

				if (graph_.match_graph_[id_img_i*num_imgs+id_img_j] > 0 && is_img_processed_[id_img_j])
				{
					// find the reference camera
					std::map<int, int >::iterator iter_i_c = img_cam_map_.find(id_img_j);
					if (iter_i_c == img_cam_map_.end())
					{
						continue;
					}
					int idx_cam = iter_i_c->second;

					// find the 2d-3d correspondences			
					std::vector<std::pair<int, int>> corres_2d2d_ij;
					graph_.QueryMatch(id_img_i, id_img_j, corres_2d2d_ij);
					if (!corres_2d2d_ij.size())
					{
						continue;
					}

					// 
					int count_2d3d_ij = 0;
					int idx_i_local, idx_j_local, idx_j_global;
					int count_temp = 0;
					for (size_t m = 0; m < corres_2d2d_ij.size(); m++)
					{
						idx_i_local = corres_2d2d_ij[m].first;
						idx_j_local = corres_2d2d_ij[m].second;
						idx_j_global = idx_j_local + options_.idx_max_per_image*id_img_j;

						std::map<int, Point3D*>::iterator iter_cam_3dpt = cams_[idx_cam]->pts_.find(idx_j_global);
						if (iter_cam_3dpt != cams_[idx_cam]->pts_.end() && (!iter_cam_3dpt->second->is_bad_estimated_))
						{
							corres_2d3d_i.insert(std::pair<int, int>(idx_i_local, iter_cam_3dpt->second->id_));

							double mse_3dpt = iter_cam_3dpt->second->mse_;
							if (iter_cam_3dpt->second->cams_.size() <= 2)
							{
								mse_3dpt += 3.0;
							}
							corres_2d3d_info_i.insert(std::pair<int, double>(idx_i_local, mse_3dpt));
							count_2d3d_ij++;

							count_temp++;
						}
					}

					if (count_2d3d_ij > 5)
					{
						visible_cams_i.push_back(idx_cam);
					}
				}
			}

			//
			if (!corres_2d3d_i.size())
			{
				no_corres[i] = 1;
				continue;
			}

			// sort according to the mse
			int count = 0;
			std::vector<std::pair<int, double>> corres_2d3d_info_sorted_i;
			for (auto itr = corres_2d3d_info_i.begin(); itr != corres_2d3d_info_i.end(); ++itr)
			{
				corres_2d3d_info_sorted_i.push_back(*itr);
			}
			std::sort(corres_2d3d_info_sorted_i.begin(), corres_2d3d_info_sorted_i.end(), [](const std::pair<int, double> &lhs, const std::pair<int, double> &rhs) { return lhs.second < rhs.second; });

			std::vector<std::pair<int, int>> corres_2d3d_sorted_i;
			for (size_t m = 0; m < corres_2d3d_info_sorted_i.size(); m++)
			{
				int key_m = corres_2d3d_info_sorted_i[m].first;
				corres_2d3d_sorted_i.push_back(std::pair<int, int>(key_m, corres_2d3d_i[key_m]));
			}

			corres_2d3d[i] = corres_2d3d_sorted_i;
			visible_cams[i] = visible_cams_i;
		}

		// sort according to the number of 2d-3d correspondings and localize failed times
		std::vector<std::pair<int, double>> idx_num(image_ids.size());
		for (size_t i = 0; i < image_ids.size(); i++)
		{
			int id_img = image_ids[i];
			idx_num[i].first = i;
			idx_num[i].second = corres_2d3d[i].size() / (5 + localize_fail_times_[id_img]);
		}
		std::sort(idx_num.begin(), idx_num.end(), [](const std::pair<int, double> &lhs, const std::pair<int, double> &rhs) { return lhs.second > rhs.second; });

		std::vector<int> image_ids_sort;
		std::vector<std::vector<std::pair<int, int>>> corres_2d3d_sort;
		std::vector<std::vector<int>> visible_cams_sort;
		for (size_t i = 0; i < idx_num.size(); i++)
		{
			if (idx_num[i].second > 0)
			{
				int idx = idx_num[i].first;
				image_ids_sort.push_back(image_ids[idx]);
				corres_2d3d_sort.push_back(corres_2d3d[idx]);
				visible_cams_sort.push_back(visible_cams[idx]);
			}
		}
		image_ids = image_ids_sort;
		corres_2d3d = corres_2d3d_sort;
		visible_cams = visible_cams_sort;
	}

	bool IncrementalSfM::LocalizeImage(int id_img, std::vector<std::pair<int, int>> &corres_2d3d, std::vector<int> &visible_cams)
	{
		if (corres_2d3d.size() < 3)
		{
			return false;
		}

		// load image data
		db_.ReadinImageFeatures(id_img);

		Camera* cam_new = new Camera;
		cam_new->AssociateImage(id_img);

		if (!CameraAssociateCameraModel(cam_new))
		{
			CameraModel* cam_model_new = new CameraModel(cam_models_.size(),
				db_.image_infos_[id_img]->rows,
				db_.image_infos_[id_img]->cols,
				db_.image_infos_[id_img]->f_mm,
				//db_.image_infos_[id_img].f_pixel,
				0.0,
				db_.image_infos_[id_img]->cam_maker,
				db_.image_infos_[id_img]->cam_model);
			cam_new->AssociateCamereModel(cam_model_new);
		}

		// estimate the pose of the image via pnp algorithm
		std::vector<Eigen::Vector3d> pts_w(corres_2d3d.size());
		std::vector<Eigen::Vector2d> pts_2d(corres_2d3d.size());
		for (int i = 0; i < corres_2d3d.size(); ++i)
		{
			int idx_2d = corres_2d3d[i].first;
			int idx_3d = corres_2d3d[i].second;
			pts_2d[i] = Eigen::Vector2d(db_.keypoints_[id_img]->pts[idx_2d].pt.x, db_.keypoints_[id_img]->pts[idx_2d].pt.y);
			pts_w[i] = Eigen::Vector3d(pts_[idx_3d]->data[0], pts_[idx_3d]->data[1], pts_[idx_3d]->data[2]);
		}


		/*
		// draw
		cv::Mat img_draw = cv::imread(db_.image_paths_[id_img]);
		std::string path_temp = db_.output_fold_ + "//3dpts.txt";
		std::ofstream ofs(path_temp);

		for (size_t i = 0; i < corres_2d3d.size(); i++)
		{
			int idx_2d = corres_2d3d[i].first;
			int idx_3d = corres_2d3d[i].second;
			double x = db_.keypoints_[id_img][idx_2d].pt.x + db_.image_infos_[id_img].cols / 2.0;
			double y = db_.keypoints_[id_img][idx_2d].pt.y + db_.image_infos_[id_img].rows / 2.0;

			if (i < 100)
			{
				cv::circle(img_draw, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), 2);
				ofs << pts_[idx_3d]->data[0] << " "
					<< pts_[idx_3d]->data[1] << " "
					<< pts_[idx_3d]->data[2] << " "
					<< 255 << " "
					<< 0 << " "
					<< 0 << std::endl;
			}
			else
			{
				cv::circle(img_draw, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), 2);
				ofs << pts_[idx_3d]->data[0] << " "
					<< pts_[idx_3d]->data[1] << " "
					<< pts_[idx_3d]->data[2] << " "
					<< 0 << " "
					<< 0 << " "
					<< 255 << std::endl;
			}
		}
		ofs.close();
		std::string out_draw = db_.output_fold_ + "//2dpts.jpg";
		cv::imwrite(out_draw, img_draw);
		*/

		RTPose pose_absolute;
		std::vector<double> error_reproj;
		double avg_error = 0.0;
		if (cam_new->cam_model_->f_)
		{
			bool isOK = AbsolutePoseEstimation::AbsolutePoseWithFocalLength(pts_w, pts_2d,
				cam_new->cam_model_, pose_absolute, error_reproj, avg_error, options_.th_mse_localization);
			if (!isOK)
			{
				localize_fail_times_[id_img]++;

				std::cout << "111 Failure " << id_img << " times " << localize_fail_times_[id_img] << std::endl;

				// write out pts and cams
				std::string path1= db_.output_fold_ + "//all_pts_1.txt";
				WriteCameraPointsOut(path1);

				// write out for debugging
				std::string path = db_.output_fold_ + "//test_1.txt";
				std::ofstream ofs(path);
				ofs << cam_new->cam_model_->f_ << std::endl;
				ofs << pts_w.size() << std::endl;
				for (size_t i = 0; i < pts_w.size(); i++)
				{
					ofs << pts_w[i](0) << " " << pts_w[i](1) << " " << pts_w[i](2)
						<< " " << pts_2d[i](0) << " " << pts_2d[i](1) << std::endl;
				}
				ofs.close();

				return false;
			}
		}
		else
		{
			bool isOK = AbsolutePoseEstimation::AbsolutePoseWithoutFocalLength(pts_w, pts_2d,
				cam_new->cam_model_, pose_absolute, error_reproj, avg_error, options_.th_mse_localization);
			if (!isOK)
			{
				localize_fail_times_[id_img]++;

				std::cout << "222 Failure " << id_img << " times " << localize_fail_times_[id_img] << std::endl;

				// write out pts and cams
				std::string path1 = db_.output_fold_ + "//all_pts_2.txt";
				WriteCameraPointsOut(path1);

				// write out for debugging
				std::string path = db_.output_fold_ + "//test_2.txt";
				std::ofstream ofs(path);
				ofs << cam_new->cam_model_->f_ << std::endl;
				ofs << pts_w.size() << std::endl;
				for (size_t i = 0; i < pts_w.size(); i++)
				{
					ofs << pts_w[i](0) << " " << pts_w[i](1) << " " << pts_w[i](2)
						<< " " << pts_2d[i](0) << " " << pts_2d[i](1) << std::endl;
				}
				ofs.close();

				return false;
			}
		}
		cam_new->SetRTPose(pose_absolute.R, pose_absolute.t);

		// add new observations on the new image to the corresponding 3D points
		int count_inliers = 0;
		for (size_t i = 0; i < corres_2d3d.size(); i++)
		{
			int idx_3d = corres_2d3d[i].second;

			if (error_reproj[i] > avg_error)
			{
				pts_[idx_3d]->is_bad_estimated_ = true;
				//pts_[idx_3d]->ReleaseAll();
				continue;
			}

			int idx_2d_local = corres_2d3d[i].first;
			int idx_2d_global = idx_2d_local + cam_new->id_img_*options_.idx_max_per_image;
			if (!pts_[idx_3d]->is_new_added_)
			{
				pts_[idx_3d]->AddObservation(cam_new, pts_2d[i](0), pts_2d[i](1), idx_2d_global);
				pts_[idx_3d]->is_new_added_ = true;
				cam_new->AddPoints(pts_[idx_3d], idx_2d_global);

				count_inliers++;
			}
		}
		std::cout << "----------Localization: inliers: " << count_inliers << " out of " << corres_2d3d.size() << std::endl;

		// update the cameras
		if (cam_new->cam_model_->num_cams_ == 0)
		{
			cam_models_.push_back(cam_new->cam_model_);
		}
		cam_new->cam_model_->AddCamera(cams_.size());
		cams_.push_back(cam_new);

		// optimize the pose of the new camera via bundle adjustment
		if (cam_new->cam_model_->num_cams_ == 1)
		{
			SingleBundleAdjustment(cams_.size() - 1);
		}

		// update the visible graph of camera
		UpdateVisibleGraph(cams_.size() - 1, visible_cams);

		db_.ReleaseImageFeatures(id_img);

		return true;
	}

	void IncrementalSfM::GenerateNew3DPoints()
	{
		// find new match with visible cams
		int idx_cam_1 = cams_.size() - 1;
		int id_img_1 = cams_[idx_cam_1]->id_img_;
		db_.ReadinImageFeatures(id_img_1);

		int num_init = cams_[idx_cam_1]->pts_.size();

		int count_all_pts = pts_.size();
		std::vector<std::pair<Point3DNew*, double>> pts_new;
		for (size_t i = 0; i < cams_[idx_cam_1]->visible_cams_.size(); i++)
		{
			int idx_cam_2 = cams_[idx_cam_1]->visible_cams_[i];
			if (idx_cam_2 == idx_cam_1)
			{
				continue;
			}
			int id_img_2 = cams_[idx_cam_2]->id_img_;
			db_.ReadinImageFeatures(id_img_2);

			std::vector<std::pair<int, int>> matches;
			graph_.QueryMatch(id_img_1, id_img_2, matches);

			// determine the triangulation angle
			double th_tri_angle = options_.th_angle_small;
			if (matches.size() > 500) {
				th_tri_angle = options_.th_angle_large;
			}

			
			std::vector<std::pair<Point3DNew*, double>> pts_new_temp;
			
			// draw
			//cv::Mat img1 = cv::imread(db_.image_paths_[id_img_1]);
			//cv::Mat img2 = cv::imread(db_.image_paths_[id_img_2]);
			//cv::Point2f offset1(db_.image_infos_[id_img_1].cols / 2.0, db_.image_infos_[id_img_1].rows / 2.0);
			//cv::Point2f offset2(db_.image_infos_[id_img_2].cols / 2.0, db_.image_infos_[id_img_2].rows / 2.0);

			for (size_t j = 0; j < matches.size(); j++)
			{
				int id_pt1_local = matches[j].first;
				int id_pt2_local = matches[j].second;

				int id_pt1_global = id_pt1_local + id_img_1 * options_.idx_max_per_image;
				int id_pt2_global = id_pt2_local + id_img_2 * options_.idx_max_per_image;

				// if the matches point has already be triangulated before, continue
				if (cams_[idx_cam_1]->pts_.find(id_pt1_global) != cams_[idx_cam_1]->pts_.end() ||
					cams_[idx_cam_2]->pts_.find(id_pt2_global) != cams_[idx_cam_2]->pts_.end())
				{
					continue;
				}

				Point3D *pt_temp = new Point3D;
				pt_temp->AddObservation(cams_[idx_cam_1],
					db_.keypoints_[id_img_1]->pts[id_pt1_local].pt.x,
					db_.keypoints_[id_img_1]->pts[id_pt1_local].pt.y,
					id_pt1_global);
				pt_temp->AddObservation(cams_[idx_cam_2],
					db_.keypoints_[id_img_2]->pts[id_pt2_local].pt.x,
					db_.keypoints_[id_img_2]->pts[id_pt2_local].pt.y,
					id_pt2_global);
				pt_temp->is_new_added_ = true;

				if (pt_temp->Trianglate2(options_.th_mse_reprojection, th_tri_angle))
				{
					Point3DNew *new_pt = new Point3DNew;
					new_pt->pt = pt_temp;
					new_pt->id_cam1 = idx_cam_1;
					new_pt->id_cam2 = idx_cam_2;
					new_pt->id_pt1 = id_pt1_global;
					new_pt->id_pt2 = id_pt2_global;
					pts_new_temp.push_back(std::pair<Point3DNew*,int>(new_pt, pt_temp->mse_));

					//cv::circle(img1, db_.keypoints_[id_img_1][id_pt1_local].pt + offset1, 2, cv::Scalar(255, 0, 0), 2);
					//cv::circle(img2, db_.keypoints_[id_img_2][id_pt2_local].pt + offset2, 2, cv::Scalar(255, 0, 0), 2);
				}
				else
				{
					//cv::circle(img1, db_.keypoints_[id_img_1][id_pt1_local].pt + offset1, 2, cv::Scalar(0, 0, 255), 2);
					//cv::circle(img2, db_.keypoints_[id_img_2][id_pt2_local].pt + offset2, 2, cv::Scalar(0, 0, 255), 2);
					delete pt_temp;
				}
			}
			//std::string path_temp1 = db_.output_fold_ + "//match1.jpg";
			//std::string path_temp2 = db_.output_fold_ + "//match2.jpg";
			//cv::imwrite(path_temp1, img1);
			//cv::imwrite(path_temp2, img2);

			/*
			// write out
			std::string path_temp = db_.output_fold_ + "//new_pts_temp.txt";
			std::ofstream ofs(path_temp);
			for (size_t j = 0; j < pts_new_temp.size(); j++)
			{
				int id1 = pts_new_temp[j].first->id_cam1;
				int id2 = pts_new_temp[j].first->id_cam2;
				ofs << pts_new_temp[j].first->pt->data[0] << " "
					<< pts_new_temp[j].first->pt->data[1] << " "
					<< pts_new_temp[j].first->pt->data[2] << " "
					<< 255 << " "
					<< 0 << " "
					<< 0 << std::endl;
			}

			double scale = 0.1;
			std::vector<int> cam_ids;
			cam_ids.push_back(idx_cam_1);
			cam_ids.push_back(idx_cam_2);
			for (size_t j = 0; j < cam_ids.size(); j++)
			{
				std::vector<Eigen::Vector3d> axis(3);
				axis[0] = cams_[j]->pos_rt_.R.inverse() * Eigen::Vector3d(1, 0, 0);
				axis[1] = cams_[j]->pos_rt_.R.inverse() * Eigen::Vector3d(0, 1, 0);
				axis[2] = cams_[j]->pos_rt_.R.inverse() * Eigen::Vector3d(0, 0, 1);

				std::vector<Eigen::Vector3d> cam_pts;
				GenerateCamera3D(cams_[j]->pos_ac_.c, axis, cams_[j]->cam_model_->f_,
					cams_[j]->cam_model_->w_, cams_[j]->cam_model_->h_, scale, cam_pts);

				for (size_t m = 0; m < cam_pts.size(); m++)
				{
					ofs << cam_pts[m](0) << " " << cam_pts[m](1) << " " << cam_pts[m](2)
						<< " " << 255 << " " << 0 << " " << 0 << std::endl;
				}
			}
			ofs.close();
			*/

			for (size_t j = 0; j < pts_new_temp.size(); j++)
			{
				pts_new_temp[j].first->pt->SetID(count_all_pts);
				pts_new.push_back(pts_new_temp[j]);
				count_all_pts++;
			}

			db_.ReleaseImageFeatures(id_img_2);
		}

		// sort
		std::sort(pts_new.begin(), pts_new.end(), [](const std::pair<Point3DNew*, double> &lhs, const std::pair<Point3DNew*, double> &rhs) { return lhs.second < rhs.second; });
		int max_num_new_pts = pts_new.size();
		for (size_t i = 0; i < max_num_new_pts; i++)
		{
			pts_new[i].first->pt->SetID(pts_.size());
			pts_.push_back(pts_new[i].first->pt);

			int id_cam1 = pts_new[i].first->id_cam1;
			int id_cam2 = pts_new[i].first->id_cam2;
			int id_pt1  = pts_new[i].first->id_pt1;
			int id_pt2  = pts_new[i].first->id_pt2;
			cams_[id_cam1]->AddPoints(pts_new[i].first->pt, id_pt1);
			cams_[id_cam2]->AddPoints(pts_new[i].first->pt, id_pt2);
		}
		
		db_.ReleaseImageFeatures(id_img_1);

		std::cout << "----------GenerateNew3DPoints: " << max_num_new_pts <<" init: "<<num_init << std::endl;
	}

	void IncrementalSfM::PartialBundleAdjustment(int idx)
	{
		ImmutableCamsPoints();

		// optimize only the new camera and new points	
		for (size_t i = 0; i < cams_[idx]->cam_model_->idx_cams_.size(); i++)
		{
			int idx_cam = cams_[idx]->cam_model_->idx_cams_[i];
			cams_[idx_cam]->SetMutable(true);
			for (auto iter = cams_[idx_cam]->pts_.begin(); iter != cams_[idx_cam]->pts_.end(); iter++)
			{
				if (!iter->second->is_bad_estimated_)
				{
					iter->second->SetMutable(true);
				}
			}
		}
		for (size_t i = 0; i < cams_[idx]->visible_cams_.size(); i++)
		{
			int idx_cam = cams_[idx]->visible_cams_[i];
			cams_[idx_cam]->SetMutable(true);
			for (auto iter = cams_[idx_cam]->pts_.begin(); iter != cams_[idx_cam]->pts_.end(); iter++)
			{
				if (!iter->second->is_bad_estimated_)
				{
					iter->second->SetMutable(true);
				}
			}
		}

		int count_cams = 0, count_pts = 0;
		for (size_t i = 0; i < cams_.size(); i++)
		{
			if (cams_[i]->is_mutable_)
			{
				count_cams++;
			}
		}
		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (pts_[i]->is_mutable_)
			{
				count_pts++;
			}
		}
		std::cout << "adjust cams: " << count_cams << std::endl;
		std::cout << "adjust pts: " << count_pts << std::endl;

		/*
		double count_2views = 0, count_3views = 0;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (pts_[i]->is_new_added_)
			{
				pts_[i]->SetMutable(true);
				count_pts++;

				if (pts_[i]->cams_.size() < 3)
				{
					count_2views++;
				}
				else
				{
					count_3views++;
				}
			}
		}

		std::cout << "-------------adjust pts: " << count_pts << " adjust cams: " << count_cams << std::endl;
		*/

		/*
		int count_cams = 0;
		int count_pts = 0;
		for (size_t i = 0; i < cams_[idx]->visible_cams_.size(); i++)
		{
			int visible_cam_idx = cams_[idx]->visible_cams_[i];
			cams_[visible_cam_idx]->SetMutable(true);
			count_cams++;

			std::map<int, Point3D*>::iterator iter = cams_[visible_cam_idx]->pts_.begin();
			while (iter != cams_[visible_cam_idx]->pts_.end())
			{
				iter->second->SetMutable(true);
				count_pts++;
				iter++;
			}
		}
		std::cout << "adjust cams: " << count_cams << std::endl;
		std::cout << "adjust pts: " << count_pts << std::endl;
		*/

		// run bundle adjustment 
		objectsfm::BundleAdjuster bundler(cams_, cam_models_, pts_);
		bundler.SetOptions(bundle_partial_options_);
		bundler.RunOptimizetion(!found_seed_, 2.0);
		bundler.UpdateParameters();
	}

	void IncrementalSfM::FullBundleAdjustment()
	{
		// mutable all the cameras and points
		MutableCamsPoints();

		// bundle adjustment 
		objectsfm::BundleAdjuster bundler(cams_, cam_models_, pts_);
		bundler.SetOptions(bundle_full_options_);
		bundler.RunOptimizetion(!found_seed_,1.0);
		bundler.UpdateParameters();
	}

	void IncrementalSfM::SingleBundleAdjustment(int idx)
	{
		ImmutableCamsPoints();

		cams_[idx]->SetMutable(true);

		//int count_cams = 0, count_pts = 0;
		//for (size_t i = 0; i < cams_[idx]->cam_model_->idx_cams_.size(); i++)
		//{
		//	int idx_cam = cams_[idx]->cam_model_->idx_cams_[i];
		//	cams_[idx_cam]->SetMutable(true);
		//	count_cams++;
		//}
		//for (size_t i = 0; i < cams_[idx]->visible_cams_.size(); i++)
		//{
		//	int visible_cam_idx = cams_[idx]->visible_cams_[i];
		//	cams_[visible_cam_idx]->SetMutable(true);
		//	count_cams++;
		//}

		// run bundle adjustment 
		objectsfm::BundleAdjuster bundler(cams_, cam_models_, pts_);
		bundler.SetOptions(bundle_partial_options_);
		bundler.RunOptimizetion(!found_seed_,1.0);
		bundler.UpdateParameters();
	}

	void IncrementalSfM::SaveModel()
	{
		std::string fold = db_.output_fold_ + "//" + std::to_string(num_reconstructions_);
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

		img_cam_map_.clear();
		found_seed_ = false;
		localize_fail_times_ = std::vector<int>(db_.num_imgs_, 0);

		num_reconstructions_++;
	}

	void IncrementalSfM::WriteCameraPointsOut(std::string path)
	{
		std::ofstream ofs(path);
		if (!ofs.is_open())
		{
			return;
		}

		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (pts_[i]->is_bad_estimated_)
			{
				continue;
			}
			ofs << pts_[i]->data[0] << " " << pts_[i]->data[1] << " " << pts_[i]->data[2] 
				<< " " << 0 << " " << 0 << " " << 0 << std::endl;
		}

		//double scale = (cams_[1]->pos_ac_.c- cams_[0]->pos_ac_.c).squaredNorm() / 100;
		double scale = 0.1;
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
				ofs << cam_pts[j](0) << " " << cam_pts[j](1) << " " << cam_pts[j](2)
					<< " " << 255 << " " << 0 << " " << 0 << std::endl;
			}
		}
		ofs.close();
	}

	void IncrementalSfM::WriteTempResultOut(std::string path)
	{
		std::ofstream ofs(path);
		if (!ofs.is_open())
		{
			return;
		}
		ofs.precision(20);

		// write out cam-models
		ofs << cam_models_.size() << std::endl;
		for (size_t i = 0; i < cam_models_.size(); i++)
		{
			ofs << cam_models_[i]->id_ << std::endl;

			ofs << cam_models_[i]->cam_maker_ << std::endl;
			ofs << cam_models_[i]->cam_model_ << std::endl;

			ofs << cam_models_[i]->w_ << " " << cam_models_[i]->h_ << std::endl;
			ofs << cam_models_[i]->f_mm_
				<< " " << cam_models_[i]->f_
				<< " " << cam_models_[i]->f_hyp_
				<< " " << cam_models_[i]->px_
				<< " " << cam_models_[i]->py_
				<< " " << cam_models_[i]->k1_
				<< " " << cam_models_[i]->k2_
				<< " " << cam_models_[i]->data[0]
				<< " " << cam_models_[i]->data[1]
				<< " " << cam_models_[i]->data[2]
				<< std::endl;

			ofs << cam_models_[i]->num_cams_ << std::endl;
		}

		// write out cams
		ofs << cams_.size() << std::endl;
		for (size_t i = 0; i < cams_.size(); i++)
		{
			ofs << cams_[i]->id_img_ << std::endl;
			ofs << cams_[i]->cam_model_->id_ << std::endl;
			ofs << cams_[i]->is_mutable_ << std::endl;

			ofs << " " << cams_[i]->data[0]
				<< " " << cams_[i]->data[1]
				<< " " << cams_[i]->data[2]
				<< " " << cams_[i]->data[3]
				<< " " << cams_[i]->data[4]
				<< " " << cams_[i]->data[5]
				<< std::endl;

			ofs << cams_[i]->pts_.size() << std::endl;
			for (auto iter = cams_[i]->pts_.begin(); iter != cams_[i]->pts_.end(); ++iter)
			{
				ofs << iter->first << " " << iter->second->id_ << " ";
			}
			ofs << std::endl;

			ofs << cams_[i]->visible_cams_.size() << std::endl;
			for (size_t j = 0; j < cams_[i]->visible_cams_.size(); j++)
			{
				ofs << cams_[i]->visible_cams_[j] << " ";
			}
			ofs << std::endl;
		}

		// write out points
		ofs << pts_.size() << std::endl;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			ofs << pts_[i]->id_ << std::endl;

			ofs << pts_[i]->is_mutable_
				<< " " << pts_[i]->is_bad_estimated_
				<< " " << pts_[i]->is_new_added_
				<< std::endl;

			ofs << pts_[i]->data[0]
				<< " " << pts_[i]->data[1]
				<< " " << pts_[i]->data[2]
				<< std::endl;

			ofs << pts_[i]->cams_.size() << std::endl;
			for (auto iter = pts_[i]->cams_.begin(); iter != pts_[i]->cams_.end(); ++iter)
			{
				ofs << iter->first << " " << iter->second->id_img_ << " ";
			}
			ofs << std::endl;

			ofs << pts_[i]->pts2d_.size() << std::endl;
			for (auto iter = pts_[i]->pts2d_.begin(); iter != pts_[i]->pts2d_.end(); ++iter)
			{
				ofs << iter->first << " " << iter->second(0) << " " << iter->second(1) << " ";
			}
			ofs << std::endl;

			ofs << pts_[i]->key_new_obs_ << std::endl;

			ofs << pts_[i]->mse_ << std::endl;
		}

		// write out localization failure
		for (size_t i = 0; i < localize_fail_times_.size(); i++)
		{
			ofs << localize_fail_times_[i] << " ";
		}

		ofs.close();
	}

	void IncrementalSfM::ReadTempResultIn(std::string path)
	{
		std::ifstream ifs(path);
		if (!ifs.is_open())
		{
			return;
		}

		// read cam-models
		int cam_models__size = 0;
		ifs >> cam_models__size;
		cam_models_.resize(cam_models__size);
		for (size_t i = 0; i < cam_models_.size(); i++)
		{
			cam_models_[i] = new CameraModel;
			ifs >> cam_models_[i]->id_;

			std::string temp;
			std::getline(ifs, temp);
			std::getline(ifs, cam_models_[i]->cam_maker_);
			std::getline(ifs, cam_models_[i]->cam_model_);

			ifs >> cam_models_[i]->w_ >> cam_models_[i]->h_;
			ifs >> cam_models_[i]->f_mm_
				>> cam_models_[i]->f_
				>> cam_models_[i]->f_hyp_
				>> cam_models_[i]->px_
				>> cam_models_[i]->py_
				>> cam_models_[i]->k1_
				>> cam_models_[i]->k2_
				>> cam_models_[i]->data[0]
				>> cam_models_[i]->data[1]
				>> cam_models_[i]->data[2];

			ifs >> cam_models_[i]->num_cams_;
		}

		// read cams
		int cams_size = 0;
		ifs >> cams_size;
		cams_.resize(cams_size);
		std::vector<int> cam_model_info(cams_size);
		std::vector<std::vector<std::pair<int, int>>> cams_pts_info(cams_size);
		for (size_t i = 0; i < cams_.size(); i++)
		{
			cams_[i] = new Camera;
			ifs >> cams_[i]->id_img_;

			is_img_processed_[cams_[i]->id_img_] = true;
			img_cam_map_.insert(std::pair<int, int>(cams_[i]->id_img_, i));

			int cam_model_id = 0;
			ifs >> cam_model_info[i];

			ifs >> cams_[i]->is_mutable_;

			ifs >> cams_[i]->data[0]
				>> cams_[i]->data[1]
				>> cams_[i]->data[2]
				>> cams_[i]->data[3]
				>> cams_[i]->data[4]
				>> cams_[i]->data[5];

			int num_cam_pts = 0;
			ifs >> num_cam_pts;
			int key, id;
			for (size_t j = 0; j<num_cam_pts; ++j)
			{
				ifs >> key >> id;
				cams_pts_info[i].push_back(std::pair<int, int>(key, id));
			}

			int num_cam_visible = 0;
			ifs >> num_cam_visible;
			cams_[i]->visible_cams_.resize(num_cam_visible);
			for (size_t j = 0; j < cams_[i]->visible_cams_.size(); j++)
			{
				ifs >> cams_[i]->visible_cams_[j];
			}
		}

		// read points
		int pts_size = 0;
		ifs >> pts_size;
		pts_.resize(pts_size);
		std::vector<std::vector<std::pair<int, int>>> pts_cams_info(pts_size);
		for (size_t i = 0; i < pts_.size(); i++)
		{
			pts_[i] = new Point3D();

			ifs >> pts_[i]->id_;

			ifs >> pts_[i]->is_mutable_
				>> pts_[i]->is_bad_estimated_
				>> pts_[i]->is_new_added_;

			ifs >> pts_[i]->data[0]
				>> pts_[i]->data[1]
				>> pts_[i]->data[2];

			int num_pts_cams;
			ifs >> num_pts_cams;
			int key, id;
			for (size_t j = 0; j<num_pts_cams; ++j)
			{
				ifs >> key >> id;
				pts_cams_info[i].push_back(std::pair<int, int>(key, id));
			}

			int num_pts_obs;
			ifs >> num_pts_obs;
			double x, y;
			for (size_t j = 0; j<num_pts_cams; ++j)
			{
				ifs >> key >> x >> y;
				pts_[i]->pts2d_.insert(std::pair<int, Eigen::Vector2d>(key, Eigen::Vector2d(x, y)));
			}

			ifs >> pts_[i]->key_new_obs_;
			ifs >> pts_[i]->mse_;
		}

		// localization failure
		localize_fail_times_.resize(db_.num_imgs_, 0);
		for (size_t i = 0; i < localize_fail_times_.size(); i++)
		{
			ifs >> localize_fail_times_[i];
		}

		ifs.close();

		// association
		std::map<int, CameraModel*> cam_models_map_;
		for (size_t i = 0; i < cam_models_.size(); i++)
		{
			cam_models_map_.insert(std::pair<int, CameraModel*>(cam_models_[i]->id_, cam_models_[i]));
		}

		std::map<int, Camera*> cams_map_;
		for (size_t i = 0; i < cams_.size(); i++)
		{
			cams_map_.insert(std::pair<int, Camera*>(cams_[i]->id_img_, cams_[i]));
		}

		std::map<int, Point3D*> pts_map_;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			pts_map_.insert(std::pair<int, Point3D*>(pts_[i]->id_, pts_[i]));
		}

		// cam
		for (size_t i = 0; i < cams_.size(); i++)
		{
			int id_cam_model = cam_model_info[i];
			cams_[i]->AssociateCamereModel(cam_models_map_[id_cam_model]);
			cams_[i]->UpdatePoseFromData();

			for (size_t j = 0; j < cams_pts_info[i].size(); j++)
			{
				int idx_local = cams_pts_info[i][j].first;
				int id_pts3d = cams_pts_info[i][j].second;
				cams_[i]->AddPoints(pts_map_[id_pts3d], idx_local);
			}
		}

		// points
		for (size_t i = 0; i < pts_.size(); i++)
		{
			for (size_t j = 0; j < pts_cams_info[i].size(); j++)
			{
				int idx_local = pts_cams_info[i][j].first;
				int id_cam = pts_cams_info[i][j].second;
				pts_[i]->cams_.insert(std::pair<int, Camera*>(idx_local, cams_map_[id_cam]));
			}
		}
	}

	void IncrementalSfM::Drawmatch(int idx1, int idx2, cv::Mat & img)
	{
		db_.ReadinImageFeatures(idx1);
		db_.ReadinImageFeatures(idx2);
		cv::Mat image1 = cv::imread(db_.image_paths_[idx1]);
		cv::Mat image2 = cv::imread(db_.image_paths_[idx2]);
		float ratio1 = db_.image_infos_[idx1]->zoom_ratio;
		float ratio2 = db_.image_infos_[idx2]->zoom_ratio;
		//float ratio2 = db_->image_infos_[idx2]->zoom_ratio;

		// query matches
		std::vector<std::pair<int, int>> matchs_inliers;
		graph_.QueryMatch(idx1, idx2, matchs_inliers);


		int pitch = 128;
		cv::resize(image1, image1, cv::Size(image1.cols*ratio1, image1.rows*ratio1));
		cv::resize(image2, image2, cv::Size(image2.cols*ratio2, image2.rows*ratio2));
		for (size_t m = 0; m < matchs_inliers.size(); m++)
		{
			int id_pt1_local = matchs_inliers[m].first;
			int id_pt2_local = matchs_inliers[m].second;
			cv::Point2f offset1(image1.cols / 2.0, image1.rows / 2.0);
			cv::Point2f offset2(image2.cols / 2.0, image2.rows / 2.0);
			cv::line(image1, db_.keypoints_[idx1]->pts[id_pt1_local].pt + offset1,
				db_.keypoints_[idx2]->pts[id_pt2_local].pt + offset2, cv::Scalar(0, 0, 255), 1);
		}
		img = image1;
	}

	bool IncrementalSfM::CameraAssociateCameraModel(Camera * cam)
	{
		// load the information of the accosiated image
		db_.ReadinImageFeatures(cam->id_img_);

		if (options_.use_same_camera)
		{
			// check if there are same camera model existing
			int idx_cam_model = -1;
			for (size_t i = 0; i < cam_models_.size(); i++)
			{
				//if (!db_.image_infos_[cam->id_img_]->f_mm
				//	|| !(db_.image_infos_[cam->id_img_]->cam_maker.length()
				//		|| db_.image_infos_[cam->id_img_]->cam_model.length()))
				//{
				//	continue;
				//}

				if (db_.image_infos_[cam->id_img_]->f_mm == cam_models_[i]->f_mm_
					&& db_.image_infos_[cam->id_img_]->rows == cam_models_[i]->h_
					&& db_.image_infos_[cam->id_img_]->cols == cam_models_[i]->w_)
				{
					idx_cam_model = i;
					break;
				}
			}

			if (idx_cam_model >= 0)
			{
				cam->AssociateCamereModel(cam_models_[idx_cam_model]);
				cam_models_[idx_cam_model]->num_cams_++;
				return true;
			}
		}
		return false;
	}

	void IncrementalSfM::SortImagePairs(std::vector<std::pair<int, int>> &seed_pair_hyps)
	{
		int num_img = db_.num_imgs_;
		std::vector<float> match_strength(num_img, 0);
		for (size_t i = 0; i < num_img; i++)
		{
			math::sum((graph_.match_graph_+i*num_img), num_img, match_strength[i]);
			match_strength[i] = std::log(match_strength[i]+2.0);
		}

		// calculate the image pair matching strength
		std::vector<std::pair<int, float>> pairs;
		for (size_t i = 0; i < num_img - 1; i++)
		{
			if (is_img_processed_[i])
			{
				continue;
			}

			for (size_t j = i + 1; j < num_img; j++)
			{
				int num_match_ij = graph_.match_graph_[i*num_img+j];
				if (!num_match_ij || is_img_processed_[j])
				{
					continue;
				}

				double strength = match_strength[i] * match_strength[j] * std::log(num_match_ij);
				pairs.push_back(std::pair<int, float>(i*num_img + j, strength));
			}
		}
		std::sort(pairs.begin(), pairs.end(), [](const std::pair<int, float> &lhs, const std::pair<int, float> &rhs) { return lhs.second > rhs.second; });

		for (size_t i = 0; i < pairs.size(); i++)
		{
			int id1 = pairs[i].first / num_img;
			int id2 = pairs[i].first % num_img;
			seed_pair_hyps.push_back(std::pair<int, int>(id1, id2));
		}
	}

	void IncrementalSfM::RemovePointOutliers()
	{
		int count_outliers = 0;
		int count_outliers_new_add = 0;
		int count_new_add = 0;
		for (size_t i = 0; i < pts_.size(); i++)
		{
			if (pts_[i]->is_bad_estimated_)
			{
				continue;
			}
			if (pts_[i]->is_new_added_)
			{
				count_new_add++;
			}

			//pts_[i]->is_bad_estimated_ = false;
			pts_[i]->Reprojection();
			if (std::sqrt(pts_[i]->mse_) > options_.th_mse_outliers)
			{
				pts_[i]->is_bad_estimated_ = true;
				count_outliers++;
				if (pts_[i]->is_new_added_)
				{
					count_outliers_new_add++;
				}
			}
			pts_[i]->is_new_added_ = false;
		}

		std::cout << "----------RemovePointOutliers: " << count_outliers << " of " << pts_.size() << std::endl;
		std::cout << "----------RemovePointOutliers New: " << count_outliers_new_add << " of " << count_new_add << std::endl;
	}

	void IncrementalSfM::ImmutableCamsPoints()
	{
		for (size_t i = 0; i < cams_.size(); i++)
		{
			cams_[i]->SetMutable(false);

			std::map<int, Point3D*>::iterator iter = cams_[i]->pts_.begin();
			while (iter != cams_[i]->pts_.end())
			{
				iter->second->SetMutable(false);
				iter++;
			}
		}
	}

	void IncrementalSfM::MutableCamsPoints()
	{
		for (size_t i = 0; i < cams_.size(); i++)
		{
			cams_[i]->SetMutable(true);

			std::map<int, Point3D*>::iterator iter = cams_[i]->pts_.begin();
			while (iter != cams_[i]->pts_.end())
			{
				iter->second->SetMutable(true);
				iter++;
			}
		}
	}

	void IncrementalSfM::UpdateVisibleGraph(int idx_new_cam, std::vector<int> idxs_visible_cam)
	{
		cams_[idx_new_cam]->AddVisibleCamera(idx_new_cam);
		for (size_t i = 0; i < idxs_visible_cam.size(); i++)
		{
			cams_[idx_new_cam]->AddVisibleCamera(idxs_visible_cam[i]);
			cams_[idxs_visible_cam[i]]->AddVisibleCamera(idx_new_cam);
		}
	}

	void IncrementalSfM::UndistortedPts(std::vector<Eigen::Vector2d> pts, std::vector<Eigen::Vector2d> &pts_undistorted, CameraModel* cam_model)
	{
		
	}

}  // namespace objectsfm
