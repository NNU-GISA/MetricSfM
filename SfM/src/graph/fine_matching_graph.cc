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

#include "fine_matching_graph.h"

#include <fstream>
#include <omp.h>
#include <boost/thread.hpp>

#include "utils/basic_funcs.h"
#include "utils/geo_verification.h"

#include "feature/feature_matching.h"
#include "flann/flann.h"

namespace objectsfm {

	FineMatchingGraph::FineMatchingGraph()
	{
	}

	FineMatchingGraph::~FineMatchingGraph()
	{
	}

	void FineMatchingGraph::AssociateDatabase(Database * db) {
		db_ = db;
	}

	void FineMatchingGraph::BuildMatchGraph(std::vector<std::vector<int>> &match_graph_init)
	{
		float thRatio = 0.5;
		float thDis = 0.2;

		num_imgs_ = db_->num_imgs_;
		
		// read in the existing matching results
		std::vector<int> messing_matches = CheckMissingMatchingFile();
		if (!messing_matches.size()) return;

		std::vector<int> existing_matches = math::vector_subtract(num_imgs_, messing_matches);
		RecoverMatchingGraph(existing_matches);

		// do matching
		std::string file_match = db_->output_fold_ + "//match_index.txt";
		std::ofstream of_match_index(file_match, std::ios::app);
		for (size_t i = 0; i < messing_matches.size(); i++)
		{
			int idx1 = messing_matches[i];
			std::cout << "---Matching images " << idx1 << "/" << num_imgs_ << std::endl;

			std::vector<int> set_idx2 = match_graph_init[idx1];
			if (!set_idx2.size()) {
				of_match_index << idx1 << std::endl;
				continue;
			}
			db_->ReadinImageFeatures(idx1);

			// generate kd-tree for idx1
			struct FLANNParameters p;
			p = DEFAULT_FLANN_PARAMETERS;
			p.algorithm = FLANN_INDEX_KDTREE;
			p.trees = 8;
			p.log_level = FLANN_LOG_INFO;
			p.checks = 64;
			float speedup;

			float* data_idx1 = (float*)db_->descriptors_[idx1]->data;
			flann_index_t kdtree_idx1 = flann_build_index(data_idx1, db_->descriptors_[idx1]->rows, db_->descriptors_[idx1]->cols, &speedup, &p);

			// knn querying
			std::cout << idx1 << "  knn querying ..." << std::endl;
			std::vector<int*> knn_id(set_idx2.size());
			std::vector<float*> knn_dis(set_idx2.size());
			for (int j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];
				db_->ReadinImageFeatures(idx2);

				std::cout << "  -------" << j << " " << db_->descriptors_[idx1]->rows
					<< " " << db_->descriptors_[idx2]->rows << std::endl;

				int count = db_->keypoints_[idx2]->pts.size();
				knn_id[j] = new int[count * 2];
				knn_dis[j] = new float[count * 2];
				flann_find_nearest_neighbors_index(kdtree_idx1, (float*)db_->descriptors_[idx2]->data, count, knn_id[j], knn_dis[j], 2, &p);
			}
			flann_free_index(kdtree_idx1, &p);

			// geo-verification
			for (int j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];
				int num2 = db_->keypoints_[idx2]->pts.size();

				// the inital matches
				std::vector<std::pair<int, int>> matches;
				std::vector<cv::Point2f> pt1, pt2;
				int* ptr_id = knn_id[j];
				float* ptr_dis = knn_dis[j];
				for (size_t m = 0; m < num2; m++)
				{
					float ratio = ptr_dis[0] / ptr_dis[1];
					if (ratio < thRatio && ptr_dis[0] < thDis) {
						pt1.push_back(db_->keypoints_[idx1]->pts[ptr_id[0]].pt);
						pt2.push_back(db_->keypoints_[idx2]->pts[m].pt);
						matches.push_back(std::pair<int, int>(ptr_id[0], m));
					}
					ptr_id += 2;
					ptr_dis += 2;
				}

				// geo-verification via flow
				std::vector<int> match_inliers;
				//bool isOK = GeoVerification::GeoVerificationLocalFlow(pt1, pt2, match_inliers);
				//bool isOK = GeoVerification::GeoVerificationFundamental(pt1, pt2, match_inliers);
				bool isOK = GeoVerification::GeoVerificationPatchFundamental(pt1, pt2, match_inliers);
				db_->ReleaseImageFeatures(idx2);
				delete[] knn_id[j];
				delete[] knn_dis[j];

				std::cout << idx1 << "  " << idx2 << "   matches: " << match_inliers.size() << std::endl;

				// draw
				if (0)
				{
					db_->ReadinImageFeatures(idx2);
					cv::Mat image1 = cv::imread(db_->image_paths_[idx1]);
					cv::Mat image2 = cv::imread(db_->image_paths_[idx2]);
					float ratio1 = db_->image_infos_[idx1]->zoom_ratio;
					float ratio2 = db_->image_infos_[idx1]->zoom_ratio;

					int pitch = 128;
					cv::resize(image1, image1, cv::Size(image1.cols*ratio1, image1.rows*ratio1));
					cv::resize(image2, image2, cv::Size(image2.cols*ratio2, image2.rows*ratio2));
					for (size_t m = 0; m < match_inliers.size(); m++)
					{
						int idx = match_inliers[m];
						int id_pt1_local = matches[idx].first;
						int id_pt2_local = matches[idx].second;
						cv::Point2f offset1(image1.cols / 2.0, image1.rows / 2.0);
						cv::Point2f offset2(image2.cols / 2.0, image2.rows / 2.0);
						cv::line(image1, db_->keypoints_[idx1]->pts[id_pt1_local].pt + offset1,
							db_->keypoints_[idx2]->pts[id_pt2_local].pt + offset2, cv::Scalar(0,0,255), 1);
					}
					std::string path = "F:\\" + std::to_string(idx2) + "cuda.jpg";
					cv::imwrite(path, image1);
				}

				if (isOK)
				{
					WriteOutMatches(idx1, idx2, matches);
					match_graph[idx1][idx2] = matches.size();
				}
			}

			of_match_index << idx1 << std::endl;
		}
		of_match_index.close();

		WriteOutMatchGraph();
	}


	bool FineMatchingGraph::CheckMatchIndexFile()
	{
		std::string path = db_->output_fold_ + "//match_index.txt";
		std::ifstream infile(path);
		if (!infile.good())
		{
			return false;
		}

		return true;
	}

	std::vector<int> FineMatchingGraph::CheckMissingMatchingFile()
	{
		std::vector<int> missing_idx;

		std::string path = db_->output_fold_ + "//match_index.txt";
		std::ifstream infile(path);
		if (!infile.good())
		{
			for (size_t i = 0; i < num_imgs_; i++)
			{
				missing_idx.push_back(i);
			}
			return missing_idx;
		}

		std::vector<int> index(num_imgs_, 0);
		int idx = -1;
		while (!infile.eof())
		{
			infile >> idx;
			if (idx >= 0)
			{
				index[idx] = 1;
			}
		}
		
		for (size_t i = 0; i < num_imgs_; i++)
		{
			if (!index[i])
			{
				missing_idx.push_back(i);
			}
		}

		return missing_idx;
	}


	void FineMatchingGraph::WriteOutMatches(int idx1, int idx2, std::vector<std::pair<int, int>>& matches)
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
		std::string match_file_i = db_->output_fold_ + "//" + std::to_string(idx1) + "_match";
		ofsi.open(match_file_i, std::ios::out | std::ios::app | std::ios::binary);
		ofsi.write((const char*)(&idx2), sizeof(int));
		ofsi.write((const char*)(&num_match), sizeof(int));
		ofsi.write((const char*)(tempi), num_match * 2 * sizeof(int));
		ofsi.close();

		delete[] tempi;
	}


	void FineMatchingGraph::WriteOutMatchGraph()
	{
		std::string path = db_->output_fold_ + "//" + "graph_matching.txt";
		std::ofstream ofs(path, std::ios::binary);

		if (!ofs.is_open()) {
			return;
		}

		for (size_t i = 0; i < num_imgs_; i++) {
			for (size_t j = 0; j < match_graph[i].size(); j++) {
				ofs << match_graph[i][j] << " ";
			}
			ofs << std::endl;

		}
		ofs.close();
	}

	void FineMatchingGraph::RecoverMatchingGraph(std::vector<int>& existing_matches)
	{
		match_graph.resize(num_imgs_);
		for (size_t i = 0; i < num_imgs_; i++) {
			match_graph[i].resize(num_imgs_);
		}

		// read in data
		for (size_t i = 0; i < existing_matches.size(); i++)
		{
			int idx = existing_matches[i];

			std::string match_file = db_->output_fold_ + "//" + std::to_string(idx) + "_match";
			std::ifstream ifs;
			ifs.open(match_file, std::ios::in | std::ios::binary);
			if (!ifs.is_open()) continue;

			int id, num_match;
			while (ifs.read((char*)(&id), sizeof(int)))
			{
				ifs.read((char*)(&num_match), sizeof(int));

				int *temp = new int[num_match * 2];
				ifs.read((char*)(temp), num_match * 2 * sizeof(int));
				delete[] temp;

				match_graph[idx][id] = num_match;
			}
			ifs.close();
		}
		
	}

}