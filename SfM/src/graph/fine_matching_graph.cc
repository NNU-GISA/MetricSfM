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
#include "feature/feature_matching_flann.h"
#include "feature/feature_verification.h"

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

	void FineMatchingGraph::BuildMatchGraph(std::vector<std::vector<int>>& match_graph_init)
	{
		// first matching
		GraphMatching(match_graph_init);

		// then verification
		MatchingVerification();
	}

	void FineMatchingGraph::GraphMatching(std::vector<std::vector<int>> &match_graph_init)
	{
		float thRatio = 0.85;
		float thDis = 0.5;

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

			// read in image feature
			db_->ReadinImageFeatures(idx1);
			std::vector<cv::Mat*> set_descriptor2(set_idx2.size());
			for (size_t j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];
				db_->ReadinImageFeatures(idx2);
				set_descriptor2[j] = db_->descriptors_[idx2];
			}

			// do initial matching via feature
			std::vector<std::vector<std::pair<int, int>>> matches;
			FeatureMatchingFlann::Run(db_->descriptors_[idx1], set_descriptor2, thRatio, thDis, matches);
			//FeatureMatchingCudaSift::Run(db_->descriptors_[idx1], set_descriptor2, thRatio, thDis, matches_init);

			for (size_t j = 0; j < matches.size(); j++) {
				int idx2 = set_idx2[j];
				db_->ReleaseImageFeatures(idx2);
				WriteOutMatches(idx1, idx2, matches[j]);
				match_graph[idx1][idx2] = matches[j].size();
			}

			db_->ReleaseImageFeatures(idx1);

			of_match_index << idx1 << std::endl;
		}
		of_match_index.close();

		WriteOutMatchGraph();
	}

	void FineMatchingGraph::MatchingVerification()
	{
		num_imgs_ = db_->num_imgs_;
		std::string verify_type = "geo_verify";

		// read in the existing matching results
		std::vector<int> messing_matches = CheckMissingMatchingFile2();
		if (!messing_matches.size()) return;

		std::vector<int> existing_matches = math::vector_subtract(num_imgs_, messing_matches);
		RecoverMatchingGraph(existing_matches);

		//		
		std::string file_match = db_->output_fold_ + "//match_index2.txt";
		std::ofstream of_match_index(file_match, std::ios::app);
		for (size_t i = 0; i < messing_matches.size(); i++)
		{
			int idx1 = messing_matches[i];
			std::cout << "---Matching images " << idx1 << "/" << num_imgs_ << std::endl;

			// read in initial matches
			std::vector<int> idx2_list;
			std::vector<std::vector<std::pair<int, int>>> matches_init;
			ReadinMatch(idx1, idx2_list, matches_init);

			// do verification
			if (verify_type == "cross_check") {
				for (size_t j = 0; j < idx2_list.size(); j++)
				{
					int idx2 = idx2_list[j];
					std::vector<int> idx_temp;
					std::vector<std::vector<std::pair<int, int>>> matches_temp;
					ReadinMatch(idx2, idx_temp, matches_temp);

					std::vector<std::pair<int, int>> matchs_inliers;
					for (size_t m = 0; m < idx_temp.size(); m++) {
						if (idx_temp[m] == idx1) {	
							FeatureVerification::CrossCheck(matches_init[j], matches_temp[m], matchs_inliers);
							WriteOutMatches2(idx1, idx2, matchs_inliers);
							match_graph[idx1][idx2] = matchs_inliers.size();
						}
					}

					// draw
					if (0)
					{
						db_->ReadinImageFeatures(idx1);
						db_->ReadinImageFeatures(idx2);
						cv::Mat image1 = cv::imread(db_->image_paths_[idx1]);
						cv::Mat image2 = cv::imread(db_->image_paths_[idx2]);
						float ratio1 = db_->image_infos_[idx1]->zoom_ratio;
						float ratio2 = db_->image_infos_[idx2]->zoom_ratio;
						//float ratio2 = db_->image_infos_[idx2]->zoom_ratio;

						int pitch = 128;
						cv::resize(image1, image1, cv::Size(image1.cols*ratio1, image1.rows*ratio1));
						cv::resize(image2, image2, cv::Size(image2.cols*ratio2, image2.rows*ratio2));
						for (size_t m = 0; m < matchs_inliers.size(); m++)
						{
							int id_pt1_local = matchs_inliers[m].first;
							int id_pt2_local = matchs_inliers[m].second;
							cv::Point2f offset1(image1.cols / 2.0, image1.rows / 2.0);
							cv::Point2f offset2(image2.cols / 2.0, image2.rows / 2.0);
							cv::line(image1, db_->keypoints_[idx1]->pts[id_pt1_local].pt + offset1,
								db_->keypoints_[idx2]->pts[id_pt2_local].pt + offset2, cv::Scalar(0, 0, 255), 1);
						}
						std::string path = "F:\\" + std::to_string(idx2) + "cuda.jpg";
						cv::imwrite(path, image1);
					}
				}
			}
			else if (verify_type == "geo_verify") {
				db_->ReadinImageFeatures(idx1);

				for (size_t j = 0; j < idx2_list.size(); j++)
				{
					int idx2 = idx2_list[j];
					db_->ReadinImageFeatures(idx2);
					std::vector<cv::Point2f> pt1, pt2;
					for (size_t m = 0; m < matches_init[j].size(); m++)
					{
						int id1 = matches_init[j][m].first;
						int id2 = matches_init[j][m].second;
						pt1.push_back(db_->keypoints_[idx1]->pts[id1].pt);
						pt2.push_back(db_->keypoints_[idx2]->pts[id2].pt);
					}
					std::vector<int> inliers;
					FeatureVerification::GeoVerificationPatchFundamental(pt1, pt2, inliers);

					//
					std::vector<std::pair<int, int>> matchs_inliers(inliers.size());
					for (size_t n = 0; n < inliers.size(); n++) {
						matchs_inliers[n] = matches_init[j][inliers[n]];
					}
					WriteOutMatches2(idx1, idx2, matchs_inliers);


					// draw
					if (0)
					{
						db_->ReadinImageFeatures(idx1);
						db_->ReadinImageFeatures(idx2);
						cv::Mat image1 = cv::imread(db_->image_paths_[idx1]);
						cv::Mat image2 = cv::imread(db_->image_paths_[idx2]);
						float ratio1 = db_->image_infos_[idx1]->zoom_ratio;
						float ratio2 = db_->image_infos_[idx2]->zoom_ratio;
						//float ratio2 = db_->image_infos_[idx2]->zoom_ratio;

						int pitch = 128;
						cv::resize(image1, image1, cv::Size(image1.cols*ratio1, image1.rows*ratio1));
						cv::resize(image2, image2, cv::Size(image2.cols*ratio2, image2.rows*ratio2));
						for (size_t m = 0; m < matchs_inliers.size(); m++)
						{
							int id_pt1_local = matchs_inliers[m].first;
							int id_pt2_local = matchs_inliers[m].second;
							cv::Point2f offset1(image1.cols / 2.0, image1.rows / 2.0);
							cv::Point2f offset2(image2.cols / 2.0, image2.rows / 2.0);
							cv::line(image1, db_->keypoints_[idx1]->pts[id_pt1_local].pt + offset1,
								db_->keypoints_[idx2]->pts[id_pt2_local].pt + offset2, cv::Scalar(0, 0, 255), 1);
						}
						std::string path = "F:\\" + std::to_string(idx2) + "cuda.jpg";
						cv::imwrite(path, image1);
					}


					db_->ReleaseImageFeatures(idx2);
					match_graph[idx1][idx2] = matchs_inliers.size();
				}
			}
			else {
				std::cout << "Choose a matching verification method!" << std::endl;
			}

			db_->ReleaseImageFeatures(idx1);
			of_match_index << idx1 << std::endl;
		}
		of_match_index.close();

		WriteOutMatchGraph2();
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


	std::vector<int> FineMatchingGraph::CheckMissingMatchingFile2()
	{
		std::vector<int> missing_idx;

		std::string path = db_->output_fold_ + "//match_index2.txt";
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

	void FineMatchingGraph::RecoverMatchingGraph2(std::vector<int>& existing_matches)
	{
		match_graph2.resize(num_imgs_);
		for (size_t i = 0; i < num_imgs_; i++) {
			match_graph2[i].resize(num_imgs_);
		}

		// read in data
		for (size_t i = 0; i < existing_matches.size(); i++)
		{
			int idx = existing_matches[i];

			std::string match_file = db_->output_fold_ + "//" + std::to_string(idx) + "_match2";
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

				match_graph2[idx][id] = num_match;
			}
			ifs.close();
		}
	}

	void FineMatchingGraph::WriteOutMatches2(int idx1, int idx2, std::vector<std::pair<int, int>>& matches)
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
		std::string match_file_i = db_->output_fold_ + "//" + std::to_string(idx1) + "_match2";
		ofsi.open(match_file_i, std::ios::out | std::ios::app | std::ios::binary);
		ofsi.write((const char*)(&idx2), sizeof(int));
		ofsi.write((const char*)(&num_match), sizeof(int));
		ofsi.write((const char*)(tempi), num_match * 2 * sizeof(int));
		ofsi.close();

		delete[] tempi;
	}

	void FineMatchingGraph::WriteOutMatchGraph2()
	{
		std::string path = db_->output_fold_ + "//" + "graph_matching2.txt";
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

	void FineMatchingGraph::ReadinMatch(int idx1, std::vector<int>& idx2, std::vector<std::vector<std::pair<int, int>>> & matches)
	{
		// read in data
		std::string match_file = db_->output_fold_ + "//" + std::to_string(idx1) + "_match";
		std::ifstream ifs;
		ifs.open(match_file, std::ios::in | std::ios::binary);

		int id, num_match;
		while (ifs.read((char*)(&id), sizeof(int)))
		{
			ifs.read((char*)(&num_match), sizeof(int));

			int *temp = new int[num_match * 2];
			ifs.read((char*)(temp), num_match * 2 * sizeof(int));

			// matches
			idx2.push_back(id);
			std::vector<std::pair<int, int>> matches_temp(num_match);
			for (size_t i = 0; i < num_match; i++)
			{
				matches_temp[i].first = temp[2 * i + 0];
				matches_temp[i].second = temp[2 * i + 1];
			}
			matches.push_back(matches_temp);

			delete[] temp;
		}
		ifs.close();
	}

}