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

#include "initial_matching_graph.h"
#include "similarity_graph.h"

#include <fstream>
#include <omp.h>

#include "utils/basic_funcs.h"
#include "utils/ellipsoid_utm_info.h"
#include "utils/converter_utm_latlon.h"
#include "utils/geo_verification.h"
#include "flann/flann.h"

namespace objectsfm {

	InitialMatchingGraph::InitialMatchingGraph()
	{
	}

	InitialMatchingGraph::~InitialMatchingGraph()
	{
	}

	void InitialMatchingGraph::AssociateDatabase(Database * db)
	{
		db_ = db;
		num_imgs_ = db_->num_imgs_;
	}

	void InitialMatchingGraph::BuildInitialMatchGraph()
	{
		match_graph_init = std::vector<std::vector<int>>(num_imgs_);

		// step1: read in existing initial match rsults
		ReadinInitMatchGraph(id_last_init_match_);
		if (id_last_init_match_ == num_imgs_ - 1)
			return;

		// step2: do initial matching
		if (options_.matching_type == "all")
		{
			for (size_t i = 0; i < num_imgs_; i++) {
				for (size_t j = 0; j < num_imgs_; j++) {					
					if (j != i) {
						match_graph_init[i].push_back(j);
					}
				}
			}
		}
		else if (options_.matching_type == "priori")
		{
			if (options_.priori_type == "llt") 
			{
				match_graph_priori_ll();
			}
			if (options_.priori_type == "xyz")
			{
				match_graph_priori_xy();
			}
		}
		else {
			match_graph_feature();
		}

		WriteOutInitMatchGraph(num_imgs_ - 1);
	}

	void InitialMatchingGraph::match_graph_priori_ll()
	{
		// read in the llt file
		std::ifstream ff(options_.priori_file);
		std::vector<cv::Point2d> pos(num_imgs_);

		int id = 0;
		double lat = 0.0, lon = 0.0, alt = 0.0;
		double x = 0.0, y = 0.0;
		while (!ff.eof())
		{
			ff >> id >> lat >> lon >> alt;

			// convert into xyz
			LLtoUTM(options_.ellipsoid_id_, lat, lon, y, x, (char*)options_.zone_id_.c_str());
			pos[id] = cv::Point2d(x, y);
		}
		ff.close();

		// generate initial matching graph
		match_graph_priori_xy(pos);
	}

	void InitialMatchingGraph::match_graph_priori_xy()
	{
		// read in the xy pos
		std::vector<cv::Point2d> pos(num_imgs_);

		match_graph_priori_xy(pos);
	}

	void InitialMatchingGraph::match_graph_priori_xy(std::vector<cv::Point2d>& pts)
	{
		double th_dis = 1.0;

		// find close images
		std::vector<std::pair<int, double>> id_dis(pts.size());
		for (size_t i = 0; i < pts.size(); i++) {
			id_dis[i].first = i;
			id_dis[i].second = pts[i].x + pts[i].y;
		}
		std::sort(id_dis.begin(), id_dis.end(), [](const std::pair<int, double> &lhs, const std::pair<int, double> &rhs) { return lhs.second < rhs.second; });

		std::vector<int> is_redundency(pts.size(), 0);
		double dis_pre = id_dis[0].second - 100.0;
		for (size_t i = 0; i < id_dis.size(); i++)
		{
			int idx = id_dis[i].first;
			double dis = id_dis[i].second;
			if (abs(dis - dis_pre) < th_dis) {
				is_redundency[idx] = 1;
			}
			else {
				dis_pre = dis;
			}
		}

		// 
		int k = MIN_(options_.knn, num_imgs_ / 10);
		for (int i = 0; i < pts.size(); i++)
		{
			if (is_redundency[i]) {
				continue;
			}

			std::vector<std::pair<int, double>> info;
			for (int j = 0; j < pts.size(); j++) {
				if (j != i && !is_redundency[j]) {
					double dis = abs(pts[i].x - pts[j].x) + abs(pts[i].y - pts[j].y);
					info.push_back(std::pair<int, double>(j, dis));
				}
			}

			std::sort(info.begin(), info.end(), [](const std::pair<int, double> &lhs, const std::pair<int, double> &rhs) { return lhs.second < rhs.second; });
			int t = MIN_(info.size(), k);
			for (size_t j = 0; j < t; j++) {
				match_graph_init[i].push_back(info[j].first);
			}
		}
	}

	void InitialMatchingGraph::match_graph_feature()
	{
		int th_init_samewords = 30;

		int th_num_match = MIN(MAX(200, num_imgs_ / 10), num_imgs_ - 1);
		if (th_num_match > 500) th_num_match = 500;

		// step2: generate bag of words 
		db_->BuildWords();

		// step3: generate similarity matrix
		std::vector<std::vector<float>> similarity_matrix;
		if (!CheckSimilarityGraph())
		{
			SimilarityGraph simer;
			simer.AssociateDatabase(db_);
			simer.BuildSimilarityGraph(1, similarity_matrix);
			WriteOutSimilarityGraph(similarity_matrix);
		}
		else {
			ReadinSimilarityGraph(similarity_matrix);
		}

		// step2: generate point_id to word_id map for each image
		std::vector<std::map<int, int>> pt_word_map(num_imgs_);
		for (size_t i = 0; i < num_imgs_; i++)
		{
			db_->ReadinWordsForImage(i);
			std::vector<int> unique_word_idxs;
			if (!db_->words_id_[i].size()) continue;
			math::keep_unique_idx_vector(db_->words_id_[i], unique_word_idxs);

			for (size_t j = 0; j < unique_word_idxs.size(); j++)
			{
				int idx = unique_word_idxs[j];
				int id_word = db_->words_id_[i][idx];
				pt_word_map[i].insert(std::pair<int, int>(id_word, idx));
			}
			db_->ReleaseWordsForImage(i);
		}

		for (size_t i = 0; i < num_imgs_; i++) {
			db_->ReadinImageKeyPoints(i);
		}

		// step4: do matching via quarying
		for (size_t i = id_last_init_match_; i < num_imgs_; i++)
		{
			std::cout << i << std::endl;

			int idx1 = i;

			// find matching hypotheses
			std::vector<std::pair<int, float>> sim_sort;
			for (size_t j = 0; j < num_imgs_; j++)
			{
				if (j == idx1 || similarity_matrix[i][j] < 0) {
					continue;
				}
				sim_sort.push_back(std::pair<int, float>(j, similarity_matrix[i][j]));
			}
			std::sort(sim_sort.begin(), sim_sort.end(), [](const std::pair<int, float> &lhs, const std::pair<int, float> &rhs) { return lhs.second > rhs.second; });

			std::vector<int> set_idx2(th_num_match);
			for (size_t j = 0; j < th_num_match; j++) {
				int idx2 = sim_sort[j].first;
				set_idx2[j] = idx2;
			}

			// matching parallelly
			std::cout << "---Matching images " << idx1 << "  hypotheses " << set_idx2.size() << std::endl;
			std::vector<int> num_matches(set_idx2.size(), 0);
#pragma omp parallel for
			for (int j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];

				// initial matching via words
				std::vector<std::pair<int, int>> matches_init;
				for (auto iter1 : pt_word_map[idx1])
				{
					int id_word = iter1.first;
					int id_pt1 = iter1.second;
					auto iter2 = pt_word_map[idx2].find(id_word);
					if (iter2 != pt_word_map[idx2].end())
					{
						int id_pt2 = iter2->second;
						matches_init.push_back(std::pair<int, int>(id_pt1, id_pt2));
					}
				}
				if (matches_init.size() < 30) {
					continue;
				}

				// refine matching via geo-verification
				std::vector<int> match_init_inliers;
				std::vector<cv::Point2f> pts1, pts2;
				for (size_t m = 0; m < matches_init.size(); m++)
				{
					int index1 = matches_init[m].first;
					int index2 = matches_init[m].second;
					pts1.push_back(db_->keypoints_[idx1]->pts[index1].pt);
					pts2.push_back(db_->keypoints_[idx2]->pts[index2].pt);
				}

				if (!GeoVerification::GeoVerificationFundamental(pts1, pts2, match_init_inliers)) {
					continue;
				}

				if (match_init_inliers.size() > 20) {
					num_matches[j] = match_init_inliers.size();
					std::cout << idx1 << "  " << idx2 << "   matches: " << match_init_inliers.size() << std::endl;
				}
			}

			for (size_t j = 0; j < num_matches.size(); j++) {
				if (num_matches[j]) {
					match_graph_init[i].push_back(set_idx2[j]);
				}
			}

			if (i % 100 == 0) {
				WriteOutInitMatchGraph(idx1);
			}
		}

		for (size_t i = 0; i < num_imgs_; i++) {
			db_->ReleaseImageKeyPoints(i);
		}
	}

	void InitialMatchingGraph::ReadinInitMatchGraph(int & id_last)
	{
		std::string path = db_->output_fold_ + "//" + "init_match_graph.txt";
		std::ifstream ifs(path);
		if (!ifs.is_open()) {
			match_graph_init.resize(num_imgs_);
			return;
		}

		int num_img = 0;
		ifs >> num_img;
		ifs >> id_last;
		match_graph_init.resize(num_img);
		for (size_t i = 0; i < match_graph_init.size(); i++)
		{
			int num_matched_imgs;
			ifs >> num_matched_imgs;
			if (num_matched_imgs > 0)
			{
				match_graph_init[i].resize(num_matched_imgs);
				for (size_t j = 0; j < match_graph_init[i].size(); j++) {
					ifs >> match_graph_init[i][j];
				}
			}
		}
		ifs.close();
	}

	void InitialMatchingGraph::WriteOutInitMatchGraph(int id_last)
	{
		std::string path = db_->output_fold_ + "//" + "init_match_graph.txt";
		std::ofstream ofs(path);
		if (!ofs.is_open()) {
			return;
		}

		ofs << match_graph_init.size() << std::endl;
		ofs << id_last << std::endl;
		for (size_t i = 0; i < match_graph_init.size(); i++)
		{
			ofs << match_graph_init[i].size() << " ";
			for (size_t j = 0; j < match_graph_init[i].size(); j++)
			{
				ofs << match_graph_init[i][j] << " ";
			}
			ofs << std::endl;
		}
		ofs.close();
	}

	bool InitialMatchingGraph::CheckSimilarityGraph()
	{
		std::string path_sim_graph = db_->output_fold_ + "//" + "similarity_graph.txt";
		std::ifstream infile(path_sim_graph);
		if (!infile.is_open()) {
			return false;
		}

		return true;
	}

	void InitialMatchingGraph::WriteOutSimilarityGraph(std::vector<std::vector<float>> &similarity_matrix)
	{
		std::string path_pts_words = db_->output_fold_ + "//" + "similarity_graph.txt";
		std::ofstream ofs(path_pts_words);

		int num = similarity_matrix.size();
		ofs << num << std::endl;
		for (size_t i = 0; i < num; i++)
		{
			for (size_t j = 0; j < similarity_matrix[i].size(); j++)
			{
				ofs << similarity_matrix[i][j] << " ";
			}
			ofs << std::endl;
		}
		ofs.close();
	}

	void InitialMatchingGraph::ReadinSimilarityGraph(std::vector<std::vector<float>> &similarity_matrix)
	{
		std::string path_pts_words = db_->output_fold_ + "//" + "similarity_graph.txt";
		std::ifstream ifs(path_pts_words);

		int num = 0;
		ifs >> num;
		similarity_matrix.resize(num);
		for (size_t i = 0; i < num; i++)
		{
			similarity_matrix[i].resize(num);
			for (size_t j = 0; j < num; j++)
			{
				ifs >> similarity_matrix[i][j];
			}
		}
		ifs.close();
	}


}