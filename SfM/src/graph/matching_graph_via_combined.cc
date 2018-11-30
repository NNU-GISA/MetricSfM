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

#include "matching_graph_via_combined.h"

#include <fstream>
#include <omp.h>
#include <boost/thread.hpp>

#include "utils/basic_funcs.h"

#include "feature/feature_matching.h"
#include "graph/similarity_graph_bow_distance.h"
#include "graph/similarity_graph_word_number.h"

#include "flann/flann.h"

namespace objectsfm {

	MatchingGraphViaCombined::MatchingGraphViaCombined()
	{
	}

	MatchingGraphViaCombined::~MatchingGraphViaCombined()
	{
	}

	void MatchingGraphViaCombined::AssociateDatabase(Database * db)
	{
		db_ = db;
	}

	void MatchingGraphViaCombined::BuildMatchGraph()
	{
		num_imgs_ = db_->num_imgs_;
		match_graph_ = new int[num_imgs_*num_imgs_];

		// step2: find initial matches via invertd file
		//if (!CheckInitMatchGraph())
		{
			BuildInitialMatchGraph();
		}
		
		// step3: refine matches via flann matching
		RefineMatchGraphViaFlann();

		delete[] match_graph_;
	}

	void MatchingGraphViaCombined::BuildInitialMatchGraph()
	{
		std::vector<std::vector<int>> match_graph_init(num_imgs_);
		if (options_.matching_type == "All_Matching")
		{
			for (size_t i = 0; i < num_imgs_; i++)
			{
				for (size_t j = 0; j < num_imgs_; j++)
				{
					if (j != i) {
						match_graph_init[i].push_back(j);
					}
				}
			}
		}
		else if (options_.matching_type == "Via_Priori")
		{

		}
		else
		{
			// step3: read in existing initial match rsults
			int id_last_init_match = 0;
			ReadinInitMatchGraph(match_graph_init, id_last_init_match);
			if (id_last_init_match == num_imgs_ - 1)
				return;

			int th_num_match = MIN(MAX(100, num_imgs_ / 10), num_imgs_ - 1);
			if (th_num_match > 500) th_num_match = 500;

			// step1: generate similarity matrix via inverted file method
			std::vector<std::vector<float>> similarity_matrix;
			if (!CheckSimilarityGraph())
			{
				SimilarityGraphWordNumber simer;
				simer.AssociateDatabase(db_);
				simer.BuildSimilarityGraph(similarity_matrix);
				WriteOutSimilarityGraph(similarity_matrix);
			}
			else
			{
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

			for (size_t i = 0; i < num_imgs_; i++)
			{
				db_->ReadinImageKeyPoints(i);
			}

			// step4: do matching via quarying
			for (size_t i = id_last_init_match; i < num_imgs_; i++)
			{
				std::cout << i << std::endl;

				int idx1 = i;

				// find matching hypotheses
				std::vector<std::pair<int, float>> sim_sort;
				for (size_t j = 0; j < num_imgs_; j++)
				{
					match_graph_[idx1*num_imgs_ + j] = 0;
					if (j == idx1 || similarity_matrix[i][j] < 0)
					{
						continue;
					}
					sim_sort.push_back(std::pair<int, float>(j, similarity_matrix[i][j]));
				}
				std::sort(sim_sort.begin(), sim_sort.end(), [](const std::pair<int, float> &lhs, const std::pair<int, float> &rhs) { return lhs.second > rhs.second; });

				std::vector<int> set_idx2(th_num_match);
				for (size_t j = 0; j < th_num_match; j++)
				{
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
					if (matches_init.size() < 30)
					{
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

					if (!GeoVerification(pts1, pts2, match_init_inliers))
					{
						continue;
					}
					if (match_init_inliers.size() > 20)
					{
						num_matches[j] = match_init_inliers.size();
						std::cout << idx1 << "  " << idx2 << "   matches: " << match_init_inliers.size() << std::endl;
					}
				}

				for (size_t j = 0; j < num_matches.size(); j++)
				{
					if (num_matches[j])
					{
						match_graph_init[i].push_back(set_idx2[j]);
					}
				}

				if (i % 100 == 0)
				{
					WriteOutInitMatchGraph(match_graph_init, idx1);
				}
			}

			for (size_t i = 0; i < num_imgs_; i++)
			{
				db_->ReleaseImageKeyPoints(i);
			}
		}

		//
		WriteOutInitMatchGraph(match_graph_init, num_imgs_ - 1);
		
	}

	void MatchingGraphViaCombined::RefineMatchGraphViaFlann()
	{
		// read in the existing matching results
		std::vector<int> messing_matches = CheckMissingMatchingFile();
		if (!messing_matches.size()) return;

		std::vector<int> existing_matches = math::vector_subtract(num_imgs_, messing_matches);
		RecoverMatchingGraph(existing_matches);

		// read in the initial match graph
		int id_last_init_match = 0;
		std::vector<std::vector<int>> match_graph_init;
		ReadinInitMatchGraph(match_graph_init, id_last_init_match);

		std::string file_match = db_->output_fold_ + "//match_index.txt";
		std::ofstream of_match_index(file_match, std::ios::app);
		for (size_t i = 0; i < messing_matches.size(); i++)
		{
			int idx1 = messing_matches[i];
			std::cout << "---Matching images " << idx1 << "/" << num_imgs_ << std::endl;

			std::vector<int> set_idx2 = match_graph_init[idx1];
			if (!set_idx2.size())
			{
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

				std::vector<std::pair<int, int>> matches;
				bool isOK = FeatureMatching::KNNMatchingWithGeoVerify(db_->keypoints_[idx1]->pts, db_->keypoints_[idx2]->pts, 
					knn_id[j], knn_dis[j], matches);
				db_->ReleaseImageFeatures(idx2);
				delete[] knn_id[j];
				delete[] knn_dis[j];

				std::cout << idx1 << "  " << idx2 << "   matches: " << matches.size() << std::endl;

				// draw
				if (0)
				{
					db_->ReadinImageFeatures(idx2);
					cv::Mat image1 = cv::imread(db_->image_paths_[idx1]);
					cv::Mat image2 = cv::imread(db_->image_paths_[idx2]);
					int pitch = 128;
					cv::resize(image1, image1, cv::Size((image1.cols / pitch + 1) * pitch, (image1.rows / pitch + 1) * pitch));
					cv::resize(image2, image2, cv::Size((image2.cols / pitch + 1) * pitch, (image2.rows / pitch + 1) * pitch));
					for (size_t m = 0; m < matches.size(); m++)
					{
						int id_pt1_local = matches[m].first;
						int id_pt2_local = matches[m].second;
						cv::Point2f offset1(db_->image_infos_[idx1]->cols / 2.0, db_->image_infos_[idx1]->rows / 2.0);
						cv::Point2f offset2(db_->image_infos_[idx2]->cols / 2.0, db_->image_infos_[idx2]->rows / 2.0);
						cv::line(image1, db_->keypoints_[idx1]->pts[id_pt1_local].pt + offset1,
							db_->keypoints_[idx2]->pts[id_pt2_local].pt + offset2, cv::Scalar(0), 1);
					}
					std::string path = "F:\\" + std::to_string(idx2) + "cuda.jpg";
					cv::imwrite(path, image1);
				}

				if (isOK && matches.size()>20)
				{
					WriteOutMatches(idx1, idx2, matches);
					match_graph_[idx1*num_imgs_ + idx2] = matches.size();
				}
			}

			of_match_index << idx1 << std::endl;
		}
		of_match_index.close();

		WriteOutMatchGraph();
	}


	bool MatchingGraphViaCombined::CheckMatchIndexFile()
	{
		std::string path = db_->output_fold_ + "//match_index.txt";
		std::ifstream infile(path);
		if (!infile.good())
		{
			return false;
		}

		return true;
	}

	std::vector<int> MatchingGraphViaCombined::CheckMissingMatchingFile()
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

	bool MatchingGraphViaCombined::GeoVerification(std::vector<cv::Point2f>& pt1, std::vector<cv::Point2f>& pt2,
		std::vector<int>& match_inliers)
	{
		if (pt1.size() < 30)
		{
			return false;
		}

		cv::Mat HMatrix1 = cv::findHomography(pt1, pt2);
		if (std::abs(HMatrix1.at<double>(0, 0) - 0.995) < 0.01 &&
			std::abs(HMatrix1.at<double>(1, 1) - 0.995) < 0.01 &&
			std::abs(HMatrix1.at<double>(2, 2) - 0.995) < 0.01)
		{
			return false;
		}

		float th_epipolar1 = 2.0;
		std::vector<uchar> ransac_status1(pt1.size());
		cv::findFundamentalMat(pt1, pt2, ransac_status1, cv::FM_RANSAC, th_epipolar1);
		for (size_t i = 0; i < ransac_status1.size(); i++)
		{
			if (ransac_status1[i])
			{
				match_inliers.push_back(i);
			}
		}

		if (match_inliers.size() < 30)
		{
			return false;
		}

		return true;
	}

	void MatchingGraphViaCombined::WriteOutMatches(int idx1, int idx2, std::vector<std::pair<int, int>>& matches)
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

	bool MatchingGraphViaCombined::CheckInitMatchGraph()
	{
		std::string path_init_match_graph = db_->output_fold_ + "//" + "init_match_graph.txt";
		std::ifstream infile(path_init_match_graph);
		if (!infile.is_open())
		{
			return false;
		}

		return true;
	}

	void MatchingGraphViaCombined::WriteOutInitMatchGraph(std::vector<std::vector<int>> &match_graph_init, int id_last)
	{
		std::string path = db_->output_fold_ + "//" + "init_match_graph.txt";
		std::ofstream ofs(path);
		if (!ofs.is_open())
		{
			return;
		}

		ofs << match_graph_init.size() << std::endl;
		ofs << id_last << std::endl;
		for (size_t i = 0; i < match_graph_init.size(); i++)
		{
			ofs << match_graph_init[i].size() << " ";
			for (size_t j = 0; j < match_graph_init[i].size(); j++)
			{
				ofs << match_graph_init[i][j]<< " ";
			}
			ofs << std::endl;
		}
		ofs.close();
	}

	void MatchingGraphViaCombined::ReadinInitMatchGraph(std::vector<std::vector<int>>& match_graph_init, int& id_last)
	{
		std::string path = db_->output_fold_ + "//" + "init_match_graph.txt";
		std::ifstream ifs(path);
		if (!ifs.is_open())
		{
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
				for (size_t j = 0; j < match_graph_init[i].size(); j++)
				{
					ifs >> match_graph_init[i][j];
				}
			}
		}
		ifs.close();
	}

	void MatchingGraphViaCombined::WriteOutMatchGraph()
	{
		std::string path = db_->output_fold_ + "//" + "graph_matching";
		std::ofstream ofs(path, std::ios::binary);

		if (!ofs.is_open())
		{
			return;
		}
		ofs.write((const char*)match_graph_, db_->num_imgs_*db_->num_imgs_ * sizeof(int));
		ofs.close();
	}

	void MatchingGraphViaCombined::RecoverMatchingGraph(std::vector<int>& existing_matches)
	{
		int num_imgs2 = num_imgs_ * num_imgs_;
		for (size_t i = 0; i < num_imgs2; i++) {
			match_graph_[i] = 0;
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

				match_graph_[idx * num_imgs_ + id] = num_match;
			}
			ifs.close();
		}
		
	}

	bool MatchingGraphViaCombined::CheckSimilarityGraph()
	{
		std::string path_sim_graph = db_->output_fold_ + "//" + "similarity_graph.txt";
		std::ifstream infile(path_sim_graph);
		if (!infile.is_open())
		{
			return false;
		}

		return true;
	}

	void MatchingGraphViaCombined::WriteOutSimilarityGraph(std::vector<std::vector<float>> &similarity_matrix)
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

	void MatchingGraphViaCombined::ReadinSimilarityGraph(std::vector<std::vector<float>> &similarity_matrix)
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