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

#include "matching_graph_via_feature.h"

#include <fstream>
#include <omp.h>
#include <boost/thread.hpp>

#include "utils/basic_funcs.h"

#include "feature/feature_matching.h"
#include "feature/feature_matching_cuda_sift.h"

#include "cudaSift/image.h"
#include "cudaSift/sift.h"
#include "cudaSift/utils.h"

#include "graph/similarity_graph_bow_distance.h"
#include "graph/similarity_graph_word_number.h"

#include "flann/flann.h"

namespace objectsfm {

	MatchingGraphViaFeature::MatchingGraphViaFeature()
	{
	}

	MatchingGraphViaFeature::~MatchingGraphViaFeature()
	{
	}

	void MatchingGraphViaFeature::AssociateDatabase(Database * db)
	{
		db_ = db;
	}

	void MatchingGraphViaFeature::BuildMatchGraph()
	{
		num_imgs_ = db_->num_imgs_;
		match_graph_ = new int[num_imgs_*num_imgs_];

		bool match_file_ok = CheckMatchIndexFile();
		if (match_file_ok)
		{
			return;
		}

		// step1: generate similarity matrix
		if (options_.sim_graph_type == "BoWDistance")  // use bow similarity
		{
			SimilarityGraphBowDistance simer;
			simer.AssociateDatabase(db_);
			simer.BuildSimilarityGraph(similarity_matrix_);
		}
		else  // use inverted file
		{
			SimilarityGraphWordNumber simer;
			simer.AssociateDatabase(db_);
			simer.BuildSimilarityGraph(similarity_matrix_);
		}

		// 
		if (options_.use_gpu)
		{
			MatchingWithGPU();
		}
		else
		{
			//MatchingWithCPU_OpenCV();
			//MatchingWithCPU_KDTree();	
			MatchingWithCPU_KDTree_Flann();
		}
		
		//
		WriteOutMatchGraph();

		delete[] match_graph_;
	}

	void MatchingGraphViaFeature::MatchingWithCPU_OpenCV()
	{
		int th_num_match = MIN(MAX(50, num_imgs_ / 10), num_imgs_);

		for (size_t i = 0; i < num_imgs_; i++)
		{
			int idx1 = i;
			//idx1 = 142;

			// find matching hypotheses
			std::vector<std::pair<int, float>> sim_sort;
			for (size_t j = 0; j < num_imgs_; j++)
			{
				match_graph_[idx1*num_imgs_ + j] = 0;
				if (j == idx1 || similarity_matrix_[i][j] <= 0)
				{
					continue;
				}
				sim_sort.push_back(std::pair<int, float>(j, similarity_matrix_[i][j]));
			}
			std::sort(sim_sort.begin(), sim_sort.end(), [](const std::pair<int, float> &lhs, const std::pair<int, float> &rhs) { return lhs.second > rhs.second; });
			th_num_match = MIN(sim_sort.size(), th_num_match);

			std::vector<int> set_idx2(th_num_match);
			for (size_t j = 0; j < th_num_match; j++)
			{
				int idx2 = sim_sort[j].first;
				set_idx2[j] = idx2;
			}

			// generate kd-tree for idx1
			db_->ReadinImageFeatures(idx1);
			cv::flann::Index* kd_tree;
			FeatureMatching::GenerateKDIndex(*db_->descriptors_[idx1], &kd_tree);

			// do matching
			std::vector<std::vector<std::pair<int, int>>> set_matches(set_idx2.size());
			for (size_t j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];
				//idx2 = 143;

				db_->ReadinImageFeatures(idx2);
				bool isOK = FeatureMatching::KNNMatchingWithGeoVerify(db_->keypoints_[idx1]->pts, kd_tree,
					db_->keypoints_[idx2]->pts, *db_->descriptors_[idx2], set_matches[j]);
				db_->ReleaseImageFeatures(idx2);
				std::cout << idx1 << "  " << idx2 << "   matches: " << set_matches[j].size() << std::endl;
			}

			// write out matching result
			for (int j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];
				if (set_matches[j].size())
				{
					WriteOutMatches(idx1, idx2, set_matches[j]);
					match_graph_[idx1*num_imgs_ + idx2] = set_matches[j].size();
				}
			}

			db_->ReleaseImageFeatures(idx1);
		}
	}

	void MatchingGraphViaFeature::MatchingWithCPU_OpenCV_OneThread(std::vector<int> ids)
	{
		//int num_threads = omp_get_max_threads() - 1;

		//std::vector<std::vector<int>> ids_per_thread(num_threads);
		//for (size_t i = 0; i < num_imgs_; i++)
		//{
		//	int id_thread = i % num_threads;
		//	ids_per_thread[id_thread].push_back(i);
		//}

		//boost::thread_group thread_group;
		//for (size_t i = 0; i < num_threads; i++)
		//{
		//	std::cout << i << std::endl;
		//	boost::function0< void> f = boost::bind(&MatchingGraphViaFeature::MatchingWithCPU_OpenCV_OneThread, this, ids_per_thread[i]);
		//	thread_group.create_thread(f);
		//}
		//thread_group.join_all();

		//
		int th_num_match = MIN(MAX(50, num_imgs_ / 10), num_imgs_);

		for (size_t i = 0; i < ids.size(); i++)
		{
			int idx1 = ids[i];

			// find matching hypotheses
			std::vector<std::pair<int, float>> sim_sort;
			for (size_t j = 0; j < num_imgs_; j++)
			{
				match_graph_[idx1*num_imgs_ + j] = 0;
				if (j == idx1 || similarity_matrix_[i][j] <= 0)
				{
					continue;
				}
				sim_sort.push_back(std::pair<int, float>(j, similarity_matrix_[i][j]));
			}
			std::sort(sim_sort.begin(), sim_sort.end(), [](const std::pair<int, float> &lhs, const std::pair<int, float> &rhs) { return lhs.second > rhs.second; });
			th_num_match = MIN(sim_sort.size(), th_num_match);

			std::vector<int> set_idx2(th_num_match);
			for (size_t j = 0; j < th_num_match; j++)
			{
				int idx2 = sim_sort[j].first;
				set_idx2[j] = idx2;
			}

			// generate kd-tree for idx1
			db_->ReadinImageFeatures(idx1);
			cv::flann::Index* kd_tree;
			FeatureMatching::GenerateKDIndex(*db_->descriptors_[idx1], &kd_tree);

			// do matching
			std::vector<std::vector<std::pair<int, int>>> set_matches(set_idx2.size());
			for (size_t j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];
				db_->ReadinImageFeatures(idx2);
				bool isOK = FeatureMatching::KNNMatchingWithGeoVerify(db_->keypoints_[idx1]->pts, kd_tree,
					db_->keypoints_[idx2]->pts, *db_->descriptors_[idx2], set_matches[j]);
				db_->ReleaseImageFeatures(idx2);
				std::cout << idx1 << "  " << idx2 << "   matches: " << set_matches[j].size() << std::endl;
			}

			// write out matching result
			for (int j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];
				if (set_matches[j].size())
				{
					WriteOutMatches(idx1, idx2, set_matches[j]);
					match_graph_[idx1*num_imgs_ + idx2] = set_matches[j].size();
				}
			}

			db_->ReleaseImageFeatures(idx1);
		}
	}

	void MatchingGraphViaFeature::MatchingWithCPU_KDTree_Nanoflann()
	{
		int th_num_match = MAX(50, num_imgs_ / 10);

		std::string file_match = db_->output_fold_ + "//match_index.txt";
		std::ofstream of_match_index(file_match, std::ios::app);
		for (size_t i = 0; i < num_imgs_; i++)
		{
			//if (i % 5 == 0)
			{
				std::cout << "---Matching images " << i << "/" << num_imgs_ << std::endl;
			}
			int idx1 = i;

			// find matching hypotheses
			std::vector<std::pair<int, float>> sim_sort;
			for (size_t j = 0; j < num_imgs_; j++)
			{
				match_graph_[idx1*num_imgs_ + j] = 0;
				if (j == idx1 || similarity_matrix_[i][j] <= 0)
				{
					continue;
				}
				sim_sort.push_back(std::pair<int, float>(j, similarity_matrix_[i][j]));
			}
			std::sort(sim_sort.begin(), sim_sort.end(), [](const std::pair<int, float> &lhs, const std::pair<int, float> &rhs) { return lhs.second > rhs.second; });
			th_num_match = MIN(sim_sort.size(), th_num_match);

			std::vector<int> set_idx2(th_num_match);
			for (size_t j = 0; j < th_num_match; j++)
			{
				int idx2 = sim_sort[j].first;
				set_idx2[j] = idx2;
			}

			// generate kd-tree for idx1
			db_->ReadinImageFeatures(idx1);
			SiftList<float> data_idx1;
			for (size_t j = 0; j < db_->descriptors_[idx1]->rows; j++)
			{
				data_idx1.pts.push_back(SiftList<float>::SiftData((float*)db_->descriptors_[idx1]->ptr<float>(j), j));
			}
			my_kd_tree_t* kdtree1 = new my_kd_tree_t(128 /*dim*/, data_idx1, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
			kdtree1->buildIndex();

			// matching via querying
			std::vector<std::vector<std::pair<int, int>>> set_matches(set_idx2.size());
//#pragma omp parallel
			for (int j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];
				db_->ReadinImageFeatures(idx2);

				bool isOK = FeatureMatching::KNNMatchingWithGeoVerify(db_->keypoints_[idx1]->pts, kdtree1,
					db_->keypoints_[idx2]->pts, *db_->descriptors_[idx2], set_matches[j]);
				db_->ReleaseImageFeatures(idx2);
				std::cout << idx1 << "  " << idx2 << "   matches: " << set_matches[j].size() << std::endl;

				// draw
				if (0)
				{
					db_->ReadinImageFeatures(idx2);
					cv::Mat image1 = cv::imread(db_->image_paths_[idx1]);
					cv::Mat image2 = cv::imread(db_->image_paths_[idx2]);
					int pitch = 128;
					cv::resize(image1, image1, cv::Size((image1.cols / pitch + 1) * pitch, (image1.rows / pitch + 1) * pitch));
					cv::resize(image2, image2, cv::Size((image2.cols / pitch + 1) * pitch, (image2.rows / pitch + 1) * pitch));
					for (size_t m = 0; m < set_matches[j].size(); m++)
					{
						int id_pt1_local = set_matches[j][m].first;
						int id_pt2_local = set_matches[j][m].second;

						cv::Point2f offset1(db_->image_infos_[idx1]->cols / 2.0, db_->image_infos_[idx1]->rows / 2.0);
						cv::Point2f offset2(db_->image_infos_[idx2]->cols / 2.0, db_->image_infos_[idx2]->rows / 2.0);
						cv::line(image1, db_->keypoints_[idx1]->pts[id_pt1_local].pt + offset1,
							db_->keypoints_[idx2]->pts[id_pt2_local].pt + offset2, cv::Scalar(0), 1);
					}

					std::string path = "F:\\" + std::to_string(idx2) + "cuda.jpg";
					cv::imwrite(path, image1);
				}
			}

			for (int j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];
				if (set_matches[j].size())
				{
					WriteOutMatches(idx1, idx2, set_matches[j]);
					match_graph_[idx1*num_imgs_ + idx2] = set_matches[j].size();
				}
			}
			//
			of_match_index << idx1 << std::endl;
		}
		of_match_index.close();

		exit(0);
	}

	void MatchingGraphViaFeature::MatchingWithCPU_KDTree_Flann()
	{
		//int th_num_match = MAX(50, num_imgs_ / 10);
		int th_num_match = MIN(100, num_imgs_);

		std::string file_match = db_->output_fold_ + "//match_index.txt";
		std::ofstream of_match_index(file_match, std::ios::app);
		for (size_t i = 0; i < num_imgs_; i++)
		{
			//if (i % 5 == 0)
			{
				std::cout << "---Matching images " << i << "/" << num_imgs_ << std::endl;
			}
			int idx1 = i;

			// find matching hypotheses
			std::vector<std::pair<int, float>> sim_sort;
			for (size_t j = 0; j < num_imgs_; j++)
			{
				match_graph_[idx1*num_imgs_ + j] = 0;
				if (j == idx1 || similarity_matrix_[i][j] <= 0)
				{
					continue;
				}
				sim_sort.push_back(std::pair<int, float>(j, similarity_matrix_[i][j]));
			}
			std::sort(sim_sort.begin(), sim_sort.end(), [](const std::pair<int, float> &lhs, const std::pair<int, float> &rhs) { return lhs.second > rhs.second; });
			th_num_match = MIN(sim_sort.size(), th_num_match);

			std::vector<int> set_idx2(th_num_match);
			for (size_t j = 0; j < th_num_match; j++)
			{
				int idx2 = sim_sort[j].first;
				set_idx2[j] = idx2;
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
			flann_index_t kdtree_idx1;
			kdtree_idx1 = flann_build_index(data_idx1, db_->descriptors_[idx1]->rows, db_->descriptors_[idx1]->cols, &speedup, &p);

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
	}

	void MatchingGraphViaFeature::MatchingWithGPU()
	{
		int th_num_match = MIN(MAX(50, num_imgs_ / 10), num_imgs_);

		std::string file_match = db_->output_fold_ + "//match_index.txt";
		std::ofstream of_match_index(file_match, std::ios::app);
		for (size_t i = 0; i < num_imgs_; i++)
		{
			if (i % 5 == 0)
			{
				std::cout << "---Matching images " << i << "/" << num_imgs_ << std::endl;
			}
			int idx1 = i;
			
			// find matching hypotheses
			std::vector<std::pair<int, float>> sim_sort(num_imgs_);
			for (size_t j = 0; j < num_imgs_; j++)
			{
				match_graph_[idx1*num_imgs_ + j] = 0;
				if (j != idx1)
				{
					sim_sort[j].first = j;
					sim_sort[j].second = similarity_matrix_[i][j];
				}
			}
			std::sort(sim_sort.begin(), sim_sort.end(), [](const std::pair<int, float> &lhs, const std::pair<int, float> &rhs) { return lhs.second > rhs.second; });

			std::vector<int> set_idx2(th_num_match);
			for (size_t j = 0; j < th_num_match; j++)
			{
				int idx2 = sim_sort[j].first;
				set_idx2[j] = idx2;
			}

			// 
			db_->ReadinImageFeatures(idx1);
			cudaSift::SiftData *siftdata1 = NULL;
			siftdata1 = new cudaSift::SiftData;
			FeatureMatchingCudaSift::DataConvert(db_->keypoints_[idx1]->pts, (*db_->descriptors_[idx1]), siftdata1);

			for (int j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];
				if (similarity_matrix_[idx1][idx2] == 0)
				{
					continue;
				}
				db_->ReadinImageFeatures(idx2);

				std::vector<std::pair<int, int>> matches;
				cudaSift::SiftData *siftdata2 = new cudaSift::SiftData;
				FeatureMatchingCudaSift::DataConvert(db_->keypoints_[idx2]->pts, (*db_->descriptors_[idx2]), siftdata2);
				FeatureMatchingCudaSift::Run(siftdata1, siftdata2, matches);
				cudaSift::FreeSiftData(*siftdata2);
				db_->ReleaseImageFeatures(idx2);

				if (matches.size() < 20)
				{
					continue;
				}
				std::cout << idx1 << "  " << idx2 << "   matches: " << matches.size() << std::endl;

				// draw
				if (0)
				{
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

				WriteOutMatches(idx1, idx2, matches);
				match_graph_[idx1*num_imgs_ + idx2] = matches.size();
			}
			cudaSift::FreeSiftData(*siftdata1);
			db_->ReadinImageFeatures(idx1);

			//
			of_match_index << idx1 << std::endl;
		}
		of_match_index.close();
	}


	void MatchingGraphViaFeature::GenerateInvertedFile(std::vector<std::vector<int>> &inverted_file)
	{
		inverted_file.resize(num_words_);

		// condition 1
		for (size_t i = 0; i < num_imgs_; i++)
		{
			std::vector<int> word_ids_temp = db_->words_id_[i];
			math::keep_unique_vector(word_ids_temp);
			for (size_t j = 0; j < word_ids_temp.size(); j++)
			{
				int id = word_ids_temp[j];
				inverted_file[id].push_back(i);
			}
			db_->ReleaseWordsForImage(i);
		}

		// condition 2
		int th_bin_size = num_words_ / 100;
		for (size_t i = 0; i < num_words_; i++)
		{
			if (!inverted_file[i].empty() && inverted_file[i].size() > th_bin_size)
			{
				inverted_file[i].clear();
			}
		}
	}

	void MatchingGraphViaFeature::GenerateSimilarityMatrix(std::vector<std::vector<int>> &inverted_file)
	{
		similarity_matrix_.resize(num_imgs_);
		for (size_t i = 0; i < num_imgs_; i++)
		{
			similarity_matrix_[i].resize(num_imgs_, 0);
		}
		for (size_t i = 0; i < num_words_; i++)
		{
			if (inverted_file[i].size())
			{
				for (size_t m = 0; m < inverted_file[i].size() - 1; m++)
				{
					int id_img1 = inverted_file[i][m];
					for (size_t n = m + 1; n < inverted_file[i].size(); n++)
					{
						int id_img2 = inverted_file[i][n];
						similarity_matrix_[id_img1][id_img2]++;
						similarity_matrix_[id_img2][id_img1]++;
					}
				}
			}
		}
	}

	bool MatchingGraphViaFeature::CheckMatchIndexFile()
	{
		std::string path = db_->output_fold_ + "//match_index.txt";
		std::ifstream infile(path);
		if (!infile.good())
		{
			return false;
		}

		std::vector<int> index(db_->num_imgs_, 0);
		int idx = -1;
		while (!infile.eof())
		{
			infile >> idx;
			if (idx >= 0)
			{
				index[idx] = 1;
			}
		}

		for (size_t i = 0; i < db_->num_imgs_; i++)
		{
			if (!index[i])
			{
				return false;
			}
		}

		return true;
	}

	void MatchingGraphViaFeature::WriteOutMatches(int idx1, int idx2, std::vector<std::pair<int, int>>& matches)
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

	void MatchingGraphViaFeature::WriteOutMatchGraph()
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
}