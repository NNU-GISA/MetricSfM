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

#include "matching_graph_word_querying.h"

#include <fstream>
#include <omp.h>

#include "utils/basic_funcs.h"
#include "utils/nanoflann_utils.h"
#include "utils/nanoflann.hpp"

using namespace nanoflann;

namespace objectsfm {

	MatchingGraphWordExpending::MatchingGraphWordExpending()
	{
	}

	MatchingGraphWordExpending::~MatchingGraphWordExpending()
	{
	}

	void MatchingGraphWordExpending::AssociateDatabase(Database * db)
	{
		db_ = db;
	}

	void MatchingGraphWordExpending::BuildMatchGraph()
	{
		num_imgs_ = db_->num_imgs_;
		match_graph_ = new int[num_imgs_*num_imgs_]();
		similarity_graph_ = new float[num_imgs_*num_imgs_]();

		// step1: build vocabulary tree
		BuildVocabularyTree();

		// step2: generate image descriptor and keypoint words
		if (!CheckWordFile())
		{
			GenerateImageWordFeatureViaExpending();
		}

		// step3: generate similarity matrix
		if (!CheckSimilarityGraph())
		{
			BuildSimilarityGraph();
		}

		// step4: do matching
		DoMatching();
	}

	void MatchingGraphWordExpending::BuildVocabularyTree()
	{
		// vocabulary tree
		db_->options.fbow_k = 10;
		db_->options.fbow_l = 6;
		if (db_->CheckVocabularyTreeExist())
		{
			db_->ReadinVocabularyTree();
		}
		else
		{
			std::cout << "---begin vocabulary generation" << std::endl;
			db_->BuildVocabularyTree();
			db_->WriteoutVocabularyTree();
		}
		std::cout << "---Number of blocks " << db_->voc_.size() << std::endl;
	}

	bool MatchingGraphWordExpending::CheckWordFile()
	{
		for (size_t i = 0; i < num_imgs_; i++)
		{
			std::string path_words = db_->output_fold_ + "//" + std::to_string(i) + "words";
			std::ifstream infile(path_words);
			if (!infile.is_open())
			{
				return false;
			}
		}
		return true;
	}

	void MatchingGraphWordExpending::GenerateImageWordFeatureViaQuerying()
	{
		typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, SiftList<float> >, SiftList<float>, 128/*dim*/ > my_kd_tree_t;

		// collect enough clusters
		std::vector<int> blocks_leaves;
		std::vector<int> blocks_current_level;
		std::vector<float*> feature_leaves;
		std::vector<float*> feature_current_level;

		blocks_current_level.push_back(0);
		while (blocks_leaves.size() + blocks_current_level.size() < 1000)
		{
			std::vector<int> blocks_current_level_new;
			std::vector<float*> feature_current_level_new;
			for (size_t i = 0; i < blocks_current_level.size(); i++)
			{
				int id_parent = blocks_current_level[i];
				fbow::Vocabulary::Block block_parent = db_->voc_.getBlock(id_parent);

				int num_N = block_parent.getN();
				for (size_t j = 0; j < num_N; j++)
				{
					fbow::Vocabulary::block_node_info *info_child = block_parent.getBlockNodeInfo(j);
					float* ptr_child = block_parent.getFeature<float>(j);
					if (ptr_child[0] != ptr_child[0])
					{
						continue;
					}

					int id_child = info_child->getId();
					if (info_child->isleaf())
					{
						blocks_leaves.push_back(id_child);
						feature_leaves.push_back(ptr_child);
					}
					else
					{
						blocks_current_level_new.push_back(id_child);
						feature_current_level_new.push_back(ptr_child);
					}
				}
			}
			blocks_current_level = blocks_current_level_new;
			feature_current_level.clear();
			feature_current_level = feature_current_level_new;
		}

		// build the level one
		std::vector<int> blocks_level_one;
		blocks_level_one.reserve(blocks_leaves.size() + blocks_current_level.size());
		blocks_level_one.insert(blocks_level_one.end(), blocks_leaves.begin(), blocks_leaves.end());
		blocks_level_one.insert(blocks_level_one.end(), blocks_current_level.begin(), blocks_current_level.end());

		std::vector<float*> feature_level_one;
		feature_level_one.reserve(feature_leaves.size() + feature_current_level.size());
		feature_level_one.insert(feature_level_one.end(), feature_leaves.begin(), feature_leaves.end());
		feature_level_one.insert(feature_level_one.end(), feature_current_level.begin(), feature_current_level.end());

		SiftList<float> data_level_one;
		for (size_t i = 0; i < blocks_level_one.size(); i++)
		{
			data_level_one.pts.push_back(SiftList<float>::SiftData(feature_level_one[i], i));
		}
		my_kd_tree_t* index_level_one = new my_kd_tree_t(128 /*dim*/, data_level_one, KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
		index_level_one->buildIndex();

		// build level two
		std::vector<SiftList<float> > data_level_two(blocks_level_one.size());
		std::vector<my_kd_tree_t*> index_level_two(blocks_level_one.size());
#pragma omp parallel for
		for (int i = 0; i < blocks_level_one.size(); i++)
		{
			int count = 0;
			std::vector<int> id_blocks;
			id_blocks.push_back(blocks_level_one[i]);
			while (count < id_blocks.size())
			{
				int id = id_blocks[count];
				fbow::Vocabulary::Block block = db_->voc_.getBlock(id);
				for (size_t j = 0; j < block.getN(); j++)
				{
					fbow::Vocabulary::block_node_info *info_j = block.getBlockNodeInfo(j);
					int id_j = info_j->getId();

					if (info_j->isleaf())
					{
						float* ptr_j = block.getFeature<float>(j);
						if (ptr_j[0] == ptr_j[0])
						{
							data_level_two[i].pts.push_back(SiftList<float>::SiftData(ptr_j, 0));
						}
					}
					else
					{
						id_blocks.push_back(id_j);
					}
				}
				count++;
			}

			index_level_two[i] = new my_kd_tree_t(128 /*dim*/, data_level_two[i], KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
			index_level_two[i]->buildIndex();
		}

		// generate image descriptor for each image and word for each feature point
		fbow::Vocabulary::Block block_root = db_->voc_.getBlock(0);
		int num_N = block_root.getN();
		int num_block_one = blocks_level_one.size();
		float* img_descriptor = new float[num_imgs_*num_block_one]();

#pragma omp parallel for
		for (int i = 0; i < num_imgs_; i++)
		{
			std::cout << i << std::endl;
			db_->ReadinImageFeatures(i);
			int num_pts = db_->descriptors_[i]->rows;

			float* img_descriptor_temp = img_descriptor + i * num_block_one;
			int* ptr_words_idx = new int[num_pts * 3];
			float* ptr_words_dis = new float[num_pts * 3];
			for (size_t j = 0; j < num_pts; j++)
			{
				// first query on root node
				size_t idx_root[1];
				float dis_root[1];
				nanoflann::KNNResultSet<float> result_root(1);
				result_root.init(idx_root, dis_root);
				float *ptr_desc_root = db_->descriptors_[i]->ptr<float>(j);
				index_level_one->findNeighbors(result_root, &ptr_desc_root[0], nanoflann::SearchParams(1));

				// second query on the level1 node
				int id_block_one = idx_root[0];
				img_descriptor_temp[id_block_one] += 1.0 / num_pts;

				size_t idx_i[3];
				float dis_i[3];
				nanoflann::KNNResultSet<float> result_i(3);
				result_i.init(idx_i, dis_i);
				float *ptr_desc_i = db_->descriptors_[i]->ptr<float>(j);
				index_level_two[id_block_one]->findNeighbors(result_i, &ptr_desc_i[0], nanoflann::SearchParams(3));

				// 
				ptr_words_idx[3 * j + 0] = idx_root[0] * num_N + idx_i[0];
				ptr_words_idx[3 * j + 1] = idx_root[0] * num_N + idx_i[1];
				ptr_words_idx[3 * j + 2] = idx_root[0] * num_N + idx_i[2];
				ptr_words_dis[3 * j + 0] = dis_i[0];
				ptr_words_dis[3 * j + 1] = dis_i[1];
				ptr_words_dis[3 * j + 2] = dis_i[2];
			}
			//WriteOutWords(i, num_pts, ptr_words_idx, ptr_words_dis);
			delete[] ptr_words_idx;
			delete[] ptr_words_dis;
		}
		
		// write out the descriptor of all the images
		std::string path_img_descriptor = db_->output_fold_  + "//descriptor";
		std::ofstream ofs_desc(path_img_descriptor, std::ios::binary);
		ofs_desc.write((const char*)(&num_block_one), sizeof(int));
		ofs_desc.write((const char*)img_descriptor, num_imgs_*num_block_one * sizeof(float));
		ofs_desc.close();

		delete[] img_descriptor;
	}

	void MatchingGraphWordExpending::GenerateImageWordFeatureViaExpending()
	{
#pragma omp parallel for
		for (int i = 0; i < num_imgs_; i++)
		{
			db_->ReadinImageFeatures(i);
			
			std::vector<std::vector<int>> ids;
			std::vector<std::vector<float>> dis;
			db_->voc_.transform(*db_->descriptors_[i], ids, dis);

			// 
			int N = 3;
			int num_pts = ids.size();
			int* ids_ptr = new int[num_pts*N];
			float* dis_ptr = new float[num_pts*N];
			for (size_t i = 0; i < num_pts; i++)
			{
				int* ids_ptr_cur = ids_ptr + i * N;
				float* dis_ptr_cur = dis_ptr + i * N;

				int j = 0;
				for (j = 0; j < ids[i].size(); j++)
				{
					if (ids[i][j] != ids[i][j] || dis[i][j] != dis[i][j])
					{
						ids_ptr_cur[j] = -1;
						dis_ptr_cur[j] = -1;
					}
					else
					{
						ids_ptr_cur[j] = ids[i][j];
						dis_ptr_cur[j] = dis[i][j];
					}
				}

				for (size_t p = j; p < N; p++)
				{
					ids_ptr_cur[p] = -1;
					dis_ptr_cur[p] = -1;
				}
			}
			db_->ReleaseImageFeatures(i);

			WriteOutWords(i, num_pts, N, ids_ptr, dis_ptr);

			delete[] ids_ptr;
			delete[] dis_ptr;
		}
	}

	void MatchingGraphWordExpending::WriteOutWords(int idx, int num_pts, int N, int * ids_ptr, float * dis_ptr)
	{
		std::string path_pts_words = db_->output_fold_ + "//" + std::to_string(idx) + "words";
		std::ofstream ofs_word(path_pts_words, std::ios::binary);
		ofs_word.write((const char*)(&num_pts), sizeof(int));
		ofs_word.write((const char*)(&N), sizeof(int));
		ofs_word.write((const char*)ids_ptr, num_pts * N * sizeof(int));
		ofs_word.write((const char*)dis_ptr, num_pts * N * sizeof(float));
		ofs_word.close();
	}

	void MatchingGraphWordExpending::ReadinWords(int idx, int & num_pts, int & N, int ** ids_ptr, float ** dis_ptr)
	{
		std::string path_pts_words = db_->output_fold_ + "//" + std::to_string(idx) + "words";
		std::ifstream ifs_word(path_pts_words, std::ios::binary);

		ifs_word.read((char*)(&num_pts), sizeof(int));
		ifs_word.read((char*)(&N), sizeof(int));

		*ids_ptr = new int[num_pts * N];
		*dis_ptr = new float[num_pts * N];

		ifs_word.read((char*)*ids_ptr, num_pts * N * sizeof(int));
		ifs_word.read((char*)*dis_ptr, num_pts * N * sizeof(float));

		ifs_word.close();
	}

	void MatchingGraphWordExpending::WriteOutWordPtMap(int idx, int num, int * word_pt_ptr)
	{
		std::string path_pts_words = db_->output_fold_ + "//" + std::to_string(idx) + "words_pt";
		std::ofstream ofs_word(path_pts_words, std::ios::binary);
		ofs_word.write((const char*)(&num), sizeof(int));
		ofs_word.write((const char*)word_pt_ptr, 2 * num * sizeof(int));
		ofs_word.close();
	}

	void MatchingGraphWordExpending::ReadinWordPtMap(int idx, int & num, int ** word_pt_ptr)
	{
		std::string path_pts_words = db_->output_fold_ + "//" + std::to_string(idx) + "words_pt";
		std::ifstream ifs_word(path_pts_words, std::ios::binary);

		ifs_word.read((char*)(&num), sizeof(int));

		*word_pt_ptr = new int[2*num];

		ifs_word.read((char*)*word_pt_ptr, 2 * num * sizeof(int));

		ifs_word.close();
	}

	bool MatchingGraphWordExpending::CheckSimilarityGraph()
	{
		for (size_t i = 0; i < num_imgs_; i++)
		{
			std::string path_words = db_->output_fold_ + "//" + std::to_string(i) + "words_pt";
			std::ifstream infile(path_words);
			if (!infile.is_open())
			{
				return false;
			}
		}

		std::string path_sim_graph = db_->output_fold_ + "//" + "similarity_graph";
		std::ifstream infile(path_sim_graph);
		if (!infile.is_open())
		{
			return false;
		}

		return true;
	}

	void MatchingGraphWordExpending::BuildSimilarityGraph()
	{
		// generate word list of each image
		std::vector<std::vector<int>> words_per_img(num_imgs_);
#pragma omp parallel for
		for (int i = 0; i < num_imgs_; i++)
		{
			std::cout << i << std::endl;
			int num_pts, N;
			int* ids;
			float *dis;
			ReadinWords(i, num_pts, N, &ids, &dis);

			std::map<int, int> word_closest_id;
			std::map<int, float> word_closest_dis;
			std::vector<int> words_cur_img;
			for (size_t j = 0; j < num_pts; j++)
			{
				int* ids_j = ids + j * N;
				float *dis_j = dis + j * N;

				//for (size_t m = 0; m < N; m++)
				for (size_t m = 0; m < 1; m++)
				{
					int id_m = ids_j[m];
					float dis_m = dis_j[m];
					if (id_m < 0)
					{
						continue;
					}

					auto iter_id = word_closest_id.find(id_m);
					if (iter_id != word_closest_id.end())
					{
						auto iter_dis = word_closest_dis.find(id_m);
						if (dis_m < iter_dis->second)
						{
							iter_id->second = j;
							iter_dis->second = dis_m;
						}
					}
					else
					{
						word_closest_id.insert(std::pair<int, int>(id_m, j));
						word_closest_dis.insert(std::pair<int, float>(id_m, dis_m));
						words_cur_img.push_back(id_m);
					}
				}
			}

			// write out
			int* word_pt_ptr = new int[2*word_closest_id.size()];
			int j = 0;
			for (auto iter : word_closest_id)
			{
				word_pt_ptr[2 * j + 0] = iter.first;
				word_pt_ptr[2 * j + 1] = iter.second;
				j++;
			}
			WriteOutWordPtMap(i, word_closest_id.size(), word_pt_ptr);

			//
			words_per_img[i] = words_cur_img;
			delete[] ids;
			delete[] dis;
		}

		// calculate similarity matrix
		for (size_t i = 0; i < num_imgs_; i++)
		{
			std::sort(words_per_img[i].begin(), words_per_img[i].end(), [](const int &lhs, const int &rhs) { return lhs < rhs; });
		}

		for (size_t i = 0; i < num_imgs_-1; i++)
		{
			std::cout << i << std::endl;
#pragma omp parallel for
			for (int j = i+1; j < num_imgs_; j++)
			{
				float sim = math::simOfBows(words_per_img[i], words_per_img[j]);
				similarity_graph_[i*num_imgs_ + j] = sim;
				similarity_graph_[j*num_imgs_ + i] = sim;
			}
		}
		WriteOutSimilarityGraph();
	}

	void MatchingGraphWordExpending::WriteOutSimilarityGraph()
	{
		std::string path_pts_words = db_->output_fold_ + "//" + "similarity_graph";
		std::ofstream ofs_word(path_pts_words, std::ios::binary);
		ofs_word.write((const char*)similarity_graph_, num_imgs_ * num_imgs_ * sizeof(float));
		ofs_word.close();
	}

	void MatchingGraphWordExpending::ReadinSimilarityGraph()
	{
		std::string path_pts_words = db_->output_fold_ + "//" + "similarity_graph";
		std::ifstream ifs_word(path_pts_words, std::ios::binary);

		ifs_word.read((char*)similarity_graph_, num_imgs_ * num_imgs_ * sizeof(float));
		ifs_word.close();
	}

	void MatchingGraphWordExpending::DoMatching()
	{
		// step1: read in information
		std::vector<std::map<int, int>> word_pt_maps(num_imgs_);
		for (size_t i = 0; i < num_imgs_; i++)
		{
			int num = 0;
			int* ptr_word_pt;
			ReadinWordPtMap(i, num, &ptr_word_pt);

			int *ptr_word_pt_temp = ptr_word_pt;
			for (size_t j = 0; j < num/2; j++)
			{
				word_pt_maps[i].insert(std::pair<int,int>(ptr_word_pt_temp[0], ptr_word_pt_temp[1]));
				ptr_word_pt_temp += 2;
			}
			delete[] ptr_word_pt;
		}
		ReadinSimilarityGraph();

		// step2: do matching for each image
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
			db_->ReadinImageFeatures(idx1);

			// find matching hypotheses
			std::vector<std::pair<int, float>> sim_sort(num_imgs_);
			for (size_t j = 0; j < num_imgs_; j++)
			{
				if (j != idx1)
				{
					sim_sort[j].first = j;
					sim_sort[j].second = similarity_graph_[i*num_imgs_+j];
				}
			}
			std::sort(sim_sort.begin(), sim_sort.end(), [](const std::pair<int, float> &lhs, const std::pair<int, float> &rhs) { return lhs.second > rhs.second; });

			// do matching
			for (size_t j = 0; j < th_num_match; j++)
			{
				int idx2 = sim_sort[j].first;
				db_->ReadinImageFeatures(idx2);

				// initial matching via words
				std::vector<std::pair<int, int>> matches_init;
				for (auto iter1 : word_pt_maps[idx1])
				{
					int id_word = iter1.first;
					int id_pt1 = iter1.second;
					auto iter2 = word_pt_maps[idx2].find(id_word);
					if (iter2 != word_pt_maps[idx2].end())
					{
						int id_pt2 = iter2->second;
						matches_init.push_back(std::pair<int, int>(id_pt1, id_pt2));
					}
				}
				if (matches_init.size() < 30)
				{
					db_->ReleaseImageFeatures(idx2);
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
					db_->ReleaseImageFeatures(idx2);
					continue;
				}

				std::vector<std::pair<int, int>> matches;
				for (size_t m = 0; m < match_init_inliers.size(); m++)
				{
					int index1 = matches_init[match_init_inliers[m]].first;
					int index2 = matches_init[match_init_inliers[m]].second;
					matches.push_back(std::pair<int, int>(index1, index2));
				}
				std::cout << idx1 << "  " << idx2 << "   matches: " << matches.size() << std::endl;

				// draw
				if (1)
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

				//
				WriteOutMatches(idx1, idx2, matches);
				match_graph_[idx1*num_imgs_ + idx2] = matches.size();
				db_->ReleaseImageFeatures(idx2);
			}

			of_match_index << idx1 << std::endl;
			db_->ReleaseImageFeatures(idx1);
		}
		of_match_index.close();
		WriteOutMatchGraph();

		delete[] match_graph_;
	}

	bool MatchingGraphWordExpending::GeoVerification(std::vector<cv::Point2f>& pt1, std::vector<cv::Point2f>& pt2, std::vector<int>& match_inliers)
	{
		// iter 1
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

		float th_epipolar1 = 3.0;
		std::vector<uchar> ransac_status1(pt1.size());
		cv::findFundamentalMat(pt1, pt2, ransac_status1, cv::FM_RANSAC, th_epipolar1);
		std::vector<cv::Point2f> pt1_, pt2_;
		std::vector<int> index;
		for (size_t i = 0; i < ransac_status1.size(); i++)
		{
			if (ransac_status1[i])
			{
				pt1_.push_back(pt1[i]);
				pt2_.push_back(pt2[i]);
				index.push_back(i);
			}
		}

		// iter 2
		if (pt1_.size() < 30)
		{
			return false;
		}

		cv::Mat HMatrix2 = cv::findHomography(pt1_, pt2_);
		if (std::abs(HMatrix2.at<double>(0, 0) - 0.995) < 0.01 &&
			std::abs(HMatrix2.at<double>(1, 1) - 0.995) < 0.01 &&
			std::abs(HMatrix2.at<double>(2, 2) - 0.995) < 0.01)
		{
			return false;
		}

		float th_epipolar2 = 1.0;
		std::vector<uchar> ransac_status2(pt1_.size());
		cv::findFundamentalMat(pt1_, pt2_, ransac_status2, cv::FM_RANSAC, th_epipolar2);
		for (size_t i = 0; i < ransac_status2.size(); i++)
		{
			if (ransac_status2[i])
			{
				match_inliers.push_back(index[i]);
			}
		}

		if (match_inliers.size() < 30)
		{
			return false;
		}

		return true;
	}

	bool MatchingGraphWordExpending::CheckMatchIndexFile()
	{
		std::string path = db_->output_fold_ + "//match_index.txt";
		std::ifstream infile(path);
		if (!infile.good())
		{
			return false;
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
				return false;
			}
		}

		return true;
	}

	void MatchingGraphWordExpending::WriteOutMatches(int idx1, int idx2, std::vector<std::pair<int, int>>& matches)
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

	void MatchingGraphWordExpending::WriteOutMatchGraph()
	{
		std::string path = db_->output_fold_ + "//" + "graph_matching";
		std::ofstream ofs(path, std::ios::binary);

		if (!ofs.is_open())
		{
			return;
		}
		ofs.write((const char*)match_graph_, num_imgs_*num_imgs_ * sizeof(int));
		ofs.close();
	}

}