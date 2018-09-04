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

#include "matching_graph_via_invertedfile.h"

#include <fstream>

#include "utils/basic_funcs.h"

namespace objectsfm {

	MatchingGraphInvertedFile::MatchingGraphInvertedFile()
	{
	}

	MatchingGraphInvertedFile::~MatchingGraphInvertedFile()
	{
	}

	void MatchingGraphInvertedFile::AssociateDatabase(Database * db)
	{
		db_ = db;
	}

	void MatchingGraphInvertedFile::BuildMatchGraph()
	{
		num_imgs_ = db_->num_imgs_;
		match_graph_ = new int[num_imgs_*num_imgs_];

		bool match_file_ok = CheckMatchIndexFile();
		if (match_file_ok)
		{
			return;
		}

		// step1: build words for each image first
		db_->BuildWords();
		for (size_t i = 0; i < num_imgs_; i++)
		{
			db_->ReadinWordsForImage(i);
		}
		num_words_ = db_->max_words_id_;
		std::cout << "number of words: "<<num_words_ << std::endl;

		// step2: generate inverted file
		GenerateInvertedFile();

		// step3: generate adjacent matrix
		std::vector<std::vector<int>> adjacency_matrix;
		GenerateAdjacentMatrix(adjacency_matrix);

		// now can delete the inverted file
		int count = 0;
		std::vector<int> is_word_valid(num_words_, 0);
		for (size_t i = 0; i < num_words_; i++)
		{
			if (inverted_file_[i].size() >= 2)
			{
				is_word_valid[i] = 1;
				count++;
				std::vector<int>().swap(inverted_file_[i]);
			}
		}

		// step4: generate point_id to word_id map for each image
		std::vector<std::map<int, int>> pt_word_map(num_imgs_);
		for (size_t i = 0; i < num_imgs_; i++)
		{
			std::vector<int> unique_word_idxs;
			math::keep_unique_idx_vector(db_->words_id_[i], unique_word_idxs);
			for (size_t j = 0; j < unique_word_idxs.size(); j++)
			{
				int idx = unique_word_idxs[j];
				int id_word = db_->words_id_[i][idx];
				pt_word_map[i].insert(std::pair<int,int>(id_word, idx));
			}
		}

		// step5: do matching via quarying
		std::string file_match = db_->output_fold_ + "//match_index.txt";
		std::ofstream of_match_index(file_match, std::ios::app);
		//for (size_t i = 0; i < num_imgs_; i++)
		for (size_t i = 0; i < 1; i++)
		{
			int idx1 = i;
			db_->ReadinImageFeatures(idx1);

			// find matching hypotheses
			std::vector<int> set_idx2;
			for (size_t j = 0; j < num_imgs_; j++)
			{
				int idx2 = j;
				match_graph_[idx1*num_imgs_ + idx2] = 0;
				//if (idx1 != idx2 && adjacency_matrix[idx1][idx2])
				if (idx1 != idx2)
				{
					set_idx2.push_back(idx2);
				}
			}
			std::cout << idx1 << std::endl;
			
			// matching parallelly
			std::cout << "---Matching images " << idx1 << "  hypotheses " << set_idx2.size() << std::endl;
			std::vector<std::vector<std::pair<int, int>>> set_matches(set_idx2.size());
//#pragma omp parallel for
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
				db_->ReadinImageFeatures(idx2);
				std::vector<int> match_init_inliers;
				std::vector<cv::Point2f> pts1, pts2;
				for (size_t m = 0; m < matches_init.size(); m++)
				{
					int index1 = matches_init[m].first;
					int index2 = matches_init[m].second;
					pts1.push_back(db_->keypoints_[idx1]->pts[index1].pt);
					pts2.push_back(db_->keypoints_[idx2]->pts[index2].pt);
				}
				//db_->ReleaseImageFeatures(idx2);

				if (!GeoVerification(pts1, pts2, match_init_inliers))
				{
					continue;
				}

				std::vector<std::pair<int, int>> matches;
				for (size_t m = 0; m < match_init_inliers.size(); m++)
				{
					int index1 = matches_init[match_init_inliers[m]].first;
					int index2 = matches_init[match_init_inliers[m]].second;
					matches.push_back(std::pair<int, int>(index1, index2));
				}
				set_matches[j] = matches;

				std::cout << idx1 << "  " << idx2 << "   matches: " << matches.size() << std::endl;

				//
				if (idx1==0)
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

					std::string path = "F:\\" + std::to_string(idx2) + "voc.jpg";
					cv::imwrite(path, image1);
				}
			}

			for (int j = 0; j < set_idx2.size(); j++)
			{
				int idx2 = set_idx2[j];
				WriteOutMatches(idx1, idx2, set_matches[j]);
				match_graph_[idx1*num_imgs_ + idx2] = set_matches[j].size();
			}
			of_match_index << idx1 << std::endl;
			/*
			// draw
			db_->ReadinImageFeatures(idx2);
			cv::Mat img1 = cv::imread(db_->image_paths_[idx1]);
			cv::Mat img2 = cv::imread(db_->image_paths_[idx2]);
			for (size_t m = 0; m < matches.size(); m++)
			{
				int index1 = matches[m].first;
				int index2 = matches[m].second;
				cv::Point2f offset1(db_->image_infos_[idx1]->cols / 2.0, db_->image_infos_[idx1]->rows / 2.0);
				cv::Point2f offset2(db_->image_infos_[idx2]->cols / 2.0, db_->image_infos_[idx2]->rows / 2.0);
				cv::line(img1, db_->keypoints_[idx1]->pts[index1].pt + offset1,
					db_->keypoints_[idx2]->pts[index2].pt + offset2,
					cv::Scalar(0, 0, 255), 1);
			}
			std::string out_draw = "F:\\matches.jpg";
			cv::imwrite(out_draw, img1);
			*/
		}
		of_match_index.close();

		//
		WriteOutMatchGraph();

		delete[] match_graph_;
	}

	void MatchingGraphInvertedFile::GenerateInvertedFile()
	{
		inverted_file_.resize(num_words_);

		// condition 1
		for (size_t i = 0; i < num_imgs_; i++)
		{
			std::vector<int> word_ids_temp = db_->words_id_[i];
			math::keep_unique_vector(word_ids_temp);
			for (size_t j = 0; j < word_ids_temp.size(); j++)
			{
				int id = word_ids_temp[j];
				inverted_file_[id].push_back(i);
			}
		}

		// condition 2
		int th_bin_size = num_words_ / 100;
		for (size_t i = 0; i < num_words_; i++)
		{
			if (!inverted_file_[i].empty() && inverted_file_[i].size() > th_bin_size)
			{
				inverted_file_[i].clear();
			}
		}
	}

	void MatchingGraphInvertedFile::GenerateAdjacentMatrix(std::vector<std::vector<int>> &adjacency_matrix)
	{
		adjacency_matrix.resize(num_imgs_);
		for (size_t i = 0; i < num_imgs_; i++)
		{
			adjacency_matrix[i].resize(num_imgs_, 0);
		}
		for (size_t i = 0; i < num_words_; i++)
		{
			if (inverted_file_[i].size())
			{
				for (size_t m = 0; m < inverted_file_[i].size() - 1; m++)
				{
					int id_img1 = inverted_file_[i][m];
					for (size_t n = m + 1; n < inverted_file_[i].size(); n++)
					{
						int id_img2 = inverted_file_[i][n];
						adjacency_matrix[id_img1][id_img2]++;
						adjacency_matrix[id_img2][id_img1]++;
					}
				}
			}
		}

		float th_ratio = 0.02;
		for (size_t i = 0; i < num_imgs_; i++)
		{
			std::vector<std::pair<int, int>> matches_sort;
			for (size_t j = 0; j < num_imgs_; j++)
			{
				if (j != i)
				{
					matches_sort.push_back(std::pair<int, int>(j, adjacency_matrix[i][j]));
				}
				adjacency_matrix[i][j] = 0;
			}
			std::sort(matches_sort.begin(), matches_sort.end(), [](const std::pair<int, int> &lhs, const std::pair<int, int> &rhs) { return lhs.second > rhs.second; });

			int idx_th = MIN(100, matches_sort.size());
			for (size_t m = 0; m < idx_th; m++)
			{
				int id1 = i;
				int id2 = matches_sort[m].first;
				float min_pts = MIN(db_->words_id_[id1].size(), db_->words_id_[id2].size());
				if (matches_sort[m].second > th_ratio*min_pts)
				{
					adjacency_matrix[id1][id2] = matches_sort[m].second;
				}
			}
		}
	}

	bool MatchingGraphInvertedFile::GeoVerification(std::vector<cv::Point2f>& pt1, std::vector<cv::Point2f>& pt2, std::vector<int>& match_inliers)
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

	bool MatchingGraphInvertedFile::CheckMatchIndexFile()
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

	void MatchingGraphInvertedFile::WriteOutMatches(int idx1, int idx2, std::vector<std::pair<int, int>>& matches)
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

	void MatchingGraphInvertedFile::WriteOutMatchGraph()
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