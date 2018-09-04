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

#include "similarity_graph_word_number.h"

#include "utils/basic_funcs.h"

namespace objectsfm {

	SimilarityGraphWordNumber::SimilarityGraphWordNumber()
	{
	}

	SimilarityGraphWordNumber::~SimilarityGraphWordNumber()
	{
	}

	void SimilarityGraphWordNumber::AssociateDatabase(Database * db)
	{
		db_ = db;
	}

	void SimilarityGraphWordNumber::BuildSimilarityGraph(std::vector<std::vector<float>> &similarity_matrix)
	{
		num_imgs_ = db_->num_imgs_;

		// step1: build words for each image first
		db_->BuildWords();
		for (size_t i = 0; i < num_imgs_; i++)
		{
			db_->ReadinWordsForImage(i);
		}
		num_words_ = db_->max_words_id_;
		std::cout << "number of words: " << num_words_ << std::endl;

		// step2: generate inverted file
		std::vector<std::vector<int>> inverted_file;
		GenerateInvertedFile(inverted_file);

		// step3: generate adjacent matrix
		similarity_matrix.resize(num_imgs_);
		for (size_t i = 0; i < num_imgs_; i++)
		{
			similarity_matrix[i].resize(num_imgs_, 0);
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
						similarity_matrix[id_img1][id_img2]++;
						similarity_matrix[id_img2][id_img1]++;
					}
				}
			}
		}

		for (size_t i = 0; i < num_imgs_; i++)
		{
			db_->ReleaseWordsForImage(i);
		}
	}


	void SimilarityGraphWordNumber::GenerateInvertedFile(std::vector<std::vector<int>> &inverted_file)
	{
		inverted_file.resize(num_words_);

		// condition 1
		for (size_t i = 0; i < num_imgs_; i++)
		{
			if (db_->words_id_[i].size())
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
}