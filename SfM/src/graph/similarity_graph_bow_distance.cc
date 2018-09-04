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

#include "similarity_graph_bow_distance.h"

#include "utils/basic_funcs.h"

namespace objectsfm {

	SimilarityGraphBowDistance::SimilarityGraphBowDistance()
	{
	}

	SimilarityGraphBowDistance::~SimilarityGraphBowDistance()
	{
	}

	void SimilarityGraphBowDistance::AssociateDatabase(Database * db)
	{
		db_ = db;
	}

	void SimilarityGraphBowDistance::BuildSimilarityGraph(std::vector<std::vector<float>> &similarity_matrix)
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

		// step2: generate sim graph
		similarity_matrix.resize(num_imgs_);
		for (size_t i = 0; i < num_imgs_; i++)
		{
			std::cout << i << std::endl;
			similarity_matrix[i].resize(num_imgs_);
#pragma omp parallel for
			for (int j = 0; j < num_imgs_; j++)
			{
				float sim = math::simOfBows(db_->words_vector_[i], db_->words_vector_[j]);
				similarity_matrix[i][j] = sim;
			}
		}

		for (size_t i = 0; i < num_imgs_; i++)
		{
			db_->ReleaseWordsForImage(i);
		}
	}
}