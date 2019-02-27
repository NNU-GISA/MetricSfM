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

#ifndef OBJECTSFM_SIM_GRAPH_H_
#define OBJECTSFM_SIM_GRAPH_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "database.h"
#include "basic_structs.h"

namespace objectsfm {

	class SimilarityGraph
	{
	public:
		SimilarityGraph();
		~SimilarityGraph();

		void AssociateDatabase(Database* db);
		
		void BuildSimilarityGraph(int method, std::vector<std::vector<float>> &similarity_matrix);

		// method 1: invreted file
		void SimilarityGraphInvFile(std::vector<std::vector<float>> &similarity_matrix);

		void GenerateInvertedFile(std::vector<std::vector<int>> &inverted_file);

		// method 2: bow vector distance
		void SimilarityGraphBowDistance(std::vector<std::vector<float>> &similarity_matrix);

	public:
		Database * db_;
		int num_imgs_, num_words_;
	};
}
#endif // OBJECTSFM_GRAPH_WORD_NUMBER_H_