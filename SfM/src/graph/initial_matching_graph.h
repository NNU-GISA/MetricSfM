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

#ifndef OBJECTSFM_INITIAL_MATCHING_GRAPH_H_
#define OBJECTSFM_INITIAL_MATCHING_GRAPH_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "database.h"
#include "basic_structs.h"

namespace objectsfm {

	class InitialMatchingGraph
	{
	public:
		InitialMatchingGraph();
		~InitialMatchingGraph();

		void AssociateDatabase(Database* db);
		
		void BuildInitialMatchGraph();

	private:
		// via priori information
		void match_graph_priori_ll();  

		void match_graph_priori_xy();

		void match_graph_priori_xy(std::vector<cv::Point2d> &pts);

		// 
		void match_graph_feature();

		void ReadinInitMatchGraph(int &id_last);

		void WriteOutInitMatchGraph(int id_last);

		bool CheckSimilarityGraph();

		void WriteOutSimilarityGraph(std::vector<std::vector<float>> &similarity_matrix);

		void ReadinSimilarityGraph(std::vector<std::vector<float>> &similarity_matrix);

	public:
		GraphOptions options_;
		Database * db_;
		int num_imgs_, id_last_init_match_;
		std::vector<std::vector<int>> match_graph_init;
	};


}
#endif // OBJECTSFM_GRAPH_VIA_COMBINED_H_