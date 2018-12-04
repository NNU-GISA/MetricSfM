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

#ifndef OBJECTSFM_FINE_MATCHING_GRAPH_H_
#define OBJECTSFM_FINE_MATCHING_GRAPH_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "database.h"
#include "basic_structs.h"

namespace objectsfm {

	class FineMatchingGraph
	{
	public:
		FineMatchingGraph();
		~FineMatchingGraph();

		void AssociateDatabase(Database* db);
		
		void BuildMatchGraph(std::vector<std::vector<int>> &match_graph_init);

		void GraphMatching(std::vector<std::vector<int>> &match_graph_init);

		void GraphVerification();

	public:

		bool CheckMatchIndexFile();

		// initial matching
		std::vector<int> CheckMissingMatchingFile();

		void RecoverMatchingGraph(std::vector<int> &existing_matches);

		void WriteOutMatches(int idx1, int idx2, std::vector<std::pair<int, int>> &matches);

		void WriteOutMatchGraph();

		// 
		std::vector<int> CheckMissingMatchingFile2();

		void RecoverMatchingGraph2(std::vector<int> &existing_matches);

		void WriteOutMatches2(int idx1, int idx2, std::vector<std::pair<int, int>> &matches);

		void WriteOutMatchGraph2();

		void ReadinMatch(int idx1, std::vector<int> &idx2, std::vector<std::vector<std::pair<int, int>>>  &matches);

	public:
		GraphOptions options_;
		Database * db_;
		int num_imgs_;
		std::vector<std::vector<int>> match_graph;
		std::vector<std::vector<int>> match_graph2;
	};
}
#endif // OBJECTSFM_GRAPH_VIA_COMBINED_H_