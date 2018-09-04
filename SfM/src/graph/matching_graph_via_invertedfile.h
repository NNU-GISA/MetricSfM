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

#ifndef OBJECTSFM_GRAPH_VIA_INVERTED_FILE_H_
#define OBJECTSFM_GRAPH_VIA_INVERTED_FILE_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "database.h"
#include "basic_structs.h"

namespace objectsfm {

	class MatchingGraphInvertedFile
	{
	public:
		MatchingGraphInvertedFile();
		~MatchingGraphInvertedFile();

		void AssociateDatabase(Database* db);
		
		void BuildMatchGraph();

	public:
		void GenerateInvertedFile();

		void GenerateAdjacentMatrix(std::vector<std::vector<int>> &adjacency_matrix);

		bool GeoVerification(std::vector<cv::Point2f>& pt1, std::vector<cv::Point2f>& pt2, std::vector<int> &match_inliers);

		bool CheckMatchIndexFile();

		void WriteOutMatches(int idx1, int idx2, std::vector<std::pair<int, int>> &matches);

		void WriteOutMatchGraph();

	public:
		GraphOptions options_;
		Database * db_;
		int num_imgs_, num_words_;
		std::vector<std::vector<int>> inverted_file_;
		int* match_graph_; 
	};
}
#endif // OBJECTSFM_GRAPH_INVERTED_FILE_H_