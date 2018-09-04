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

#ifndef OBJECTSFM_GRAPH_WORD_EXPENDING_H_
#define OBJECTSFM_GRAPH_WORD_vMATCHING_H_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "database.h"
#include "basic_structs.h"

namespace objectsfm {

	struct KNNWords 
	{
		int idx[3];
		float dix[3];
	};

	class MatchingGraphWordExpending
	{
	public:
		MatchingGraphWordExpending();
		~MatchingGraphWordExpending();

		void AssociateDatabase(Database* db);
		
		void BuildMatchGraph();

	public:
		void BuildVocabularyTree();

		bool CheckWordFile();

		// words
		void GenerateImageWordFeatureViaQuerying();

		void GenerateImageWordFeatureViaExpending();

		void WriteOutWords(int idx, int num_pts, int N, int* ids_ptr, float* dis_ptr);

		void ReadinWords(int idx, int &num_pts, int &N, int** ids_ptr, float** dis_ptr);

		void WriteOutWordPtMap(int idx, int num, int* word_pt_ptr);

		void ReadinWordPtMap(int idx, int &num, int** word_pt_ptr);

		// similarity graph
		bool CheckSimilarityGraph();

		void BuildSimilarityGraph();

		void WriteOutSimilarityGraph();

		void ReadinSimilarityGraph();

		// matching
		void DoMatching();

		bool GeoVerification(std::vector<cv::Point2f>& pt1, std::vector<cv::Point2f>& pt2, std::vector<int> &match_inliers);

		bool CheckMatchIndexFile();

		void WriteOutMatches(int idx1, int idx2, std::vector<std::pair<int, int>> &matches);

		void WriteOutMatchGraph();

	public:
		GraphOptions options_;
		Database * db_;
		int num_imgs_, num_words_;
		float* similarity_graph_;
		int* match_graph_; 
	};
}
#endif // OBJECTSFM_GRAPH_WORD_EXPENDING_H_