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

#ifndef OBJECTSFM_GRAPH_H_
#define OBJECTSFM_GRAPH_H_

#include <vector>

#include <opencv2\opencv.hpp>   
#include <opencv2\highgui\highgui.hpp>

#include "database.h"
#include "basic_structs.h"

namespace objectsfm {


class Graph
{
public:
	Graph();
	~Graph();

	GraphOptions options_;

	void AssociateDatabase(Database* db);

	bool BuildGraph();

	void ReadinMatchingGraph();

	void ReleaseMatchingGraph();

	void QueryMatch(int idx, std::vector<int> &image_ids, std::vector<std::vector<std::pair<int,int>>> &matchs);
			  
	void QueryMatch(int idx1, int idx2, std::vector<std::pair<int,int>> &matchs);

	void QueryBestMatch(int idx1, int &idx2, std::vector<std::pair<int, int>> &matchs);

public:
	Database* db_;
	int* match_graph_;
};

}  // namespace objectsfm

#endif  // OBJECTSFM_OBJ_GRAPH_H_
