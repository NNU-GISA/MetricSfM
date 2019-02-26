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

#ifndef MAX_
#define MAX_(a,b) ( ((a)>(b)) ? (a):(b) )
#endif // !MAX

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN

#include "graph.h"

#include <fstream>
#include <Eigen/Core>
#include <Eigen/LU>

#include <omp.h>

#include "utils/basic_funcs.h"

#include "graph/matching_graph_via_combined.h"

namespace objectsfm {

	Graph::Graph()
	{
	}

	Graph::~Graph()
	{
	}

	void Graph::AssociateDatabase(Database * db)
	{
		db_ = db;
	}

	bool Graph::BuildGraph()
	{
		MatchingGraphViaCombined grapher;
		grapher.options_.use_gpu = options_.use_gpu;
		grapher.AssociateDatabase(db_);
		grapher.BuildMatchGraph(options_.all_match);
		return true;
	}

	void Graph::ReadinMatchingGraph()
	{
		std::string path = db_->output_fold_ + "//" + "graph_matching";
		std::ifstream ifs(path, std::ios::binary);
		if (!ifs.is_open())
		{
			return;
		}
		match_graph_ = new int[db_->num_imgs_*db_->num_imgs_];
		ifs.read((char*)match_graph_, db_->num_imgs_*db_->num_imgs_ * sizeof(int));
		ifs.close();
	}

	void Graph::ReleaseMatchingGraph()
	{
		delete[] match_graph_;
	}

	void Graph::QueryMatch(int idx, std::vector<int>& image_ids, std::vector<std::vector<std::pair<int, int>>>& match_pts)
	{
		// read in data
		std::string match_file = db_->output_fold_ + "//" + std::to_string(idx) + "_match";
		std::ifstream ifs;
		ifs.open(match_file, std::ios::in | std::ios::binary);

		int id, num_match;
		while (ifs.read((char*)(&id), sizeof(int)))
		{
			ifs.read((char*)(&num_match), sizeof(int));

			int *temp = new int[num_match * 2];
			ifs.read((char*)(temp), num_match * 2 * sizeof(int));

			// matches
			image_ids.push_back(id);
			std::vector<std::pair<int, int>> matches_temp(num_match);
			for (size_t i = 0; i < num_match; i++)
			{
				matches_temp[i].first = temp[2 * i + 0];
				matches_temp[i].second = temp[2 * i + 1];
			}
			match_pts.push_back(matches_temp);

			delete[] temp;
		}
		ifs.close();
	}

	// matches: (id1, id2)
	void Graph::QueryMatch(int idx1, int idx2, std::vector<std::pair<int, int>>& matchs)
	{
		std::vector<int> cameras;
		std::vector<std::vector<std::pair<int, int>>> match_all;
		QueryMatch(idx1, cameras, match_all);

		for (size_t i = 0; i < cameras.size(); i++)
		{
			if (cameras[i] == idx2)
			{
				matchs = match_all[i];
				break;
			}
		}
	}

	void Graph::QueryBestMatch(int idx1, int & idx2, std::vector<std::pair<int, int>>& matchs)
	{
		std::vector<int> image_ids;
		std::vector<std::vector<std::pair<int, int>>> match_pts;
		QueryMatch(idx1, image_ids, match_pts);

		int index_best = 0;
		int num_best = 0;
		for (size_t i = 0; i < image_ids.size(); i++)
		{
			if (match_pts[i].size() > num_best)
			{
				num_best = match_pts[i].size();
				index_best = i;
			}
		}

		idx2 = image_ids[index_best];
		matchs = match_pts[index_best];
	}

}  // namespace objectsfm
