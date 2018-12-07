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
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "feature_matching_flann.h"

#include "flann/flann.h"

namespace objectsfm {

	bool FeatureMatchingFlann::Run(cv::Mat * descriptors1, std::vector<cv::Mat*> descriptors2, float th_ratio, float th_dis, 
		std::vector<std::vector<std::pair<int, int>>>& matches)
	{
		// generate kd-tree for idx1
		struct FLANNParameters p;
		p = DEFAULT_FLANN_PARAMETERS;
		p.algorithm = FLANN_INDEX_KDTREE;
		p.trees = 8;
		p.log_level = FLANN_LOG_INFO;
		p.checks = 64;
		float speedup;

		float* data_idx1 = (float*)descriptors1->data;
		flann_index_t kdtree_idx1 = flann_build_index(data_idx1, descriptors1->rows, descriptors1->cols, &speedup, &p);

		// knn querying
		std::vector<int*> knn_id(descriptors2.size());
		std::vector<float*> knn_dis(descriptors2.size());
		for (int j = 0; j < descriptors2.size(); j++)
		{
			std::cout << "  -------" << j << " " << descriptors2[j]->rows  << std::endl;

			int count = descriptors2[j]->rows;
			knn_id[j] = new int[count * 2];
			knn_dis[j] = new float[count * 2];
			flann_find_nearest_neighbors_index(kdtree_idx1, (float*)descriptors2[j]->data, count, knn_id[j], knn_dis[j], 2, &p);
		}
		flann_free_index(kdtree_idx1, &p);

		// find inital matches via best-second ratio
		matches.resize(descriptors2.size());
		for (int j = 0; j < descriptors2.size(); j++)
		{
			int num2 = descriptors2[j]->rows;

			// the inital matches
			std::vector<cv::Point2f> pt1, pt2;
			int* ptr_id = knn_id[j];
			float* ptr_dis = knn_dis[j];
			for (size_t m = 0; m < num2; m++)
			{
				float ratio = ptr_dis[0] / ptr_dis[1];
				if (ratio < th_ratio && ptr_dis[0] < th_dis) {
					matches[j].push_back(std::pair<int, int>(ptr_id[0], m));
				}
				ptr_id += 2;
				ptr_dis += 2;
			}
			delete[] knn_id[j];
			delete[] knn_dis[j];
		}

		return true;
	}
}