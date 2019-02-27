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

#ifndef OBJECTSFM_FEATURE_MATCHING_H_
#define OBJECTSFM_FEATURE_MATCHING_H_

#include <opencv2/opencv.hpp>

#include "basic_structs.h"
#include "flann/flann.h"

namespace objectsfm {

class FeatureMatching
{
public:
	FeatureMatching() {};
	~FeatureMatching() {};

	static bool KNNMatching(std::vector<cv::KeyPoint>& kp1, cv::Mat & descriptors1, 
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2,
		std::vector<std::pair<int, int>> &matches);

	static bool KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, cv::Mat & descriptors1, 
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2,
		std::vector<std::pair<int,int>> &matches);

	static bool KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, cv::Mat & descriptors1, 
		std::vector<cv::KeyPoint>& kp2, cv::flann::Index* kdindex2,
		std::vector<std::pair<int, int>> &matches);

	static bool KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, cv::flann::Index* kdindex1,
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2,
		std::vector<std::pair<int, int>> &matches);

	static bool KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, my_kd_tree_t *kd_tree1,
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2,
		std::vector<std::pair<int, int>> &matches);

	static bool KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, flann_index_t *kd_tree1, FLANNParameters &p,
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2,
		std::vector<std::pair<int, int>> &matches);

	static bool KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
		int* id, float* dis, std::vector<std::pair<int, int>> &matches);

	// slam
	static bool KNNMatchingWithGeoVerify(std::vector<cv::Point2f>& kp1, cv::Mat & descriptors1,
		std::vector<cv::Point2f>& kp2, cv::flann::Index* kdindex2,
		std::vector<std::pair<int, int>> &matches);

	static void GenerateKDIndex(cv::Mat & descriptors, cv::flann::Index** kdindex);
};

}
#endif //OBJECTSFM_IMAGE_VLSIFT_MATCHER_H_