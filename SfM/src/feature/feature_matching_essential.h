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

#ifndef OBJECTSFM_FEATURE_MATCHING_ESSENTIAL_H_
#define OBJECTSFM_FEATURE_MATCHING_ESSENTIAL_H_

#include <opencv2/opencv.hpp>

namespace objectsfm {

class FeatureMatchingEssential
{
public:
	FeatureMatchingEssential() {};
	~FeatureMatchingEssential() {};

	static bool KNNMatching(std::vector<cv::KeyPoint>& kp1, cv::Mat & descriptors1, double f1, cv::Point2d pp1,
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2, double f2, cv::Point2d pp2,
		std::vector<std::pair<int, int>> &matches);
};

}
#endif //OBJECTSFM_FEATURE_MATCHING_ESSENTIAL_H_