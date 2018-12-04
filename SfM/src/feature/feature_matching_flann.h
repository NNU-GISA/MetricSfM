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

#ifndef OBJECTSFM_FEATURE_MATCHING_FLANN_H_
#define OBJECTSFM_FEATURE_MATCHING_FLANN_H_

#include <opencv2/opencv.hpp>


namespace objectsfm {

class FeatureMatchingFlann
{
public:
	FeatureMatchingFlann() {};
	~FeatureMatchingFlann() {};

	static bool Run(cv::Mat *descriptors1,  std::vector<cv::Mat*> descriptors2, float th_ratio, float th_dis, 
		std::vector<std::vector<std::pair<int, int>>> &matches);
};

}
#endif // OBJECTSFM_FEATURE_MATCHING_CUDASIFT_H_