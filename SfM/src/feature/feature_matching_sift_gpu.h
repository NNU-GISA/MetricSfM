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

#ifndef OBJECTSFM_FEATURE_MATCHING_CUDASIFT_H_
#define OBJECTSFM_FEATURE_MATCHING_CUDASIFT_H_

#include <opencv2/opencv.hpp>

#include "cudaSift/image.h"
#include "cudaSift/sift.h"
#include "cudaSift/utils.h"

namespace objectsfm {

class FeatureMatchingCudaSift
{
public:
	FeatureMatchingCudaSift() {};
	~FeatureMatchingCudaSift() {};

	static bool Run(std::vector<cv::KeyPoint>& kp1, cv::Mat & descriptors1, 
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2,
		std::vector<std::pair<int, int>> &matches);

	static bool Run(cudaSift::SiftData *siftdata1, cudaSift::SiftData *siftdata2, std::vector<std::pair<int, int>> &matches);

	static void DataConvert(std::vector<cv::KeyPoint>& kp, cv::Mat & descriptors, cudaSift::SiftData *data);

	static int ImproveHomography(cudaSift::SiftData &data, float *homography, int numLoops, float minScore, float maxAmbiguity, float thresh);

	static bool GeoVerification(std::vector<cv::Point2f> &pt1, std::vector<cv::Point2f> &pt2, std::vector<int> &match_inliers);
};

}
#endif // OBJECTSFM_FEATURE_MATCHING_CUDASIFT_H_