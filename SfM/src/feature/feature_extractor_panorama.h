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

#ifndef OBJECTSFM_FEATURE_EXTRACTOR_PANORAMA_H_
#define OBJECTSFM_FEATURE_EXTRACTOR_PANORAMA_H_

#include <string>
#include <opencv2/opencv.hpp>
#include "basic_structs.h"

namespace objectsfm {

class FeatureExtractorPanorama
{
public:
	FeatureExtractorPanorama() {};
	~FeatureExtractorPanorama() {};

	static void Run(cv::Mat &image, std::string method, ListKeyPoint* keypoints, cv::Mat* descriptors, std::string path);

	static void Sphere2Cyclinder(cv::Mat &img_sphere, float fov_x, float fov_y, int f, cv::Mat &img_cyclinder);

	static void BilinearInterpolation(cv::Mat &img_sphere, float x, float y, cv::Scalar &pixel);
};

}  // namespace objectsfm

#endif  // OBJECTSFM_FEATURE_EXTRACTOR_OPENCV_H_
