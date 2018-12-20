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

#include "feature_extractor_opencv.h"

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace objectsfm {


void FeatureExtractorOpenCV::Run(cv::Mat &image, std::string method, ListKeyPoint* keypoints, cv::Mat* descriptors)
{
	std::vector<cv::KeyPoint> keypoints_cur;
	cv::Mat descriptors_cur;
	cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create();
	sift->detect(image, keypoints_cur);
	sift->compute(image, keypoints_cur, descriptors_cur);

	keypoints->pts = keypoints_cur;
	descriptors = &descriptors_cur;
}

}  // namespace objectsfm
