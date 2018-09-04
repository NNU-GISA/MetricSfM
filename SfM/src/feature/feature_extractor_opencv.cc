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
#include <opencv2/nonfree/nonfree.hpp>   
#include <opencv2/nonfree/features2d.hpp>  

namespace objectsfm {


void FeatureExtractorOpenCV::Run(cv::Mat &image, std::string method, ListKeyPoint* keypoints, cv::Mat* descriptors)
{
	if (method != "SIFT" && method != "sift" && method != "SURF" && method != "surf")
	{
		method = "SIFT";
	}

	cv::initModule_nonfree(); //if use SIFT or SURF
	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(method);
	cv::Ptr<cv::DescriptorExtractor> descriptor = cv::DescriptorExtractor::create(method);

	std::vector<cv::KeyPoint> keypoints_cur;
	cv::Mat descriptors_cur;
	detector->detect(image, keypoints_cur);
	descriptor->compute(image, keypoints_cur, descriptors_cur);

	keypoints->pts = keypoints_cur;
	descriptors = &descriptors_cur;
}

}  // namespace objectsfm
