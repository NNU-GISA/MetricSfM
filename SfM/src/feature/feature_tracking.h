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

#ifndef OBJECTSFM_FEATURE_TRACKING_H_
#define OBJECTSFM_FEATURE_TRACKING_H_

#include <opencv2/opencv.hpp>

namespace objectsfm {

class FeatureTracking
{
public:
	FeatureTracking() {};
	~FeatureTracking() {};

	static void TrackNFrames(cv::VideoCapture *video_cap, cv::Size zoom_size, int N,
		cv::Mat &frame_init, std::vector<cv::Point2f> &pts_init, 
		cv::Mat &frame_tracked, std::vector<cv::Point2f> &pts_tracked);

	static void TrackNFrames(cv::VideoCapture *video_cap, cv::Size zoom_size, double ratio,
		cv::Mat &frame_init, std::vector<cv::Point2f> &pts_init,
		cv::Mat &frame_tracked, std::vector<cv::Point2f> &pts_tracked);

	static void Track1Frames(cv::VideoCapture *video_cap, cv::Size zoom_size,
		cv::Mat &frame_prev, std::vector<cv::Point2f> &pts_prev,
		cv::Mat &frame_tracked, std::vector<cv::Point2f> &pts_tracked, int &num_tracked);
};

}
#endif //OBJECTSFM_IMAGE_VLSIFT_MATCHER_H_