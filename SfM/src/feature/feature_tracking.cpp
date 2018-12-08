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

#include "feature_matching.h"
#include "feature_tracking.h"
#include "utils/basic_funcs.h"

namespace objectsfm {

	void FeatureTracking::TrackNFrames(cv::VideoCapture * video_cap, cv::Size zoom_size, int N,
		cv::Mat &frame_init, std::vector<cv::Point2f> &pts_init,
		cv::Mat &frame_tracked, std::vector<cv::Point2f>& pts_tracked)
	{
		cv::Mat frame_prev = frame_init.clone();
		std::vector<cv::Point2f> pts_prev = pts_init;
		std::vector<uchar> status_track(pts_init.size(), 1);

		// step2: feature matching via tracking
		int num_frames = 0;
		cv::Size winSize(21, 21);

		std::vector<cv::Point2f> pts_cur;
		while (num_frames < N)
		{
			*video_cap >> frame_tracked;
			cv::resize(frame_tracked, frame_tracked, zoom_size);
			cv::cvtColor(frame_tracked, frame_tracked, CV_RGB2GRAY);

			std::vector<uchar> status_cur;
			std::vector<float> err_cur;
			cv::calcOpticalFlowPyrLK(frame_prev, frame_tracked, pts_prev, pts_cur, status_cur, err_cur, winSize);
			math::vector_dot(status_track, status_cur);

			swap(pts_prev, pts_cur);
			frame_prev = frame_tracked.clone();
		}

		// remove outliers matching via Fundamental matrix
		std::vector<cv::Point2f> pt1, pt2;
		std::vector<int> idx;
		for (size_t i = 0; i < status_track.size(); i++)
		{
			if (status_track[i])
			{
				pt1.push_back(pts_init[i]);
				pt2.push_back(pts_cur[i]);
				idx.push_back(i);
			}
		}

		std::vector<uchar> status_match(pt1.size());
		cv::Mat FMatrix = cv::findFundamentalMat(pt1, pt2, status_match, cv::FM_RANSAC, 2.0);

		pts_tracked.resize(pts_init.size());
		for (size_t i = 0; i < pts_tracked.size(); i++)
		{
			pts_tracked[i].x = -1;
			pts_tracked[i].y = -1;
		}
		for (size_t i = 0; i < status_match.size(); i++)
		{
			if (status_match[i])
			{
				int id = idx[i];
				pts_tracked[id] = pt2[i];
			}
		}
	}

	void FeatureTracking::TrackNFrames(cv::VideoCapture * video_cap, cv::Size zoom_size, double ratio,
		cv::Mat &frame_init, std::vector<cv::Point2f> &pts_init,
		cv::Mat &frame_tracked, std::vector<cv::Point2f>& pts_tracked)
	{
		cv::Mat frame_prev = frame_init.clone();
		std::vector<cv::Point2f> pts_prev = pts_init;
		std::vector<uchar> status_track(pts_init.size(), 1);

		// step2: feature matching via tracking
		double ratio_track_matched = 1.0;
		cv::Size winSize(21, 21);

		std::vector<cv::Point2f> pts_cur;
		while (ratio_track_matched > ratio)
		{
			*video_cap >> frame_tracked;
			cv::resize(frame_tracked, frame_tracked, zoom_size);
			cv::cvtColor(frame_tracked, frame_tracked, CV_RGB2GRAY);

			std::vector<uchar> status_cur;
			std::vector<float> err_cur;
			cv::calcOpticalFlowPyrLK(frame_prev, frame_tracked, pts_prev, pts_cur, status_cur, err_cur, winSize);
			math::vector_dot(status_track, status_cur);

			// 
			double inliers = 0;
			for (size_t i = 0; i < status_track.size(); i++)
			{
				if (status_track[i])
				{
					inliers++;
				}
			}
			ratio_track_matched = inliers / status_track.size();

			swap(pts_prev, pts_cur);
			frame_prev = frame_tracked.clone();
		}

		// remove outliers matching via Fundamental matrix
		std::vector<cv::Point2f> pt1, pt2;
		std::vector<int> idx;
		for (size_t i = 0; i < status_track.size(); i++)
		{
			if (status_track[i])
			{
				pt1.push_back(pts_init[i]);
				pt2.push_back(pts_cur[i]);
				idx.push_back(i);
			}
		}

		std::vector<uchar> status_match(pt1.size());
		cv::Mat FMatrix = cv::findFundamentalMat(pt1, pt2, status_match, cv::FM_RANSAC, 2.0);
		
		pts_tracked.resize(pts_init.size());
		for (size_t i = 0; i < pts_tracked.size(); i++)
		{
			pts_tracked[i].x = -1;
			pts_tracked[i].y = -1;
		}
		for (size_t i = 0; i < status_match.size(); i++)
		{
			if (status_match[i])
			{
				int id = idx[i];
				pts_tracked[id] = pt2[i];
			}
		}
	}
	void FeatureTracking::Track1Frames(cv::VideoCapture * video_cap, cv::Size zoom_size,
		cv::Mat & frame_prev, std::vector<cv::Point2f>& pts_prev,
		cv::Mat & frame_tracked, std::vector<cv::Point2f>& pts_tracked, int &num_tracked)
	{
		*video_cap >> frame_tracked;
		cv::resize(frame_tracked, frame_tracked, zoom_size);
		cv::cvtColor(frame_tracked, frame_tracked, CV_RGB2GRAY);

		std::vector<uchar> status_cur;
		std::vector<float> err_cur;
		cv::calcOpticalFlowPyrLK(frame_prev, frame_tracked, pts_prev, pts_tracked, status_cur, err_cur);

		// remove outliers matching via Fundamental matrix
		std::vector<cv::Point2f> pt1, pt2;
		std::vector<int> idx;
		for (size_t i = 0; i < status_cur.size(); i++)
		{
			if (status_cur[i])
			{
				pt1.push_back(pts_prev[i]);
				pt2.push_back(pts_tracked[i]);
				idx.push_back(i);
			}
		}

		std::vector<uchar> status_match(pt1.size());
		cv::Mat FMatrix = cv::findFundamentalMat(pt1, pt2, status_match, cv::FM_RANSAC, 2.0);

		pts_tracked.resize(pts_prev.size());
		for (size_t i = 0; i < pts_tracked.size(); i++)
		{
			pts_tracked[i].x = -100;
			pts_tracked[i].y = -100;
		}
		num_tracked = 0;
		for (size_t i = 0; i < status_match.size(); i++)
		{
			if (status_match[i])
			{
				num_tracked++;
				int id = idx[i];
				pts_tracked[id] = pt2[i];
			}
		}
	}
}