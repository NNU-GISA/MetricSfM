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

#include "geo_verification.h"

#include "utils/basic_funcs.h"

namespace objectsfm
{
	GeoVerification::GeoVerification()
	{
	}
	GeoVerification::~GeoVerification()
	{
	}

	bool GeoVerification::GeoVerificationFundamental(std::vector<cv::Point2f>& pt1, std::vector<cv::Point2f>& pt2, std::vector<int>& match_inliers)
	{
		if (pt1.size() < 30) {
			return false;
		}

		//cv::Mat HMatrix1 = cv::findHomography(pt1, pt2);
		//if (std::abs(HMatrix1.at<double>(0, 0) - 0.995) < 0.01 &&
		//	std::abs(HMatrix1.at<double>(1, 1) - 0.995) < 0.01 &&
		//	std::abs(HMatrix1.at<double>(2, 2) - 0.995) < 0.01) {
		//	return false;
		//}

		float th_epipolar1 = 2.0;
		std::vector<uchar> ransac_status1(pt1.size());
		cv::findFundamentalMat(pt1, pt2, ransac_status1, cv::FM_RANSAC, th_epipolar1);
		for (size_t i = 0; i < ransac_status1.size(); i++) {
			if (ransac_status1[i]) {
				match_inliers.push_back(i);
			}
		}

		if (match_inliers.size() < 30) {
			return false;
		}

		return true;
	}

	bool GeoVerification::GeoVerificationLocalFlow(std::vector<cv::Point2f>& pt1, std::vector<cv::Point2f>& pt2, std::vector<int>& match_inliers)
	{
		int n = pt1.size();

		// get the bounding box
		int xmin = 100000, xmax = 0;
		int ymin = 100000, ymax = 0;
		for (size_t i = 0; i < pt1.size(); i++)
		{
			if (pt1[i].x < xmin) xmin = pt1[i].x;
			if (pt1[i].x > xmax) xmax = pt1[i].x;

			if (pt1[i].y < ymin) ymin = pt1[i].y;
			if (pt1[i].y > ymax) ymax = pt1[i].y;
		}

		// grid
		int sgrid = 4;
		int size = sgrid * sgrid;
		float stepx = float(xmax - xmin + 1.0) / sgrid;
		float stepy = float(ymax - ymin + 1.0) / sgrid;
		std::vector<std::vector<int>> grid(size);
		for (size_t i = 0; i < n; i++) {
			int gx = (pt1[i].x - xmin) / stepx;
			int gy = (pt1[i].y - ymin) / stepy;
			int loc = gy * sgrid + gx;
			grid[loc].push_back(i);
		}

		// averaging
		cv::Mat gridx(sgrid, sgrid, CV_32FC1, cv::Scalar(0));
		cv::Mat gridy(sgrid, sgrid, CV_32FC1, cv::Scalar(0));
		for (size_t i = 0; i < sgrid; i++) {
			for (size_t j = 0; j < sgrid; j++)
			{
				float avg_x = 0.0, avg_y = 0.0;
				int inlier_x = 0, inlier_y = 0;
				int loc = i * sgrid + j;
				//if (xgrid[loc].size() != 0) {
				//	math::vector_avg_denoise(xgrid[loc], inlier_x, avg_x);
				//	math::vector_avg_denoise(ygrid[loc], inlier_y, avg_y);
				//}

				gridx.at<float>(i, j) = avg_x;
				gridy.at<float>(i, j) = avg_y;
			}
		}

		// gausssian filtering
		//cv::GaussianBlur(gridx, gridx, cv::Size(5, 5), 0, 0);
		//cv::GaussianBlur(gridy, gridy, cv::Size(5, 5), 0, 0);

		// find out matching inliers
		float* ptrx = (float*)gridx.data;
		float* ptry = (float*)gridy.data;
		for (size_t i = 0; i < n; i++) {
			int gx = (pt1[i].x - xmin) / stepx;
			int gy = (pt1[i].y - ymin) / stepy;
			int loc = gy * sgrid + gx;
			
			float dx = pt2[i].x - pt1[i].x;
			float dy = pt2[i].y - pt1[i].y;
			float fx = ptrx[loc];
			float fy = ptry[loc];
			if (abs(fx - dx) < stepx && abs(fy - dy)<stepy) {
				match_inliers.push_back(i);
			}
		}

		if (match_inliers.size() < 30) {
			return false;
		}
		return true;
	}

	bool GeoVerification::GeoVerificationPatchFundamental(std::vector<cv::Point2f>& pt1, std::vector<cv::Point2f>& pt2, std::vector<int>& match_inliers)
	{
		if (pt1.size() < 20) {
			return false;
		}

		cv::Mat HMatrix1 = cv::findHomography(pt1, pt2);
		if (std::abs(HMatrix1.at<double>(0, 0) - 0.998) < 0.01 &&
			std::abs(HMatrix1.at<double>(1, 1) - 0.998) < 0.01 &&
			std::abs(HMatrix1.at<double>(2, 2) - 0.998) < 0.01) {
			return false;
		}

		int n = pt1.size();

		// get the bounding box
		int xmin = 100000, xmax = 0;
		int ymin = 100000, ymax = 0;
		for (size_t i = 0; i < pt1.size(); i++)
		{
			if (pt1[i].x < xmin) xmin = pt1[i].x;
			if (pt1[i].x > xmax) xmax = pt1[i].x;

			if (pt1[i].y < ymin) ymin = pt1[i].y;
			if (pt1[i].y > ymax) ymax = pt1[i].y;
		}

		// grid
		int sgrid = 2;
		int size = sgrid * sgrid;
		float stepx = float(xmax - xmin + 1.0) / sgrid;
		float stepy = float(ymax - ymin + 1.0) / sgrid;
		std::vector<std::vector<int>> grid(size);
		for (size_t i = 0; i < n; i++) {
			int gx = (pt1[i].x - xmin) / stepx;
			int gy = (pt1[i].y - ymin) / stepy;
			int loc = gy * sgrid + gx;
			grid[loc].push_back(i);
		}

		// F for patch
		for (size_t i = 0; i < size; i++)
		{
			std::vector<cv::Point2f> pt11, pt22;
			for (size_t j = 0; j < grid[i].size(); j++) {
				pt11.push_back(pt1[grid[i][j]]);
				pt22.push_back(pt2[grid[i][j]]);
			}
			if (pt11.size() < 20) {
				continue;
			}

			float th_epipolar = 3.0;
			std::vector<uchar> ransac_status(pt11.size());
			cv::findFundamentalMat(pt11, pt22, ransac_status, cv::FM_RANSAC, th_epipolar);
			for (size_t j = 0; j < ransac_status.size(); j++) {
				if (ransac_status[j]) {
					match_inliers.push_back(grid[i][j]);
				}
			}
		}

		if (match_inliers.size() < 30) {
			return false;
		}
		return true;
	}
};
