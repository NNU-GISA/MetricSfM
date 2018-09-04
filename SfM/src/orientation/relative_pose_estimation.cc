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

#include "orientation/relative_pose_estimation.h"

#include "orientation/essential_matrix_five_point.h"
#include "orientation/fundamental_matrix_eight_point.h"
#include "orientation/relative_pose_from_essential_matrix.h"
#include "orientation/relative_pose_from_fundamental_matrix.h"

#ifndef MIN_
#define MIN_(a,b) ( ((a)<(b)) ? (a):(b) )
#endif // !MIN

namespace objectsfm {
	bool RelativePoseEstimation::RelativePoseWithoutFocalLength(std::vector<Eigen::Vector2d>& pts_ref, std::vector<Eigen::Vector2d>& pts_cur, double & f_ref, double & f_cur, RTPoseRelative & pose_relative)
	{
		int num = pts_cur.size();

		// estiamte the fundamental matrix, pts_cur^t * F * pts_ref = 0
		Eigen::Matrix3d F;
		if (!FundamentalMatrixEightPoint::NormalizedEightPointFundamentalMatrixRANSAC(pts_ref, pts_cur, F))
		{
			return false;
		}
		
		if (0)
		{
			std::vector<cv::Point2f> pts1(num), pts2(num);
			for (size_t i = 0; i < num; i++)
			{
				pts1[i] = cv::Point2f(pts_ref[i](0), pts_ref[i](1));
				pts2[i] = cv::Point2f(pts_cur[i](0), pts_cur[i](1));
			}

			// homography matrix
			cv::Mat HMatrix = cv::findHomography(pts1, pts2);
			std::cout << HMatrix << std::endl;
			if (std::abs(HMatrix.at<double>(0, 0) - 0.995) < 0.01 &&
				std::abs(HMatrix.at<double>(1, 1) - 0.995) < 0.01 &&
				std::abs(HMatrix.at<double>(2, 2) - 0.995) < 0.01)
			{
				return false;
			}

			// fundamental matrix
			std::vector<uchar> ransac_status(num);
			cv::Mat FMatrix = cv::findFundamentalMat(pts1, pts2, ransac_status, cv::FM_RANSAC, 1.0);
			F << FMatrix.at<double>(0, 0), FMatrix.at<double>(0, 1), FMatrix.at<double>(0, 2),
				FMatrix.at<double>(1, 0), FMatrix.at<double>(1, 1), FMatrix.at<double>(1, 2),
				FMatrix.at<double>(2, 0), FMatrix.at<double>(2, 1), FMatrix.at<double>(2, 2);
		}
		

		// estimate pose from fundamental matrix
		double f1, f2;
		Eigen::Matrix3d R;
		Eigen::Vector3d t;
		if (!RelativePoseFromFundamentalMatrix::ReltivePoseFromFMatrix(F, pts_ref, pts_cur, f1, f2, R, t))
		{
			return false;
		}

		f_ref = f1;
		f_cur = f2;
		pose_relative.R = R;
		pose_relative.t = t;

		return true;
	}

	bool RelativePoseEstimation::RelativePoseWithSameFocalLength(std::vector<Eigen::Vector2d>& pts_ref, std::vector<Eigen::Vector2d>& pts_cur, double & f, RTPoseRelative & pose_relative)
	{
		return false;
	}

	bool RelativePoseEstimation::RelativePoseWithFocalLength(std::vector<Eigen::Vector2d>& pts_ref, std::vector<Eigen::Vector2d>& pts_cur, double f_ref, double f_cur, RTPoseRelative & pose_relative)
	{
		int num = pts_cur.size();

		std::vector<Eigen::Vector2d> pts1(num), pts2(num);
		for (size_t i = 0; i < num; i++)
		{
			pts1[i] = pts_ref[i] / f_ref;
			pts2[i] = pts_cur[i] / f_cur;
		}

		// estiamte the essential matrix, pts2^t * F * pts1 = 0
		Eigen::Matrix3d E;
		if (!EssentialMatrixFivePoints::FivePointEssentialMatrixRANSAC(pts1, pts2, E))
		{
			return false;
		}

		// 
		Eigen::Matrix3d R;
		Eigen::Vector3d t;
		if (!RelativePoseFromEssentialMatrix::ReltivePoseFromEMatrix(E, pts1, pts2, R, t))
		{
			return false;
		}

		pose_relative.R = R;
		pose_relative.t = t;

		return true;
	}
}  // namespace objectsfm

