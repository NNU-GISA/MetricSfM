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

#include "feature_matching_essential.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <Eigen/Dense>
#include "orientation/essential_matrix_five_point.h"

namespace objectsfm {


	bool FeatureMatchingEssential::KNNMatching(std::vector<cv::KeyPoint>& kp1, cv::Mat & descriptors1, double f1, cv::Point2d pp1, 
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2, double f2, cv::Point2d pp2,
		std::vector<std::pair<int, int>> &matches)
	{
		float thRatio = 0.7;
		int th_reject = 20;

		if (kp1.size() < th_reject || kp2.size() < th_reject)
		{
			return false;
		}

		cv::flann::Index kdindex(descriptors2, cv::flann::KDTreeIndexParams(4)); // the paramter is the circular paramter for contructing kd tree.

		std::vector<int> matchesof1;
		matchesof1.resize(kp1.size(), -1);
		cv::Mat mindices, mdists;
		for (size_t i = 0; i < kp1.size(); i++)
		{
			kdindex.knnSearch(descriptors1.row(i), mindices, mdists, 2, cv::flann::SearchParams(32));
			float ratio = mdists.at<float>(0) / mdists.at<float>(1);
			if (ratio < thRatio)
			{
				matchesof1[i] = mindices.at<int>(0);
			}
		}

		std::vector<float> th_epipolar(2);
		th_epipolar[0] = 3.0;
		th_epipolar[1] = 1.0;

		int iter = 0;
		while (iter < 2)
		{
			std::vector<Eigen::Vector2d> pt1, pt2;
			std::vector<int> id_pt;
			for (size_t i = 0; i < matchesof1.size(); i++)
			{
				if (matchesof1[i] >= 0)
				{
					int id1 = i, id2 = matchesof1[i];
					pt1.push_back(Eigen::Vector2d((kp1[id1].pt.x - pp1.x) / f1, (kp1[id1].pt.y - pp1.y) / f1));
					pt2.push_back(Eigen::Vector2d((kp2[id2].pt.x - pp2.x) / f2, (kp2[id2].pt.y - pp2.y) / f2));
					id_pt.push_back(i);
				}
			}
			if (pt1.size() < th_reject)
			{
				return false;
			}

			// find essential matrix
			Eigen::Matrix3d E;
			EssentialMatrixFivePoints::FivePointEssentialMatrixRANSAC(pt1, pt2, E);

			// find outliers
			std::vector<bool> outliers(pt1.size());
			for (size_t i = 0; i < outliers.size(); i++)
			{
				const Eigen::Vector3d epiline_x = E * pt1[i].homogeneous();
				const double numerator_sqrt = pt2[i].homogeneous().dot(epiline_x);
				const Eigen::Vector4d denominator(pt2[i].homogeneous().dot(E.col(0)),
					pt2[i].homogeneous().dot(E.col(1)),
					epiline_x[0], epiline_x[1]);

				// Finally, return the complete Sampson distance.
				double e = numerator_sqrt * numerator_sqrt / denominator.squaredNorm();
				if (e > 0.001)
				{
					matchesof1[i] = -1;
				}
			}

			iter++;
		}

		//
		for (size_t i = 0; i < matchesof1.size(); i++)
		{
			if (matchesof1[i] >= 0)
			{
				matches.push_back(std::pair<int, int>(i, matchesof1[i]));
			}
		}
	}

}