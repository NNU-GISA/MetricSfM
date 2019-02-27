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

#include <omp.h>

namespace objectsfm {


	bool FeatureMatching::KNNMatching(std::vector<cv::KeyPoint>& kp1, cv::Mat & descriptors1, std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2,
		std::vector<std::pair<int, int>> &matches)
	{
		float thRatio = 0.5;
		int th_reject = 20;

		if (kp1.size() < th_reject || kp2.size() < th_reject)
		{
			return false;
		}

		cv::flann::Index kdindex(descriptors2, cv::flann::KDTreeIndexParams(4)); // the paramter is the circular paramter for contructing kd tree.

		int num_matches = 0;

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
				num_matches++;
			}
		}

		//
		matches.resize(num_matches);
		int count = 0;
		for (size_t i = 0; i < matchesof1.size(); i++)
		{
			if (matchesof1[i] >= 0)
			{
				matches[count].first = i;
				matches[count].second = matchesof1[i];
				count++;
			}
		}
	}

	bool FeatureMatching::KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, cv::Mat & descriptors1, 
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2,
		std::vector<std::pair<int, int>> &matches)
	{
		float thRatio = 0.5;
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

		//

		std::vector<float> th_epipolar(2);
		th_epipolar[0] = 3.0;
		th_epipolar[1] = 1.0;

		int iter = 0;
		while (iter < 2)
		{
			std::vector<cv::Point2f> pt1, pt2;
			std::vector<int> id_pt;
			for (size_t i = 0; i < matchesof1.size(); i++)
			{
				if (matchesof1[i] >= 0)
				{
					pt1.push_back(kp1[i].pt);
					pt2.push_back(kp2[matchesof1[i]].pt);
					id_pt.push_back(i);
				}
			}
			if (pt1.size() < th_reject)
			{
				return false;
			}

			// find homography
			cv::Mat HMatrix = cv::findHomography(pt1, pt2);
			double diag_sum =  + HMatrix.at<double>(1, 1) + HMatrix.at<double>(2, 2);
			if (std::abs(HMatrix.at<double>(0, 0) - 0.995) < 0.01 &&
				std::abs(HMatrix.at<double>(1, 1) - 0.995) < 0.01 &&
				std::abs(HMatrix.at<double>(2, 2) - 0.995) < 0.01)
			{
				return false;
			}

			std::vector<uchar> ransac_status(pt1.size());
			cv::Mat FMatrix = cv::findFundamentalMat(pt1, pt2, ransac_status, cv::FM_RANSAC, th_epipolar[iter]);
			for (size_t i = 0; i < ransac_status.size(); i++)
			{
				if (!ransac_status[i])
				{
					matchesof1[id_pt[i]] = -1;
				}
			}

			iter++;
		}

		//
		for (size_t i = 0; i < matchesof1.size(); i++)
		{
			if (matchesof1[i] >= 0)
			{
				matches.push_back(std::pair<int,int>(i, matchesof1[i]));
			}
		}
	}

	bool FeatureMatching::KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, cv::Mat & descriptors1, std::vector<cv::KeyPoint>& kp2, cv::flann::Index * kdindex2, std::vector<std::pair<int, int>>& matches)
	{
		float thRatio = 0.5;
		int th_reject = 20;

		if (kp1.size() < th_reject || kp2.size() < th_reject)
		{
			return false;
		}

		std::vector<int> matchesof1;
		matchesof1.resize(kp1.size(), -1);
		cv::Mat mindices, mdists;
		for (size_t i = 0; i < kp1.size(); i++)
		{
			kdindex2->knnSearch(descriptors1.row(i), mindices, mdists, 2, cv::flann::SearchParams(32));
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
			std::vector<cv::Point2f> pt1, pt2;
			std::vector<int> id_pt;
			for (size_t i = 0; i < matchesof1.size(); i++)
			{
				if (matchesof1[i] >= 0)
				{
					pt1.push_back(kp1[i].pt);
					pt2.push_back(kp2[matchesof1[i]].pt);
					id_pt.push_back(i);
				}
			}
			if (pt1.size() < th_reject)
			{
				return false;
			}

			// find homography
			cv::Mat HMatrix = cv::findHomography(pt1, pt2);
			double diag_sum = +HMatrix.at<double>(1, 1) + HMatrix.at<double>(2, 2);
			if (std::abs(HMatrix.at<double>(0, 0) - 1.0) < 0.05 &&
				std::abs(HMatrix.at<double>(1, 1) - 1.0) < 0.05 &&
				std::abs(HMatrix.at<double>(2, 2) - 1.0) < 0.05)
			{
				return false;
			}

			std::vector<uchar> ransac_status(pt1.size());
			cv::Mat FMatrix = cv::findFundamentalMat(pt1, pt2, ransac_status, cv::FM_RANSAC, th_epipolar[iter]);
			for (size_t i = 0; i < ransac_status.size(); i++)
			{
				if (!ransac_status[i])
				{
					matchesof1[id_pt[i]] = -1;
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

		return true;
	}

	bool FeatureMatching::KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, cv::flann::Index * kdindex1, 
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2, std::vector<std::pair<int, int>>& matches)
	{
		float thRatio = 0.5;
		int th_reject = 20;

		if (kp1.size() < th_reject || kp2.size() < th_reject)
		{
			return false;
		}

		std::vector<int> matchesof2;
		matchesof2.resize(kp2.size(), -1);
		for (size_t i = 0; i < kp2.size(); i++)
		{
			cv::Mat mindices, mdists;
			kdindex1->knnSearch(descriptors2.row(i), mindices, mdists, 2, cv::flann::SearchParams(32));
			float ratio = mdists.at<float>(0) / mdists.at<float>(1);
			if (ratio < thRatio)
			{
				matchesof2[i] = mindices.at<int>(0);
			}
		}

		int count = 0;
		for (size_t i = 0; i < matchesof2.size(); i++)
		{
			if (matchesof2[i] >= 0)
			{
				count++;
			}
		}
		if (count < th_reject) return false;

		std::vector<cv::Point2f> pt1(count), pt2(count);
		std::vector<int> id_pt(count);
		count = 0;
		for (size_t i = 0; i < matchesof2.size(); i++)
		{
			if (matchesof2[i] >= 0)
			{
				pt1[count] = kp1[matchesof2[i]].pt;
				pt2[count] = kp2[i].pt;
				id_pt[count] = i;
				count++;
			}
		}

		// find homography
		cv::Mat HMatrix = cv::findHomography(pt1, pt2);
		double diag_sum = HMatrix.at<double>(1, 1) + HMatrix.at<double>(2, 2);
		if (std::abs(HMatrix.at<double>(0, 0) - 1.0) < 0.05 &&
			std::abs(HMatrix.at<double>(1, 1) - 1.0) < 0.05 &&
			std::abs(HMatrix.at<double>(2, 2) - 1.0) < 0.05)
		{
			return false;
		}

		std::vector<uchar> ransac_status(pt1.size());
		cv::Mat FMatrix = cv::findFundamentalMat(pt1, pt2, ransac_status, cv::FM_RANSAC, 3.0);
		count = 0;
		for (size_t i = 0; i < ransac_status.size(); i++)
		{
			if (!ransac_status[i])
			{
				matchesof2[id_pt[i]] = -1;
			}
			else
				count++;
		}

		//
		matches.resize(count);
		count = 0;
		for (size_t i = 0; i < matchesof2.size(); i++)
		{
			if (matchesof2[i] >= 0)
			{
				matches[count] = std::pair<int, int>(matchesof2[i], i);
				count++;
			}
		}

		return true;
	}

	bool FeatureMatching::KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, my_kd_tree_t * kd_tree1, 
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2, std::vector<std::pair<int, int>>& matches)
	{
		float thRatio = 0.5;
		int th_reject = 20;

		if (kp1.size() < th_reject || kp2.size() < th_reject)
		{
			return false;
		}

		std::vector<int> matchesof2;
		matchesof2.resize(kp2.size(), -1);
		for (size_t i = 0; i < kp2.size(); i++)
		{
			size_t id[2];
			float dis[2];
			kd_tree1->knnSearch((float*)descriptors2.ptr<float>(i), 2, id, dis);
			float ratio = dis[0] / dis[1];
			if (ratio < thRatio)
			{
				matchesof2[i] = id[0];
			}
		}

		// geometric verification
		std::vector<cv::Point2f> pt1, pt2;
		std::vector<int> id_pt;
		for (size_t i = 0; i < matchesof2.size(); i++)
		{
			if (matchesof2[i] >= 0)
			{
				pt2.push_back(kp2[i].pt);
				pt1.push_back(kp1[matchesof2[i]].pt);
				id_pt.push_back(i);
			}
		}
		if (pt1.size() < th_reject)
		{
			return false;
		}

		// find homography
		cv::Mat HMatrix = cv::findHomography(pt1, pt2);
		double diag_sum = HMatrix.at<double>(1, 1) + HMatrix.at<double>(2, 2);
		if (std::abs(HMatrix.at<double>(0, 0) - 1.0) < 0.05 &&
			std::abs(HMatrix.at<double>(1, 1) - 1.0) < 0.05 &&
			std::abs(HMatrix.at<double>(2, 2) - 1.0) < 0.05)
		{
			return false;
		}

		std::vector<uchar> ransac_status(pt1.size());
		cv::Mat FMatrix = cv::findFundamentalMat(pt1, pt2, ransac_status, cv::FM_RANSAC, 1.5);
		for (size_t i = 0; i < ransac_status.size(); i++)
		{
			if (!ransac_status[i])
			{
				matchesof2[id_pt[i]] = -1;
			}
		}

		//
		for (size_t i = 0; i < matchesof2.size(); i++)
		{
			if (matchesof2[i] >= 0)
			{
				matches.push_back(std::pair<int, int>(matchesof2[i], i));
			}
		}

		return true;
	}

	bool FeatureMatching::KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, flann_index_t * kd_tree1, FLANNParameters &p,
		std::vector<cv::KeyPoint>& kp2, cv::Mat & descriptors2, std::vector<std::pair<int, int>>& matches)
	{
		float thRatio = 0.5;
		int th_reject = 20;

		if (kp1.size() < th_reject || kp2.size() < th_reject)
		{
			return false;
		}

		// flann matching
		int k = 2;
		int count = kp2.size();
		int* id = new int[count * 2];
		float* dis = new float[count * 2];
		flann_find_nearest_neighbors_index(*kd_tree1, (float*)descriptors2.data, count, id, dis, 2, &p);

		std::vector<int> matchesof2;
		matchesof2.resize(kp2.size(), -1);
		int* ptr_id = id;
		float* ptr_dis = dis;
		for (size_t i = 0; i < kp2.size(); i++)
		{
			float ratio = ptr_dis[0] / ptr_dis[1];
			if (ratio < thRatio)
			{
				matchesof2[i] = ptr_id[0];
			}
			ptr_id += 2;
			ptr_dis += 2;
		}
		delete[] id;
		delete[] dis;

		// geometric verification
		std::vector<cv::Point2f> pt1, pt2;
		std::vector<int> id_pt;
		for (size_t i = 0; i < matchesof2.size(); i++)
		{
			if (matchesof2[i] >= 0)
			{
				pt2.push_back(kp2[i].pt);
				pt1.push_back(kp1[matchesof2[i]].pt);
				id_pt.push_back(i);
			}
		}
		if (pt1.size() < th_reject)
		{
			return false;
		}

		// find homography
		cv::Mat HMatrix = cv::findHomography(pt1, pt2);
		double diag_sum = HMatrix.at<double>(1, 1) + HMatrix.at<double>(2, 2);
		if (std::abs(HMatrix.at<double>(0, 0) - 1.0) < 0.05 &&
			std::abs(HMatrix.at<double>(1, 1) - 1.0) < 0.05 &&
			std::abs(HMatrix.at<double>(2, 2) - 1.0) < 0.05)
		{
			return false;
		}

		std::vector<uchar> ransac_status(pt1.size());
		cv::Mat FMatrix = cv::findFundamentalMat(pt1, pt2, ransac_status, cv::FM_RANSAC, 1.5);
		for (size_t i = 0; i < ransac_status.size(); i++)
		{
			if (!ransac_status[i])
			{
				matchesof2[id_pt[i]] = -1;
			}
		}

		//
		for (size_t i = 0; i < matchesof2.size(); i++)
		{
			if (matchesof2[i] >= 0)
			{
				matches.push_back(std::pair<int, int>(matchesof2[i], i));
			}
		}

		return true;
	}

	bool FeatureMatching::KNNMatchingWithGeoVerify(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2, 
		int * id, float * dis, std::vector<std::pair<int, int>>& matches)
	{
		float thRatio = 0.5;
		int th_reject = 20;

		if (kp1.size() < th_reject || kp2.size() < th_reject)
		{
			return false;
		}

		std::vector<int> matchesof2;
		matchesof2.resize(kp2.size(), -1);
		int* ptr_id = id;
		float* ptr_dis = dis;
		for (size_t i = 0; i < kp2.size(); i++)
		{
			float ratio = ptr_dis[0] / ptr_dis[1];
			if (ratio < thRatio)
			{
				matchesof2[i] = ptr_id[0];
			}
			ptr_id += 2;
			ptr_dis += 2;
		}

		// geometric verification
		std::vector<cv::Point2f> pt1, pt2;
		std::vector<int> id_pt;
		for (size_t i = 0; i < matchesof2.size(); i++)
		{
			if (matchesof2[i] >= 0)
			{
				pt2.push_back(kp2[i].pt);
				pt1.push_back(kp1[matchesof2[i]].pt);
				id_pt.push_back(i);
			}
		}
		if (pt1.size() < th_reject)
		{
			return false;
		}

		// find homography
		cv::Mat HMatrix = cv::findHomography(pt1, pt2);
		double diag_sum = HMatrix.at<double>(1, 1) + HMatrix.at<double>(2, 2);
		if (std::abs(HMatrix.at<double>(0, 0) - 1.0) < 0.05 &&
			std::abs(HMatrix.at<double>(1, 1) - 1.0) < 0.05 &&
			std::abs(HMatrix.at<double>(2, 2) - 1.0) < 0.05)
		{
			return false;
		}

		std::vector<uchar> ransac_status(pt1.size());
		cv::Mat FMatrix = cv::findFundamentalMat(pt1, pt2, ransac_status, cv::FM_RANSAC, 3.0);
		for (size_t i = 0; i < ransac_status.size(); i++)
		{
			if (!ransac_status[i])
			{
				matchesof2[id_pt[i]] = -1;
			}
		}

		//
		for (size_t i = 0; i < matchesof2.size(); i++)
		{
			if (matchesof2[i] >= 0)
			{
				matches.push_back(std::pair<int, int>(matchesof2[i], i));
			}
		}

		return true;
	}


	bool FeatureMatching::KNNMatchingWithGeoVerify(std::vector<cv::Point2f>& kp1, cv::Mat & descriptors1,
		std::vector<cv::Point2f>& kp2, cv::flann::Index * kdindex2,
		std::vector<std::pair<int, int>>& matches)
	{
		float thRatio = 0.5;
		int th_reject = 20;

		if (kp1.size() < th_reject || kp2.size() < th_reject)
		{
			return false;
		}

		std::vector<int> matchesof1;
		matchesof1.resize(kp1.size(), -1);
		cv::Mat mindices, mdists;
		for (size_t i = 0; i < kp1.size(); i++)
		{
			kdindex2->knnSearch(descriptors1.row(i), mindices, mdists, 2, cv::flann::SearchParams(32));
			float ratio = mdists.at<float>(0) / mdists.at<float>(1);
			if (ratio < thRatio)
			{
				matchesof1[i] = mindices.at<int>(0);
			}
		}

		std::vector<float> th_epipolar(2);
		th_epipolar[0] = 4.0;
		th_epipolar[1] = 2.0;

		int iter = 0;
		while (iter < 2)
		{
			std::vector<cv::Point2f> pt1, pt2;
			std::vector<int> id_pt;
			for (size_t i = 0; i < matchesof1.size(); i++)
			{
				if (matchesof1[i] >= 0)
				{
					pt1.push_back(kp1[i]);
					pt2.push_back(kp2[matchesof1[i]]);
					id_pt.push_back(i);
				}
			}
			if (pt1.size() < th_reject)
			{
				return false;
			}

			std::vector<uchar> ransac_status(pt1.size());
			cv::Mat FMatrix = cv::findFundamentalMat(pt1, pt2, ransac_status, cv::FM_RANSAC, th_epipolar[iter]);
			//std::cout << FMatrix << std::endl;

			for (size_t i = 0; i < ransac_status.size(); i++)
			{
				if (!ransac_status[i])
				{
					matchesof1[id_pt[i]] = -1;
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

		return true;
	}

	void FeatureMatching::GenerateKDIndex(cv::Mat & descriptors, cv::flann::Index ** kdindex)
	{
		*kdindex = new cv::flann::Index(descriptors, cv::flann::KDTreeIndexParams(4));
	}

}