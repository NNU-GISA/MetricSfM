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

// EPNP is more accurate than P3P, the P3P may suffers from error matching
// while EPnP uses more correspondences to get a robuster result
#ifndef OBJECTSFM_CAMERA_ABSOLUTE_POSE_EPNP_H_
#define OBJECTSFM_CAMERA_ABSOLUTE_POSE_EPNP_H_

#include <vector>
#include <opencv/cv.h>
#include "basic_structs.h"

namespace objectsfm {

	class AbsolutePoseEPNP 
	{
	public:
		AbsolutePoseEPNP(void);

		~AbsolutePoseEPNP();

		AbsolutePoseEPNP(const AbsolutePoseEPNP& e);

		AbsolutePoseEPNP& operator=(const AbsolutePoseEPNP& e);

		void EPNPRobust(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d,
			double f, RTPose &pose_absolute, double &error);

		void EPNPRansac(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d,
			double f, int num_pts_per_iter, int max_iter, RTPose &pose_absolute, double &error);

		bool EPNP(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d,
			int time_stamp, double f, int num_pts_per_iter, RTPose &pose_absolute);

		bool EPNP(std::vector<Eigen::Vector3d> &pts_w, std::vector<Eigen::Vector2d> &pts_2d,
			double f, RTPose &pose_absolute, double &error);

		void Error(const std::vector<Eigen::Vector3d>& pts_w, const std::vector<Eigen::Vector2d>& pts_2d, 
			double f, RTPose pose, std::vector<double> &error_reproj);

		bool CheckPtQuality(std::vector<Eigen::Vector2d> &pts_2d);

	private:
		int get_correspondence_number() { return number_of_correspondences; }

		void set_internal_parameters(const double uc, const double vc, const double fu, const double fv);

		void set_maximum_number_of_correspondences(const int n);

		void reset_correspondences(void);

		void add_correspondence(const double X, const double Y, const double Z, const double u, const double v);

		double compute_pose(double R[3][3], double T[3]);

		void relative_error(double & rot_err, double & transl_err,
			const double Rtrue[3][3], const double ttrue[3],
			const double Rest[3][3],  const double test[3]);

		void print_pose(const double R[3][3], const double t[3]);

		double reprojection_error(const double R[3][3], const double t[3]);

		void choose_control_points(void);

		void compute_barycentric_coordinates(void);

		void fill_M(CvMat * M, const int row, const double * alphas, const double u, const double v);

		void compute_ccs(const double * betas, const double * ut);

		void compute_pcs(void);

		void solve_for_sign(void);

		void find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho, double * betas);

		void find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho, double * betas);

		void find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho, double * betas);

		void qr_solve(CvMat * A, CvMat * b, CvMat * X);

		double dot(const double * v1, const double * v2);

		double dist2(const double * p1, const double * p2);

		void compute_rho(double * rho);

		void compute_L_6x10(const double * ut, double * l_6x10);

		void gauss_newton(const CvMat * L_6x10, const CvMat * Rho, double current_betas[4]);

		void compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho, double cb[4], CvMat * A, CvMat * b);

		double compute_R_and_t(const double * ut, const double * betas, double R[3][3], double t[3]);

		void estimate_R_and_t(double R[3][3], double t[3]);

		void copy_R_and_t(const double R_dst[3][3], const double t_dst[3], double R_src[3][3], double t_src[3]);

		void mat_to_quat(const double R[3][3], double q[4]);

		void rand_vector(int v_min, int v_max, std::vector<int>& values);

		double uc, vc, fu, fv;
		//pws		point 3d coordinates in world coordinate
		//us		pixel coordinates in camera coordinate
		//alphas	points alphas coordinates in the reference of four control points
		//pcs		point 3d coordinates in camera coordinate
		double * pws, * us, * alphas, * pcs;
		int maximum_number_of_correspondences;
		int number_of_correspondences;

		//control points coordinates in world coordinate and camera coordinate
		double cws[4][3], ccs[4][3];
		double cws_determinant;
	};

}
#endif //OBJECTSFM_CAMERA_ABSOLUTE_POSE_EPNP_H_