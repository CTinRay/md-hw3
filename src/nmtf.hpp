#include <eigen3/Eigen/Dense>

#ifndef NMTF_HPP
#define NMTF_HPP

Eigen::MatrixXd nmtf(double rate, double conv, Eigen::MatrixXd x, Eigen::MatrixXd mask);

#endif
