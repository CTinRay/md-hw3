#include "nmtf.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <utility>

void nmtf2(double rate, double conv,
           int d1, int d2,
           const Eigen::MatrixXd x,
           const Eigen::MatrixXd mask,
           Eigen::MatrixXd&f,
           Eigen::MatrixXd&s,
           Eigen::MatrixXd&g) {
    double d = std::numeric_limits<double>::infinity();
    f.setRandom(x.rows(), d1).array() /= 10000;
    f.array() += 1;
    g.setRandom(x.cols(), d2).array() /= 10000;
    g.array() += 1;
    s.setRandom(d1, d2).array() /= 10000;
    s.array() += 1;

    std::cout << "f:" << f.row(0) << std::endl;
    std::cout << "g:" << g.row(0) << std::endl;

    std::vector<std::pair<int, int>>ones;
    for (auto i = 0u; i < mask.rows(); ++i) {
        for (auto j = 0u; j < mask.cols(); ++j) {
            if (mask(i, j) == 1) {
                ones.push_back(std::make_pair(i, j));
            }
        }
    }
       
    std::cout << "loss: " << (x - f * s * g.transpose()).cwiseProduct(mask).squaredNorm() / ones.size() << std::endl;

    std::cout << "NMTF..." << std::endl;
    while (d > conv * conv) {
        d = 0;
        
        for (auto&p : ones) {
            int i = p.first;
            int j = p.second;
            double e = x(i, j) - f.row(i) * s * g.row(j).transpose();
            auto dfi = - e * s * g.row(j).transpose();
            auto dgi = - e * f.row(i) * s;
            auto ds  = - e * f.row(i).transpose() * g.row(j);
            d += dfi.squaredNorm() + dgi.squaredNorm() + ds.squaredNorm();
            if (std::isnan(d)) {
                std::cout << "dfi:" << dfi << std::endl
                          << "dgi:" << dgi << std::endl
                          << "ds:"  << ds  << std::endl;
            }
            
            f.row(i).array() -= dfi.array() * rate;
            g.row(j).array() -= dgi.array() * rate;
            s -= ds * rate;        
        }

        std::cout // << "s:" << s << std::endl
            << "d:" << d << std::endl
            << "f:" << f.row(0) << std::endl
            << "g:" << g.row(0) << std::endl
            << "loss: " << (x - f * s * g.transpose()).cwiseProduct(mask).squaredNorm() / ones.size()  << std::endl;
        
    }
}


void nmtf(double rate, double conv,
           int d1, int d2,
           const Eigen::MatrixXd x_orig,
           const Eigen::MatrixXd mask,
           Eigen::MatrixXd&f,
           Eigen::MatrixXd&s,
           Eigen::MatrixXd&g) {
    Eigen::MatrixXd x = x_orig;

    double sum = 0;
    int cnt = 0;
    for (auto i = 0u; i < mask.rows(); ++i) {
        for (auto j = 0u; j < mask.cols(); ++j) {
            if (mask(i, j) == 1) {
                sum += x(i, j);
                cnt += 1;
            }
        }
    }
    double avg = sum / cnt;
    for (auto i = 0u; i < mask.rows(); ++i) {
        for (auto j = 0u; j < mask.cols(); ++j) {
            if (mask(i, j) == 0) {
                x(i, j) = avg;                
            }
        }
    }
    
    
    f.setRandom(x.rows(), d1);
    f.array() /= 10;
    f.array() += 1;
    g.setRandom(x.cols(), d2);
    g.array() /= 10;
    g.array() += 1;
    s.setRandom(d1, d2);

    std::cout << "f:" << f.row(0) << std::endl;
    std::cout << "g:" << g.row(0) << std::endl;

    std::cout << "NMTF..." << std::endl;

    double d = std::numeric_limits<double>::infinity();
    while (d > conv * conv) {
        
        auto gprev = g;
        auto fprev = f;
        auto sprev = s;

        std::cout << "loss: " << (x - f * s * g.transpose()).norm() << std::endl;
        g.array() = g.array() * ((x.transpose() * f * s).array() /
                                 (g * (g.transpose() * x.transpose()) * (f * s)).array()).sqrt();

        std::cout << "loss: " << (x - f * s * g.transpose()).norm() << std::endl;        
        f.array() = f.array() * ((x * g * s.transpose()).array() /
                                 ((f * (f.transpose() * x)) * (g * s.transpose()) ).array()).sqrt();

        std::cout << "loss: " << (x - f * s * g.transpose()).norm() << std::endl;
        s.array() = s.array() * ((f.transpose() * x * g).array() /
                                 (((f.transpose() * f) * s) * (g.transpose() * g)).array()).sqrt();
        
        
        d = (fprev - f).norm() + (gprev - g).norm() + (sprev - s).norm();
        std::cout // << "s:" << s << std::endl
            << "d:" << d << std::endl
            << "f:" << f.row(0) << std::endl
            << "g:" << g.row(0) << std::endl
            << "loss: " << (x - f * s * g.transpose()).norm() << std::endl;
        
    }
}
