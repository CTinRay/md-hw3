#include "utils.hpp"
#include <fstream>
#include <cassert>
#include <iostream>
#include <cassert>

void loadMatrix(const std::string& filename,
                Eigen::MatrixXd&matrix, Eigen::MatrixXd&mask){
    assert(matrix.rows() == mask.rows());
    assert(matrix.cols() == mask.cols());
    
    std::fstream f;
    f.open(filename, std::ios::in);
    int x, y;
    double rate;
    f >> x >> y >> rate;
    while (!f.eof()) {
        assert(x < matrix.rows());
        assert(y < matrix.cols());
        matrix(x, y) = rate;
        mask(x, y) = 1;
        f >> x >> y >> rate;   
    }
}


void loadModel(const std::string& filename,
               Eigen::MatrixXd& m1, Eigen::MatrixXd& m2) {
    std::fstream f;
    f.open(filename, std::ios::in);
    int f0, m, n, k;
    float b;
    std::string buf;
    f >> buf >> f0;
    f >> buf >> m;
    f >> buf >> n;
    f >> buf >> k;
    f >> buf >> b;

    m1 = Eigen::MatrixXd::Zero(m, k);
    m2 = Eigen::MatrixXd::Zero(n, k);
    
    for (auto i = 0; i < m; ++i) {
        f >> buf;
        f >> buf;
        for (auto j = 0; j < k; ++j) {
            f >> m1(i, j);
        }
    }

    for (auto i = 0; i < n; ++i) {
        f >> buf >> buf;
        for (auto j = 0; j < k; ++j) {
            f >> m2(i, j);
        }
    }

    f.close();
}


void loadMem(const std::string& filename, Eigen::MatrixXd& mem) {
    std::fstream f;
    f.open(filename, std::ios::in);
    for (auto i = 0u; i < mem.rows(); ++i) {
        int j = 0;
        f >> j;
        assert(j >= 0 && j < mem.cols());
        mem(i, j) = 1;
    }
}
