#pragma once

#include <cmath>
#include <algorithm>

namespace bs {

inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

inline double norm_pdf(double x) {
    static const double inv_sqrt_2pi = 0.3989422804014327;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

inline double d1(double S, double K, double T, double r, double q, double sigma) {
    return (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
}

inline double d2(double S, double K, double T, double r, double q, double sigma) {
    return d1(S, K, T, r, q, sigma) - sigma * std::sqrt(T);
}

inline double bs_price(double S, double K, double T, double r, double q, double sigma, bool is_call = true) {
    if (T <= 0.0 || sigma <= 0.0) {
        double intrinsic = is_call ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
        return intrinsic;
    }
    double D1 = d1(S, K, T, r, q, sigma);
    double D2 = D1 - sigma * std::sqrt(T);
    double df_q = std::exp(-q * T);
    double df_r = std::exp(-r * T);
    if (is_call) {
        return S * df_q * norm_cdf(D1) - K * df_r * norm_cdf(D2);
    }
    return K * df_r * norm_cdf(-D2) - S * df_q * norm_cdf(-D1);
}

inline double bs_delta(double S, double K, double T, double r, double q, double sigma, bool is_call = true) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double D1 = d1(S, K, T, r, q, sigma);
    double df_q = std::exp(-q * T);
    return is_call ? df_q * norm_cdf(D1) : df_q * (norm_cdf(D1) - 1.0);
}

inline double bs_gamma(double S, double K, double T, double r, double q, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double D1 = d1(S, K, T, r, q, sigma);
    double df_q = std::exp(-q * T);
    return df_q * norm_pdf(D1) / (S * sigma * std::sqrt(T));
}

inline double bs_vega(double S, double K, double T, double r, double q, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double D1 = d1(S, K, T, r, q, sigma);
    double df_q = std::exp(-q * T);
    return S * df_q * norm_pdf(D1) * std::sqrt(T);
}

inline double bs_theta(double S, double K, double T, double r, double q, double sigma, bool is_call = true) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    double D1 = d1(S, K, T, r, q, sigma);
    double D2 = D1 - sigma * std::sqrt(T);
    double df_q = std::exp(-q * T);
    double df_r = std::exp(-r * T);
    double term1 = -(S * df_q * norm_pdf(D1) * sigma) / (2.0 * std::sqrt(T));
    if (is_call) {
        return term1 - r * K * df_r * norm_cdf(D2) + q * S * df_q * norm_cdf(D1);
    }
    return term1 + r * K * df_r * norm_cdf(-D2) - q * S * df_q * norm_cdf(-D1);
}

inline double bs_implied_vol(double price, double S, double K, double T, double r, double q, bool is_call = true) {
    double sigma = 0.2;
    const double tol = 1e-8;
    const int max_iter = 50;
    for (int i = 0; i < max_iter; ++i) {
        double p = bs_price(S, K, T, r, q, sigma, is_call);
        double v = bs_vega(S, K, T, r, q, sigma);
        double diff = p - price;
        if (std::abs(diff) < tol) return sigma;
        if (v < 1e-12) {
            sigma = sigma * 1.5 + 1e-4;
            continue;
        }
        double next = sigma - diff / v;
        if (next <= 0.0) next = sigma / 2.0;
        if (next > 5.0) next = 5.0;
        sigma = next;
    }
    return sigma;
}

} // namespace bs
