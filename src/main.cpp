#include <iostream>
#include <array>
#include <algorithm>
#include <cmath>
#include "image.h"

GrayscaleImage sobel_filter(const GrayscaleImage &img, std::vector<std::vector<bool>> &edges)
{
    int width = img.GetWidth();
    int height = img.GetHeight();
    edges.assign(height, std::vector<bool>(width, false));
    GrayscaleImage edge_detected_img(width, height);
    std::array<std::array<int, 3>, 3> filter_x = {{{-1, 0, 1},
                                                   {-2, 0, 2},
                                                   {-1, 0, 1}}};

    std::array<std::array<int, 3>, 3> filter_y = {{{1, 2, 1},
                                                   {0, 0, 0},
                                                   {-1, -2, -1}}};

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            int gx = 0, gy = 0;
            for (int i = -1; i <= 1; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    gx += filter_x[j + 1][i + 1] * img(x + i, y + j);
                    gy += filter_y[j + 1][i + 1] * img(x + i, y + j);
                }
            }
            edge_detected_img(x, y) = car(std::abs(gx) + std::abs(gy), 255);
            int mag2 = gx * gx + gy * gy;
            if (mag2 > (50 * 50))
                edges[y][x] = true;
        }
    }
    return edge_detected_img;
}

double estimate_skew_hough(const std::vector<std::vector<bool>> &edges,
                           double max_angle = M_PI / 36,
                           double angle_step = M_PI / 1800)
{
    int h = edges.size(), w = edges[0].size();
    int n_t = int((2 * max_angle) / angle_step) + 1;
    std::vector<double> thetas(n_t);
    for (int i = 0; i < n_t; ++i)
        thetas[i] = -max_angle + i * angle_step;

    double diag = std::sqrt(h * h + w * w);
    int rho_max = int(std::ceil(diag));
    int n_r = 2 * rho_max + 1;

    std::vector<std::vector<int>> acc(n_r, std::vector<int>(n_t, 0));

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            if (!edges[y][x])
                continue;
            for (int t = 0; t < n_t; ++t)
            {
                double rho = x * std::cos(thetas[t]) + y * std::sin(thetas[t]);
                int r = int(std::round(rho)) + rho_max;
                if (r >= 0 && r < n_r)
                    acc[r][t]++;
            }
        }
    }

    int best_r = 0, best_t = 0, best_votes = -1;
    for (int r = 0; r < n_r; ++r)
    {
        for (int t = 0; t < n_t; ++t)
        {
            if (acc[r][t] > best_votes)
            {
                best_votes = acc[r][t];
                best_r = r;
                best_t = t;
            }
        }
    }

    double skew = thetas[best_t];
    return skew;
}

double cubic_interpolate(double p0, double p1, double p2, double p3, double t)
{
    double a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    double b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    double c = -0.5 * p0 + 0.5 * p2;
    double d = p1;
    return a * t * t * t + b * t * t + c * t + d;
}

GrayscaleImage rotate_image_bicubic(const GrayscaleImage &src, double angle)
{
    int w = src.GetWidth();
    int h = src.GetHeight();
    GrayscaleImage dst(w, h);

    double cx = w * 0.5;
    double cy = h * 0.5;
    double c = std::cos(angle);
    double s = std::sin(angle);

    auto safe_get_pixel = [&](int x, int y) -> double
    {
        if (x < 0 || x >= w || y < 0 || y >= h)
            return 255.0;
        return static_cast<double>(src(x, y));
    };

    for (int y2 = 0; y2 < h; ++y2)
    {
        for (int x2 = 0; x2 < w; ++x2)
        {
            double dx = x2 - cx;
            double dy = y2 - cy;
            double xs = dx * c - dy * s + cx;
            double ys = dx * s + dy * c + cy;

            int x0 = int(std::floor(xs));
            int y0 = int(std::floor(ys));
            double fx = xs - x0;
            double fy = ys - y0;

            std::array<double, 4> rows;
            for (int j = 0; j < 4; ++j)
            {
                int y_cur = y0 - 1 + j;
                std::array<double, 4> pixels = {
                    safe_get_pixel(x0 - 1, y_cur),
                    safe_get_pixel(x0, y_cur),
                    safe_get_pixel(x0 + 1, y_cur),
                    safe_get_pixel(x0 + 2, y_cur)};
                rows[j] = cubic_interpolate(pixels[0], pixels[1], pixels[2], pixels[3], fx);
            }

            double result = cubic_interpolate(rows[0], rows[1], rows[2], rows[3], fy);
            dst(x2, y2) = static_cast<unsigned char>(std::clamp(result, 0.0, 255.0));
        }
    }
    return dst;
}

int main()
{
    GrayscaleImage input;
    input.Load("../input/skew-origin.png");

    std::vector<std::vector<bool>> edges;
    GrayscaleImage edge_detected_img = sobel_filter(input, edges);

    double skew_rad = estimate_skew_hough(edges);
    double skew_deg = skew_rad * 180.0 / M_PI;

    std::cout << "Estimated skew: "
              << skew_deg << "° (" << skew_rad << " rad)\n";

    GrayscaleImage deskewed = rotate_image_bicubic(input, -skew_rad);

    std::vector<std::vector<bool>> edges_deskewed;
    GrayscaleImage edge_detected_img_deskewed = sobel_filter(deskewed, edges_deskewed);

    double skew_rad_after_deskew = estimate_skew_hough(edges_deskewed);
    double skew_deg_after_deskew = skew_rad_after_deskew * 180.0 / M_PI;

    std::cout << "Skew after deskewing: "
              << skew_deg_after_deskew << "° (" << skew_rad_after_deskew << " rad)\n";

    edge_detected_img.Save("../output/edge-detected-document.png");
    deskewed.Save("../output/deskewed.png");

    return 0;
}