#include <iostream>
#include <array>
#include <algorithm>
#include <cmath>
#include <vector>
#include <filesystem>
#include "image.h"

namespace fs = std::filesystem;

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
                           double max_angle = M_PI / 6,
                           double angle_step = M_PI / 3600)
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

    double center_y = h * 0.5;
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            // non voting pixels
            if (!edges[y][x])
                continue;

            // more weight (double) given to edges in the  middle
            double weight = 1.0;
            double dist_from_center = std::abs(y - center_y) / (h * 0.5);
            if (dist_from_center < 0.5)
            {
                weight = 2.0;
            }

            // for each theta (in our grid) find the corresponding p
            for (int t = 0; t < n_t; ++t)
            {
                double rho = x * std::cos(thetas[t]) + y * std::sin(thetas[t]);
                int r = int(std::round(rho)) + rho_max;
                // if the combination of r,t is in our range, then cast a vote.
                if (r >= 0 && r < n_r)
                    acc[r][t] += static_cast<int>(weight);
            }
        }
    }

    std::vector<std::pair<int, std::pair<int, int>>> peaks;

    for (int r = 1; r < n_r - 1; ++r)
    {
        for (int t = 1; t < n_t - 1; ++t)
        {
            int votes = acc[r][t];
            if (votes > 10) // minimum threshold
            {
                // is peak? or in other words, is local maximum?
                // we compare with the 8 immediate neighbors
                bool is_peak = true;
                for (int dr = -1; dr <= 1 && is_peak; ++dr)
                {
                    for (int dt = -1; dt <= 1 && is_peak; ++dt)
                    {
                        if (dr == 0 && dt == 0)
                            continue;
                        if (acc[r + dr][t + dt] >= votes)
                        {
                            is_peak = false;
                        }
                    }
                }
                if (is_peak)
                {
                    // found a peak, record it.
                    peaks.push_back({votes, {r, t}});
                }
            }
        }
    }

    if (peaks.empty())
    {
        // fallback to global maximum
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
        return thetas[best_t];
    }
    // else, we can choose a heuristic that might result in better theta calculation.

    // votes sorting in descending order
    std::sort(peaks.begin(), peaks.end(), std::greater<>());

    // heurisitc #1
    // consider the peak with the smallest absolute angle (closest to 0 degrees)
    // or in other words, closest to horizontal.
    // to be the best.
    double best_angle = thetas[peaks[0].second.second];
    for (const auto &peak : peaks)
    {
        // we're constraining to peaks that atleast have a vote count 70%
        // of the top vote count
        double angle = thetas[peak.second.second];
        if (std::abs(angle) < std::abs(best_angle) && peak.first > peaks[0].first * 0.7)
        {
            best_angle = angle;
        }
    }

    return best_angle;
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
            return 0;
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
            dst(x2, y2) = static_cast<unsigned char>(car(result, 255.0));
        }
    }
    return dst;
}

void process_image(const std::string &input_path, const std::string &base_filename)
{
    std::cout << "\n=== Processing: " << input_path << " ===" << std::endl;

    GrayscaleImage input;
    input.Load(input_path);

    std::vector<std::vector<bool>> edges;
    GrayscaleImage edge_detected_img = sobel_filter(input, edges);

    std::string edge_filename = "../output/" + base_filename + "-edges.png";
    edge_detected_img.Save(edge_filename);

    double skew_rad = estimate_skew_hough(edges);
    double skew_deg = skew_rad * 180.0 / M_PI;

    std::cout << "Estimated skew: " << skew_deg << "° (" << skew_rad << " rad)" << std::endl;

    GrayscaleImage deskewed = rotate_image_bicubic(input, skew_rad);

    std::string corrected_filename = "../output/" + base_filename + "-corrected.png";
    deskewed.Save(corrected_filename);

    std::vector<std::vector<bool>> edges_deskewed;
    GrayscaleImage edge_detected_img_deskewed = sobel_filter(deskewed, edges_deskewed);

    double skew_rad_after_deskew = estimate_skew_hough(edges_deskewed);
    double skew_deg_after_deskew = skew_rad_after_deskew * 180.0 / M_PI;

    std::cout << "Skew after correction: " << skew_deg_after_deskew << "° (" << skew_rad_after_deskew << " rad)" << std::endl;

    std::string edge_corrected_filename = "../output/" + base_filename + "-corrected-edges.png";
    edge_detected_img_deskewed.Save(edge_corrected_filename);
}

int main()
{
    fs::create_directories("../output");

    std::string input_dir = "../input";

    std::cout << "\nProcessing all images in " << input_dir << "..." << std::endl;
    int processed_count = 0;

    for (const auto &entry : fs::directory_iterator(input_dir))
    {
        if (!entry.is_regular_file())
            continue;

        auto ext = entry.path().extension().string();
        if (ext == ".png")
        {
            std::string filename = entry.path().filename().string();
            process_image(entry.path().string(), filename);
            processed_count++;
        }
    }

    std::cout << "\nSuccessfuly processed " << processed_count << " images.\n"
              << std::endl;

    return 0;
}