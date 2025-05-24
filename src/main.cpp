#include <iostream>
#include <array>
#include <cstdint>
#include <numeric>
#include "image.h"

void sobel_filter(const GrayscaleImage &img,
                  std::vector<std::vector<double>> &magnitude,
                  std::vector<std::vector<double>> &direction)
{
    int width = img.GetWidth();
    int height = img.GetHeight();

    magnitude.resize(height, std::vector<double>(width, 0.0));
    direction.resize(height, std::vector<double>(width, 0.0));

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

            magnitude[y][x] = std::sqrt(gx * gx + gy * gy);
            direction[y][x] = std::atan2(gy, gx);
        }
    }
}

GrayscaleImage non_maximum_suppression(const std::vector<std::vector<double>> &magnitude,
                                       const std::vector<std::vector<double>> &direction)
{
    int height = magnitude.size();
    int width = magnitude[0].size();
    GrayscaleImage result(width, height);

    for (int y = 1; y < height - 1; y++)
    {
        for (int x = 1; x < width - 1; x++)
        {
            double angle = direction[y][x];
            double mag = magnitude[y][x];

            double angle_deg = angle * 180.0 / M_PI;
            if (angle_deg < 0)
                angle_deg += 180.0;

            double neighbor1 = 0, neighbor2 = 0;

            if ((angle_deg >= 0 && angle_deg < 22.5) || (angle_deg >= 157.5 && angle_deg <= 180))
            {
                neighbor1 = magnitude[y][x + 1];
                neighbor2 = magnitude[y][x - 1];
            }
            else if (angle_deg >= 22.5 && angle_deg < 67.5)
            {
                neighbor1 = magnitude[y + 1][x - 1];
                neighbor2 = magnitude[y - 1][x + 1];
            }
            else if (angle_deg >= 67.5 && angle_deg < 112.5)
            {
                neighbor1 = magnitude[y + 1][x];
                neighbor2 = magnitude[y - 1][x];
            }
            else if (angle_deg >= 112.5 && angle_deg < 157.5)
            {
                neighbor1 = magnitude[y + 1][x + 1];
                neighbor2 = magnitude[y - 1][x - 1];
            }

            if (mag >= neighbor1 && mag >= neighbor2)
            {
                result(x, y) = car(static_cast<int>(mag), 255);
            }
            else
            {
                result(x, y) = 0;
            }
        }
    }

    return result;
}

GrayscaleImage double_threshold_hysteresis(const GrayscaleImage &img,
                                           double low_threshold,
                                           double high_threshold)
{
    int width = img.GetWidth();
    int height = img.GetHeight();
    GrayscaleImage result(width, height);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uint8_t pixel = img(x, y);
            if (pixel >= high_threshold)
            {
                result(x, y) = 255;
            }
            else if (pixel >= low_threshold)
            {
                result(x, y) = 128;
            }
            else
            {
                result(x, y) = 0;
            }
        }
    }

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (int y = 1; y < height - 1; y++)
        {
            for (int x = 1; x < width - 1; x++)
            {
                if (result(x, y) == 128)
                {
                    bool connected_to_strong = false;
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        for (int dx = -1; dx <= 1; dx++)
                        {
                            if (result(x + dx, y + dy) == 255)
                            {
                                connected_to_strong = true;
                                break;
                            }
                        }
                        if (connected_to_strong)
                            break;
                    }

                    if (connected_to_strong)
                    {
                        result(x, y) = 255;
                        changed = true;
                    }
                }
            }
        }
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (result(x, y) == 128)
            {
                result(x, y) = 0;
            }
        }
    }

    return result;
}

GrayscaleImage gaussian_blur(const GrayscaleImage &img,
                             int k,
                             double std_deviation = 0.0)
{
    int width = img.GetWidth();
    int height = img.GetHeight();

    if (std_deviation <= 0.0)
        std_deviation = 0.3 * ((k - 1) * 0.5 - 1) + 0.8;

    int r = k / 2;
    std::vector<double> kernel(k);
    for (int i = -r; i <= r; ++i)
    {
        kernel[i + r] = std::exp(-(i * i) / (2 * std_deviation * std_deviation));
    }
    double sum = std::accumulate(kernel.begin(), kernel.end(), 0.0);
    for (double &value : kernel)
        value /= sum;

    GrayscaleImage temp(width, height);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double accumulator = 0.0;
            for (int i = -r; i <= r; i++)
            {
                int xx = x + i;
                xx = car(xx, width - 1);
                accumulator += kernel[i + r] * img(xx, y);
            }
            temp(x, y) = uint8_t(std::round(accumulator));
        }
    }

    GrayscaleImage blurred_img(width, height);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double accumulator = 0.0;
            for (int j = -r; j <= r; j++)
            {
                int yy = y + j;
                yy = car(yy, height - 1);
                accumulator += kernel[j + r] * temp(x, yy);
            }
            blurred_img(x, y) = uint8_t(std::round(accumulator));
        }
    }

    return blurred_img;
}

// canny filter (suppress and hysterasis functions) logic
// taken from:
// https://github.com/Nhat-Thanh/Canny-Algorithm/blob/main/canny_algorithm.cpp
GrayscaleImage canny_filter(const GrayscaleImage &img,
                            int gaussian_kernel_size = 5,
                            double sigma = 1.0,
                            double low_threshold = 50.0,
                            double high_threshold = 150.0)
{
    GrayscaleImage blurred = gaussian_blur(img, gaussian_kernel_size, sigma);

    std::vector<std::vector<double>> magnitude, direction;
    sobel_filter(blurred, magnitude, direction);

    GrayscaleImage suppressed = non_maximum_suppression(magnitude, direction);

    GrayscaleImage edges = double_threshold_hysteresis(suppressed, low_threshold, high_threshold);

    return edges;
}

int main()
{
    GrayscaleImage input;
    input.Load("../input/skew-origin.png");

    GrayscaleImage canny_edges = canny_filter(input, 5, 1.0, 50.0, 150.0);

    canny_edges.Save("../output/canny_edges.png");
    return 0;
}