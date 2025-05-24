#include <iostream>
#include <array>
#include <cstdint>
#include <numeric>
#include "image.h"

GrayscaleImage sobel_filter(const GrayscaleImage &img)
{
    int width = img.GetWidth();
    int height = img.GetHeight();

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
            edge_detected_img(x, y) = car(gx + gy, 255);
        }
    }

    return edge_detected_img;
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

int main()
{
    GrayscaleImage input;
    input.Load("../input/skew-origin.png");

    GrayscaleImage output = gaussian_blur(input, 5);

    output.Save("../output/blur.png");
    return 0;
}