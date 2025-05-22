#include <iostream>
#include <array>
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

int main()
{
    GrayscaleImage input;
    input.Load("../input/skew-origin.png");
    GrayscaleImage output = sobel_filter(input);

    output.Save("../output/edge-detected-document.png");
    return 0;
}