#include <stdio.h>
#include <stdlib.h>
#include <iostream> // for standard I/O
#include <fstream>
#include <time.h>
#include "opencv2/opencv.hpp"
#include <chrono>

using namespace cv;
using namespace std;

#define SHOW


int main(int argc, char** argv)
{
    VideoCapture camera("./bourne.mp4");
    if(!camera.isOpened())  // check if we succeeded
        return -1;

    const string output_filename = "./output.avi";   // Form the new name with container
    int ex = static_cast<int>(CV_FOURCC('M','J','P','G'));
    Size S = Size( (int)camera.get(CV_CAP_PROP_FRAME_WIDTH), (int)camera.get(CV_CAP_PROP_FRAME_HEIGHT) );
    cout << "SIZE: " << S << endl;

    VideoWriter outputVideo;                                        // Open the output
    outputVideo.open(output_filename, ex, 25, S, true);

    if (!outputVideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << output_filename << endl;
        return -1;
    }
    auto start = chrono::high_resolution_clock::now();
    auto end = chrono::high_resolution_clock::now();
    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    int tot_ms = 0;
    int count = 0;
    const char *window_name = "filter";   // Name shown in the GUI window.

#ifdef SHOW
    namedWindow(window_name); // Resizable window, might not work on Windows.
    waitKey(1);
#endif

    int max_frames = 299;
    while (true) {
        Mat cameraFrame, displayframe;
        count++;

        if(count > max_frames) break;
        camera >> cameraFrame;

        Mat filterframe = Mat(cameraFrame.size(), CV_8UC3);
        Mat grayframe,edge_x,edge_y,edge,edge_inv;
        cvtColor(cameraFrame, grayframe, CV_BGR2GRAY);

        // do video filter on CPU using OpenCV
        start = chrono::high_resolution_clock::now();
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
        GaussianBlur(grayframe, grayframe, Size(3, 3), 0, 0);
        Scharr(grayframe, edge_x, CV_8U, 0, 1, 1, 0, BORDER_DEFAULT );
        Scharr(grayframe, edge_y, CV_8U, 1, 0, 1, 0, BORDER_DEFAULT );
        addWeighted( edge_x, 0.5, edge_y, 0.5, 0, edge );
        threshold(edge, edge, 80, 255, THRESH_BINARY_INV);
        end = chrono::high_resolution_clock::now();

        cvtColor(edge, edge_inv, CV_GRAY2BGR);
        // Clear the output image to black, so that the cartoon line drawings will be black (ie: not drawn).
        memset((char *)displayframe.data, 0, displayframe.step * displayframe.rows);
        grayframe.copyTo(displayframe,edge);
        cvtColor(displayframe, displayframe, CV_GRAY2BGR);
        outputVideo << displayframe;

#ifdef SHOW
        imshow(window_name, displayframe);
        waitKey(1);
#endif
        
        diff = chrono::duration_cast<chrono::milliseconds>(end - start);
        tot_ms += diff.count();
    }
    outputVideo.release();
    camera.release();
    printf("FPS %.2lf .\n", (1000.0f * max_frames)/tot_ms);

    return EXIT_SUCCESS;
}

