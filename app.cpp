#include <stdio.h>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

#define USE_ORB

void read_directory(const std::string& name, std::vector<std::string>& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        v.push_back(dp->d_name);
    }
    closedir(dirp);
}

static bool endsWith(const std::string& str, const std::string& suffix)
{
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

static bool startsWith(const std::string& str, const std::string& prefix)
{
    return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

int main(int argc, char *argv[]){

    std::string path = "C:\\Users\\SJ9957\\Documents\\Polarizer";

    std::vector<std::string> v;
    read_directory(path, v);

    // Template feature extraction
    std::string template_filename = "Keurig.jpg";
    
    Mat img_1 = imread(template_filename, IMREAD_GRAYSCALE);   // Read the file
    
    if( !img_1.data)
    { std::cout<< " --(!) Error reading template image " << std::endl; return -1; }
    
    #ifdef USE_ORB
        Ptr<ORB> detector = ORB::create(10000, 1.2, 8, 5);
    #else
        Ptr<SIFT> detector = SIFT::create();
    #endif

    std::vector<KeyPoint> keypoints_1;
    Mat descriptors_1;

    detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
    
    // Feature Matcher
    #ifdef USE_ORB
        FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
    #else
        FlannBasedMatcher matcher = cv::FlannBasedMatcher();
    #endif

    for(std::string p:v){
        if(endsWith(p, ".bmp")){
            std::string test_filename = path + "\\" + p;
            Mat img_2 = imread(test_filename, IMREAD_GRAYSCALE);   // Read the file

            std::vector<KeyPoint> keypoints_2;
            Mat descriptors_2;
            std::clock_t    start;
            start = std::clock();

            detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );

            std::vector< DMatch > matches;
            matcher.match( descriptors_1, descriptors_2, matches );
            
            std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
            
            double max_dist = 0; double min_dist = 100;
            //-- Quick calculation of max and min distances between keypoints
            for( int i = 0; i < descriptors_1.rows; i++ )
            { double dist = matches[i].distance;
                if( dist < min_dist ) min_dist = dist;
                if( dist > max_dist ) max_dist = dist;
            }
            printf("-- Max dist : %f \n", max_dist );
            printf("-- Min dist : %f \n", min_dist );
            //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
            //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
            //-- small)
            //-- PS.- radiusMatch can also be used here.
            std::vector< DMatch > good_matches;
            for( int i = 0; i < descriptors_1.rows; i++ )
            { if( matches[i].distance <= 1.5*min_dist)
                { good_matches.push_back( matches[i]); }
            }
            //-- Draw only "good" matches
            Mat img_matches;
            try{
                drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                            good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                            std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            }catch(...){
                imshow( "Good Matches", img_2 );
            }
            //-- Show detected matches
            imshow( "Good Matches", img_matches );
            for( int i = 0; i < (int)good_matches.size(); i++ )
            { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }

            waitKey(0);                                          // Wait for a keystroke in the window
        }
        else
            continue;
    }

    return 0;
}