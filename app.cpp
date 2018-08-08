#include <stdio.h>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
// #include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
// using namespace cv::xfeatures2d;

#define USE_ORB

void read_directory(const std::string &name, std::vector<std::string> &v)
{
    DIR *dirp = opendir(name.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL)
    {
        v.push_back(dp->d_name);
    }
    closedir(dirp);
}

static bool endsWith(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static bool startsWith(const std::string &str, const std::string &prefix)
{
    return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

typedef std::vector<std::string> CommandLineStringArgs;

void findParams(std::vector<cv::Point2f> &obj, std::vector<cv::Point2f> &scene, float *mat_object) {
    float a1, a2, a3, a4;
    int n = obj.size();
    float x_avg = 0, y_avg = 0, x_d_avg = 0, y_d_avg = 0, x_t_x = 0, x_t_x_d = 0, x_t_y_d = 0, y_t_y = 0, y_t_y_d = 0, y_t_x_d = 0;
    for(int i=0; i < n; i++){
        float x = obj[i].x, y = obj[i].y, x_d = scene[i].x, y_d = scene[i].y;
        x_avg += x;
        y_avg += y;
        x_d_avg += x_d;
        y_d_avg += y_d;
        x_t_x += x*x;
        y_t_y += y*y;
        x_t_x_d += x*x_d;
        x_t_y_d += x*y_d;
        y_t_x_d += y*x_d;
        y_t_y_d += y*y_d;
    } 
    x_avg /= n;
    x_d_avg /= n;
    y_avg /= n;
    y_d_avg /= n;
    a1 = ((x_t_x_d - n*x_avg*x_d_avg) + (y_t_y_d - n*y_avg*y_d_avg))/((x_t_x - n*x_avg*x_avg) + (y_t_y - n*y_avg*y_avg));
    a2 = ((x_t_y_d - n*x_avg*y_d_avg) - (y_t_x_d - n*y_avg*x_d_avg))/((x_t_x - n*x_avg*x_avg) + (y_t_y - n*y_avg*y_avg));
    a3 = x_d_avg - a1*x_avg + a2*y_avg;
    a4 = y_d_avg - a2*x_avg - a1*y_avg;
    mat_object[0] =  a1;
    mat_object[1] = -a2;
    mat_object[2] = a3;
    mat_object[3] = a2;
    mat_object[4] = a1;
    mat_object[5] = a4;
    mat_object[6] = 0;
    mat_object[7] = 0;
    mat_object[8] = 1;
}

int main(int argc, char **argv)
{

    if (argc != 3)
    {
        std::cout << "Not enough arguments. Required 2 provided " << argc - 1 << std::endl;
        return -1;
    }

    CommandLineStringArgs cmdlineStringArgs(&argv[0], &argv[0 + argc]);

    std::string path = cmdlineStringArgs[1];

    std::vector<std::string> v;
    read_directory(path, v);

    // Template feature extraction
    std::string template_filename = cmdlineStringArgs[2];

    Mat img_1 = imread(template_filename, IMREAD_GRAYSCALE); // Read the file

    // bitwise_not ( img_1, img_1 );
    
    if (!img_1.data)
    {
        std::cout << " --(!) Error reading template image " << std::endl;
        return -1;
    }

    // cv::resize(img_1, img_1, cv::Size(img_1.cols * 0.5,img_1.rows * 0.5), 0, 0, CV_INTER_LINEAR);

#ifdef USE_ORB
    Ptr<ORB> detector = ORB::create(10000, 1.2, 8, 5);
#else
    Ptr<SIFT> detector = SIFT::create();
#endif

    std::vector<KeyPoint> keypoints_1;
    Mat descriptors_1;

    detector->detectAndCompute(img_1, Mat(), keypoints_1, descriptors_1);

// Feature Matcher
#ifdef USE_ORB
    FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
#else
    FlannBasedMatcher matcher = cv::FlannBasedMatcher();
#endif

    for (std::string filename : v)
    {
        if (!endsWith(filename, ".bmp"))
            continue;

        std::string test_filename = path + "\\" + filename;
        Mat img_2 = imread(test_filename, IMREAD_GRAYSCALE); // Read the file

        // cv::resize(img_2, img_2, cv::Size(img_2.cols * 0.8,img_2.rows * 0.8), 0, 0, CV_INTER_LINEAR);

        std::vector<KeyPoint> keypoints_2;
        Mat descriptors_2;
        std::clock_t start;
        start = std::clock();

        detector->detectAndCompute(img_2, Mat(), keypoints_2, descriptors_2);

        std::vector<std::vector<DMatch>> matches;
        matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);

        std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

        std::vector<cv::DMatch> good_matches;
        for (int i = 0; i < matches.size(); ++i)
        {
            const float ratio = 0.75; // As in Lowe's paper; can be tuned
            if (matches[i].size() < 2)
                continue;
            if (abs(matches[i][0].distance) < ratio * abs(matches[i][1].distance))
            {
                good_matches.push_back(matches[i][0]);
            }
        }

        Mat img_matches;
        try
        {
            //-- Show detected matches
            // imshow("Good Matches", img_matches);
            // for( int i = 0; i < (int)good_matches.size(); i++ )
            // { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }
            if(good_matches.size() > 6) {
                //-- Localize the object
                std::vector<Point2f> obj, transfor_obj;
                std::vector<Point2f> scene;

                for( int i = 0; i < good_matches.size(); i++ )
                {
                    //-- Get the keypoints from the good matches
                    obj.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
                    scene.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
                }

                Mat H = findHomography( obj, scene, CV_RANSAC );

                std::cout << H << std::endl;

                cv::perspectiveTransform( obj, transfor_obj, H);

                int n=0;
                std::vector<cv::DMatch> best_matches;
                for(int i=0; i < obj.size(); i++){
                    float d = pow(scene[i].x - transfor_obj[i].x, 2) + pow(scene[i].y - transfor_obj[i].y, 2);
                    if(d < 9){
                        n++;
                        best_matches.push_back(good_matches[i]);
                    }
                }

                if(n > 6){
                    drawMatches(img_1, keypoints_1, img_2, keypoints_2,
                                best_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
                    obj.clear();
                    scene.clear();
                    for( int i = 0; i < best_matches.size(); i++ )
                    {
                        //-- Get the keypoints from the good matches
                        obj.push_back( keypoints_1[ best_matches[i].queryIdx ].pt );
                        scene.push_back( keypoints_2[ best_matches[i].trainIdx ].pt );
                        printf( "-- Good Match [%d] template x: %f template y: %f -- scene x: %f scene y: %f  \n", 
                                i, keypoints_1[ best_matches[i].queryIdx ].pt.x, keypoints_1[ best_matches[i].queryIdx ].pt.y, 
                                keypoints_2[ best_matches[i].trainIdx ].pt.x, keypoints_2[ best_matches[i].trainIdx ].pt.y);
                    }


                    float mat_object[9];
                    findParams( obj, scene, mat_object );

                    Mat H_1 = cv::Mat(3, 3, CV_32F, mat_object);

                    std::cout << H_1 << std::endl;
                    
                    //-- Get the corners from the image_1 ( the object to be "detected" )
                    std::vector<Point2f> obj_corners(4);
                    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_1.cols, 0 );
                    obj_corners[2] = cvPoint( img_1.cols, img_1.rows ); obj_corners[3] = cvPoint( 0, img_1.rows);
                    std::vector<Point2f> scene_corners(4);

                    cv::perspectiveTransform( obj_corners, scene_corners, H_1);

                    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
                    cv::line( img_matches, scene_corners[0] + Point2f( img_1.cols, 0), scene_corners[1] + Point2f( img_1.cols, 0), Scalar(0, 255, 0), 4 );
                    cv::line( img_matches, scene_corners[1] + Point2f( img_1.cols, 0), scene_corners[2] + Point2f( img_1.cols, 0), Scalar( 0, 255, 0), 4 );
                    cv::line( img_matches, scene_corners[2] + Point2f( img_1.cols, 0), scene_corners[3] + Point2f( img_1.cols, 0), Scalar( 0, 255, 0), 4 );
                    cv::line( img_matches, scene_corners[3] + Point2f( img_1.cols, 0), scene_corners[0] + Point2f( img_1.cols, 0), Scalar( 0, 255, 0), 4 );
                    cv::imshow( "Good Matches & Object detection", img_matches );
                }else{
                    cv::imshow("Good Matches & Object detection", img_2);
                }
            }else{
                cv::imshow("Good Matches & Object detection", img_2);
            }
            //-- Show detected matches

        }
        catch (...)
        {
            cv::imshow("Good Matches & Object detection", img_2);
        }

        char key = waitKey(0);
        if (key == 'b')
        {
            break;
        }
    }

    return 0;
}