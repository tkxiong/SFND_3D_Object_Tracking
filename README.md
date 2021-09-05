# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## FP.1 Match 3D Objects

Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

```
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    multimap<int,int> boxmap;
    for (auto &match : matches)
    {
        cv::KeyPoint prevPoints = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currPoints = currFrame.keypoints[match.trainIdx];

        int prevBoxId = -1;
        int currBoxId = -1;

        for(auto &box : prevFrame.boundingBoxes)
        {
            if (box.roi.contains(prevPoints.pt))
            {
                prevBoxId = box.boxID;
            }
        }
        for(auto &box : currFrame.boundingBoxes)
        {
            if (box.roi.contains(currPoints.pt))
            {
                currBoxId = box.boxID;
            }
        }
        // generate currBoxId-prevBoxId map pair
        boxmap.insert({currBoxId, prevBoxId});
    }

    int prevBoxSize = prevFrame.boundingBoxes.size();

    // find the best matched previous boundingbox for each current boundingbox
    for(int i = 0; i < prevBoxSize; ++i)
    {
        auto boxmapPair = boxmap.equal_range(i);
        vector<int> currBoxCount(prevBoxSize, 0);
        for (auto pr = boxmapPair.first; pr != boxmapPair.second; ++pr)
        {
            if(pr->second >= 0)
            {
                currBoxCount[pr->second] += 1;
            }
        }
        // find the position of best prev box which has highest number of keypoint correspondences.
        int maxPosition = std::distance(currBoxCount.begin(), std::max_element(currBoxCount.begin(), currBoxCount.end()));
        bbBestMatches.insert({maxPosition, i});
        cout << "Current BoxID: " << i <<" match Previous BoxID: " << maxPosition << endl;
    }
}
```

## FP.2 Compute Lidar-based TTC

Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

```
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1 / frameRate;
    double laneWidth = 4.0;
    vector<double> lidarPointsCurrX, lidarPointsPrevX;

    for (auto &pts : lidarPointsPrev)
    {
        if (abs(pts.y) <= laneWidth / 2.0)
        {
            lidarPointsPrevX.push_back(pts.x);
        }
    }

    for (auto &pts : lidarPointsCurr)
    {
        if (abs(pts.y) <= laneWidth / 2.0)
        {
            lidarPointsCurrX.push_back(pts.x);
        }
    }

    // calculate median value
    sort(lidarPointsCurrX.begin(), lidarPointsCurrX.end());
    sort(lidarPointsPrevX.begin(), lidarPointsPrevX.end());
    int lidarPtCurrSize = lidarPointsCurrX.size();
    int lidarPtPrevSize = lidarPointsPrevX.size();

    double d1 = lidarPtCurrSize % 2 == 0 ? (lidarPointsCurrX[lidarPtCurrSize / 2 - 1] + lidarPointsCurrX[lidarPtCurrSize / 2]) / 2
                                            : lidarPointsCurrX[lidarPtCurrSize / 2];
    double d0 = lidarPtPrevSize % 2 == 0 ? (lidarPointsPrevX[lidarPtPrevSize / 2 - 1] + lidarPointsPrevX[lidarPtPrevSize / 2]) / 2
                                            : lidarPointsPrevX[lidarPtPrevSize / 2];
    TTC = d1 * dT / (d0 - d1);
}
```

## FP.3 Associate Keypoint Correspondences with Bounding Boxes

Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

```
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::map<vector<cv::DMatch>::iterator, double> theMap;
    vector<double> euclideanDistances;
    for (auto it = kptMatches.begin(); it != kptMatches.end(); it++)
    {
        cv::KeyPoint currPoints = kptsCurr[it->trainIdx];
        cv::KeyPoint prevPoints = kptsPrev[it->queryIdx];
        if (boundingBox.roi.contains(currPoints.pt))
        {
            theMap[it] = cv::norm(currPoints.pt - prevPoints.pt);
            euclideanDistances.push_back(cv::norm(currPoints.pt - prevPoints.pt));
            //cout << cv::norm(currPoints.pt - prevPoints.pt) << endl;
        }
    }

    if (euclideanDistances.size() < 0)
    {
        return;
    }

    double mean = 0;
    for (auto val : euclideanDistances)
    {
        mean += val;
    }
    mean = mean / euclideanDistances.size();
    double stdDev = calcStddev(mean, euclideanDistances);
    //cout << "mean : " << mean << " stdDev : " << stdDev << endl;

    for (auto const &pair : theMap)
    {
        if ((pair.second - mean) < stdDev)
        {
            boundingBox.kptMatches.push_back(*pair.first);
        }
    }
}
```

## FP.4 Compute Camera-based TTC

Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

```
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1.0 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}
```

## FP.5 Performance Evaluation 1

Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

* Looking through frame 0 - 18 of top view LIDAR points cloud, all the data appear good.
* Using median value is useful to reject outliers.

![pc_0](images/others/pc_0.png)
![result_0](images/others/result_0.png)

![pc_1](images/others/pc_1.png)
![result_1](images/others/result_1.png)

## FP.6 Performance Evaluation 2

Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

Based on the result from the table below, the TOP 3 detector/descriptor combinations:
* FAST + SIFT
* SHITOMASI + BRIEF
* BRISK + AKAZE

Factors that will affect the Camera TTC results:
1. YOLOv3 bounding box detection, if the bounding box area is bigger, it will extract more feature
2. HARRIS detector produce unreliable results

| Detector | Descriptor | Frame compare | LIDAR TTC | Camera TTC | TTC Difference |
| ---      | ---        | ---           | ---       | ---        | ---            |
| SHITOMASI | BRISK | 1 - 2 | 12.4156 | 13.7165 | -1.30089 |
| SHITOMASI | BRIEF | 1 - 2 | 12.4156 | 12.7726 | -0.357031 |
| SHITOMASI | ORB | 1 - 2 | 12.4156 | 13.3899 | -0.974339 |
| SHITOMASI | FREAK | 1 - 2 | 12.4156 | 13.3899 | -0.974339 |
| SHITOMASI | AKAZE | 1 - 2 | 12.4156 | x | x |
| SHITOMASI | SIFT | 1 - 2 | 12.4156 | 13.1925 | -0.776853 |
| HARRIS | BRISK | 1 - 2 | 12.4156 | nan | nan |
| HARRIS | BRIEF | 1 - 2 | 12.4156 | nan | nan |
| HARRIS | ORB | 1 - 2 | 12.4156 | nan | nan |
| HARRIS | FREAK | 1 - 2 | 12.4156 | nan | nan |
| HARRIS | AKAZE | 1 - 2 | 12.4156 | 80.7525 | -68.3369 |
| HARRIS | SIFT | 1 - 2 | 12.4156 | 80.7525 | -68.3369 |
| FAST | BRISK | 1 - 2 | 12.4156 | 11.5837 | 0.831864 |
| FAST | BRIEF | 1 - 2 | 12.4156 | 11.6641 | 0.751453 |
| FAST | ORB | 1 - 2 | 12.4156 | 11.0816 | 1.33404 |
| FAST | FREAK | 1 - 2 | 12.4156 | 14.338 | -1.92244 |
| FAST | AKAZE | 1 - 2 | 12.4156 | 11.0639 | 1.35165 |
| FAST | SIFT | 1 - 2 | 12.4156 | 12.0866 | 0.328996 |
| BRISK | BRISK | 1 - 2 | 12.4156 | 23.397 | -10.9814 |
| BRISK | BRIEF | 1 - 2 | 12.4156 | 24.3684 | -11.9528 |
| BRISK | ORB | 1 - 2 | 12.4156 | 17.7177 | -5.30207 |
| BRISK | FREAK | 1 - 2 | 12.4156 | 24.8826 | -12.467 |
| BRISK | AKAZE | 1 - 2 | 12.4156 | 11.8352 | 0.580431 |
| BRISK | SIFT | 1 - 2 | 12.4156 | 16.2788 | -3.8632 |
| ORB | BRISK | 1 - 2 | 12.4156 | 10.6151 | 1.80047 |
| ORB | BRIEF | 1 - 2 | 12.4156 | -82.8256 | 95.2412 |
| ORB | ORB | 1 - 2 | 12.4156 | 10.1192 | 2.29645 |
| ORB | FREAK | 1 - 2 | 12.4156 | 10.7758 | 1.63981 |
| ORB | AKAZE | 1 - 2 | 12.4156 | 10.8795 | 1.53609 |
| ORB | SIFT | 1 - 2 | 12.4156 | 9.20431 | 3.21129 |
| AKAZE | BRISK | 1 - 2 | 12.4156 | 17.816 | -5.40043 |
| AKAZE | BRIEF | 1 - 2 | 12.4156 | 16.3722 | -3.95663 |
| AKAZE | ORB | 1 - 2 | 12.4156 | 17.2188 | -4.80319 |
| AKAZE | FREAK | 1 - 2 | 12.4156 | 15.4471 | -3.03146 |
| AKAZE | AKAZE | 1 - 2 | 12.4156 | 15.722 | -3.30636 |
| AKAZE | SIFT | 1 - 2 | 12.4156 | 17.5661 | -5.15046 |
| SIFT | BRISK | 1 - 2 | 12.4156 | 13.7393 | -1.32368 |
| SIFT | BRIEF | 1 - 2 | 12.4156 | 14.4221 | -2.00649 |
| SIFT | ORB | 1 - 2 | 12.4156 | x | x |
| SIFT | FREAK | 1 - 2 | 12.4156 | 14.219 | -1.80339 |
| SIFT | AKAZE | 1 - 2 | 12.4156 | 10.1809 | 2.23467 |
| SIFT | SIFT | 1 - 2 | 12.4156 | 13.43 | -1.01439 |