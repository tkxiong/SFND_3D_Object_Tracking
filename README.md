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
    double meanDistance = 0.0;
    vector<cv::DMatch> insideMatches;
    for (auto &match : kptMatches)
    {
        cv::KeyPoint currPoints = kptsCurr[match.trainIdx];
        if (boundingBox.roi.contains(currPoints.pt))
        {
            insideMatches.push_back(match);
        }
    }
    // DMatch.distance - Distance between descriptors. The lower, the better similarity
    for (auto &insideMatch : insideMatches)
    {
        meanDistance += insideMatch.distance;
    }

    if (insideMatches.size() > 0)
    {
        meanDistance = meanDistance / insideMatches.size();
    }
    else
    {
        return;
    }

    for (auto insideMatch:insideMatches)
    {
        if (insideMatch.distance < meanDistance)
        {
            boundingBox.kptMatches.push_back(insideMatch);
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
* FAST + BRISK
* SHITOMASI + FREAK
* SHITOMASI + FAST

Factors that will affect the Camera TTC results:
1. YOLOv3 bounding box detection, if the bounding box area is bigger, it will extract more feature
2. HARRIS detector produce unreliable results

| Detector | Descriptor | Frame compare | LIDAR TTC | Camera TTC | TTC Difference |
| ---      | ---        | ---           | ---       | ---        | ---            |
| SHITOMASI | BRISK | 1 - 2 | 12.4156 | 12.9602 | -0.544608 |
| SHITOMASI | BRIEF | 1 - 2 | 12.4156 | 13.0092 | -0.593649 |
| SHITOMASI | ORB | 1 - 2 | 12.4156 | 13.3067 | -0.891117 |
| SHITOMASI | FREAK | 1 - 2 | 12.4156 | 12.3292 | 0.0864497 |
| SHITOMASI | AKAZE | 1 - 2 | x | x | x |
| SHITOMASI | SIFT | 1 - 2 | 12.4156 | 12.3292 | 0.0864497 |
| HARRIS | BRISK | 1 - 2 | 12.4156 | nan | nan |
| HARRIS | BRIEF | 1 - 2 | 12.4156 | 80.7525 | -68.3369 |
| HARRIS | ORB | 1 - 2 | 12.4156 | nan | nan |
| HARRIS | FREAK | 1 - 2 | 12.4156 | nan | nan |
| HARRIS | AKAZE | 1 - 2 | 12.4156 | 80.7525 | -68.3369 |
| HARRIS | SIFT | 1 - 2 | 12.4156 | 80.7525 | -68.3369 |
| FAST | BRISK | 1 - 2 | 12.4156 | 12.4822 | -0.0665892 |
| FAST | BRIEF | 1 - 2 | 12.4156 | 11.7261 | 0.689496 |
| FAST | ORB | 1 - 2 | 12.4156 | 11.3289 | 1.08675 |
| FAST | FREAK | 1 - 2 | 12.4156 | 14.108 | -1.69238 |
| FAST | AKAZE | 1 - 2 | 12.4156 | 11.7009 | 0.714749 |
| FAST | SIFT | 1 - 2 | 12.4156 | 11.7261 | 0.689496 |
| BRISK | BRISK | 1 - 2 | 12.4156 | 20.505 | -8.08943 |
| BRISK | BRIEF | 1 - 2 | 12.4156 | 17.0514 | -4.63578 |
| BRISK | ORB | 1 - 2 | 12.4156 | 17.6458 | -5.23024 |
| BRISK | FREAK | 1 - 2 | 12.4156 | 21.7186 | -9.30297 |
| BRISK | AKAZE | 1 - 2 | 12.4156 | 13.2426 | -0.827039 |
| BRISK | SIFT | 1 - 2 | 12.4156 | 17.1507 | -4.73514 |
| ORB | BRISK | 1 - 2 | 12.4156 | 10.9367 | 1.47893 |
| ORB | BRIEF | 1 - 2 | 12.4156 | 13.0997 | -0.684068 |
| ORB | FREAK | 1 - 2 | 12.4156 | 9.94887 | 2.46673 |
| ORB | ORB | 1 - 2 | 12.4156 | 10.6151 | 1.80047 |
| ORB | FREAK | 1 - 2 | 12.4156 | 9.94887 | 2.46673 |
| ORB | AKAZE | 1 - 2 | 12.4156 | 10.1226 | 2.29302 |
| ORB | SIFT | 1 - 2 | 12.4156 | 9.95376 | 2.46184 |
| AKAZE | BRISK | 1 - 2 | 12.4156 | 14.2866 | -1.87096 |
| AKAZE | BRIEF | 1 - 2 | 12.4156 | 14.29 | -1.87436 |
| AKAZE | ORB | 1 - 2 | 12.4156 | 14.3927 | -1.97713 |
| AKAZE | FREAK | 1 - 2 | 12.4156 | 12.3707 | 0.04486 |
| AKAZE | AKAZE | 1 - 2 | 12.4156 | 13.9228 | -1.50722 |
| AKAZE | SIFT | 1 - 2 | 12.4156 | 14.9833 | -2.56774 |
| SIFT | BRISK | 1 - 2 | 12.4156 | 14.8783 | -2.46271 |
| SIFT | BRIEF | 1 - 2 | 12.4156 | 15.8584 | -3.44285 |
| SIFT | ORB | 1 - 2 | 12.4156 | x | x |
| SIFT | FREAK | 1 - 2 | 12.4156 | 13.5379 | -1.12232 |
| SIFT | AKAZE | 1 - 2 | 12.4156 | 14.7038 | -2.28821 |
| SIFT | SIFT | 1 - 2 | 12.4156 | 14.2607 | -1.8451 |
