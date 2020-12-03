%% QUESTION 2: Uncalibrated Stereo

% Read in the images
left_image = rgb2gray(imread('left_last.jpg'));
right_image = rgb2gray(imread('right_last.jpg'));

figure;
imshowpair(left_image, right_image,'montage');
title('Left image; Right Image');
%%

% detect features
blobs_left = detectSURFFeatures(left_image, 'MetricThreshold', 1000);
blobs_right = detectSURFFeatures(right_image, 'MetricThreshold', 1000);

%figure;
%imshow(left_image);
%hold on;
%plot(selectStrongest(blobs_left, 30));

%figure;
%imshow(right_image);
%hold on;
%plot(selectStrongest(blobs_right, 30));

% extract feature descriptors
[features_left, valid_blobs_left] = extractFeatures(left_image, blobs_left);
[features_right, valid_blobs_right] = extractFeatures(right_image, blobs_right);

% indices of the matching pairs in both images
pair_indices = matchFeatures(features_left, features_right, 'Metric', 'SAD', 'MatchThreshold', 5);

% matches pixel coordinates
matched_left = valid_blobs_left(pair_indices(:, 1), :);
matched_right = valid_blobs_right(pair_indices(:, 2), :);

% estimate of fundamental matrix
[FM, epipolarInliers, status] = estimateFundamentalMatrix(...
    matched_left, matched_right, 'Method', 'RANSAC',...
    'NumTrials', 1000, 'DistanceThreshold', 0.1, 'Confidence', 99.99);

if status ~= 0 || isEpipoleInImage(FM, size(left_image)) ...
  || isEpipoleInImage(FM', size(right_image))
  error(['Either not enough matching points were found or '...
         'the epipoles are inside the images. You may need to '...
         'inspect and improve the quality of detected features ',...
         'and/or improve the quality of your images.']);
end

% inliers in left and right image
inliers_left = matched_left(epipolarInliers, :);
inliers_right = matched_right(epipolarInliers, :);

%figure;
%showMatchedFeatures(left_image, right_image, inliers_left, inliers_right);
%legend('Inlier points in I1', 'Inlier points in I2');

% get epipoles of both images
[U, D, V] = svd(FM);
left_epipole = V(:,3);
left_epipole = left_epipole/left_epipole(3);
right_epipole = U(:,3);
right_epipole = right_epipole/right_epipole(3);

% location of inliers
left_points = inliers_left.Location;
right_points = inliers_right.Location;

% initialize variables we'll use in the plotting loop
random10 = randi([1 size(right_points,1)], 10, 1);
[m, n] = size(left_image);
colors = {'m', 'w', 'c', 'k','b','r','g','y',[.5 .6 .7],[.8 .2 .6]};

% plotting epipolar lines and corresponding matching points on the left
% image
figure;
imshow(left_image);
title(sprintf("Epipole in left image: (%.0f, %.0f)", left_epipole(1), left_epipole(2)));
hold on;
for i=1:10
    index = random10(i);
    left_x = left_points(index, 1); 
    left_y = left_points(index, 2);
    right_x = right_points(index, 1);
    right_y = right_points(index, 2);
    plot(left_x, left_y, 'color', colors{i}, 'marker','+', 'MarkerSize', 20); % plot left-image match in left image
    plot(right_x, right_y, 'color', colors{i}, 'marker', '+', 'MarkerSize', 20); % plot right-image match in left image
    hold on;
    % compute epipolar line for the matching points and plot on left image
    left_epipolar_x = 1:2*m;
    left_epipolar_y = left_y + (left_epipolar_x-left_x)*(left_epipole(2)-left_y)/(left_epipole(1)-left_x);
    plot(left_epipolar_x, left_epipolar_y, 'color', colors{i}, 'LineStyle', '--');
end

%% 
% repeating the plotting loop for the right image
figure;
imshow(right_image);
title(sprintf("Epipole in right image: (%.0f, %.0f)", right_epipole(1), right_epipole(2)));
hold on;
for i=1:10
    index = random10(i);
    left_x = left_points(index, 1);
    left_y = left_points(index, 2);
    right_x = right_points(index, 1);
    right_y = right_points(index, 2);
    plot(left_x, left_y, 'color', colors{i}, 'marker','o', 'MarkerSize', 20);
    plot(right_x, right_y, 'color', colors{i}, 'marker', 'o', 'MarkerSize', 20);
    hold on;
    hold on;
    right_epipolar_x = 1:2*m;
    right_epipolar_y = right_y + (right_epipolar_x-right_x)*(right_epipole(2)-right_y)/(right_epipole(1)-right_x);
    plot(right_epipolar_x, right_epipolar_y, 'color', colors{i}, 'LineStyle', '--');
end
%%
% rectify the images
[t1, t2] = estimateUncalibratedRectification(FM, ...
  left_points, right_points, size(right_image));


tform1 = projective2d(t1);
tform2 = projective2d(t2);

[left_rectified, right_rectified] = rectifyStereoImages(left_image, right_image, tform1, tform2);
figure;
imshow(stereoAnaglyph(left_rectified, right_rectified));
title("Rectified image");
%%
% compute and show the disparity map of the rectified image
disparityRange = [0 48];
disparityMap = disparitySGM(left_rectified, right_rectified,'DisparityRange',disparityRange,'UniquenessThreshold',20);
imshow(disparityMap, disparityRange);
title('Disparity Map');
colormap jet;
colorbar;



