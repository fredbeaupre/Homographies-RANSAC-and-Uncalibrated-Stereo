%% ASSIGNMENT 3

% Reading in the pictures
image1 = im2double(rgb2gray(imread('original.jpg')));
image2 = im2double(rgb2gray(imread('rotated.jpg')));


% Detecting SURF features in the images
surf_points1 = detectSURFFeatures(image1);
surf_points2 = detectSURFFeatures(image2);
% extracting the features
[features1, valid_pts1] = extractFeatures(image1, surf_points1);
[features2, valid_pts2] = extractFeatures(image2, surf_points2);
% find matching pairs
pair_indices = matchFeatures(features1, features2, 'Unique', true);
matches1 = valid_pts1(pair_indices(:,1));
matches2 = valid_pts2(pair_indices(:,2));
image1_points = matches1.Location;
image2_points = matches2.Location;

%showMatchedFeatures(image1, image2, matches1, matches2);


% Estimating Homography (RANSAC)
num_pairs = size(pair_indices, 1);
H = zeros(3,3); % homography with largest C
C_size  = 0; % consensus set size
C = []; % consensus_set
dist_thresh = 1;
for i = 1:100
    A = [];
    temp_H = zeros(3,3);
    temp_C = [];
    temp_size = 0;
    % choosing four random pairs
    random4 = randi([1, num_pairs], 4, 1);
    points1 = image1_points(random4, :);
    points2 = image2_points(random4, :);
    % Building A matrix (which will be 8x9)
    for j =1:4
        x = points1(j,1);
        y = points1(j,2);
        xp = points2(j, 1);
        yp = points2(j, 2);
        A = [A; x y 1 0 0 0 (-xp*x) (-xp*y) (-xp)];
        A = [A; 0 0 0 x y 1 (-yp*x) (-yp*y) (-yp)];
    end
    % exact homography for our 4 randomly sampled pairs
    [U, S, V] = svd(A);
    temp_H = double(vpa(V(:,9)));
    temp_H = reshape(temp_H, [3 3])';
    
    % computing consensus set for the obtained homography
    for k = 1:length(image1_points)
        point_left = [image1_points(k, :) 1];
        point_right = [image2_points(k, :) 1];
        projected_point = temp_H * transpose(point_left);
        % don't forget to divide by the w factor we get in the third vector
        % entry
        projected_point = projected_point/projected_point(3); 
        distance = norm(projected_point - point_right');
        if distance < dist_thresh
            temp_size = temp_size + 1;
            temp_C = [temp_C; point_left point_right];
        end
    end
    % checking if homography is best as of yet
    if temp_size > C_size
        C_size = temp_size;
        H = temp_H;
        C = temp_C;
    end
end
% Coming out of this loop we have our best fit homography and its
% corresponding consensus set
% We now normalize the data in the consensus set and then fit the
% homography using the normalized data

% means and standard devs
mean_x = mean(C(:,1));
mean_y = mean(C(:,2));
mean_xp = mean(C(:,4));
mean_yp = mean(C(:,5));
std1 = sum( (C(:,1) - mean_x ).^2 + (C(:,2) - mean_y).^2) / (2*C_size);
std1 = sqrt(std1);
std2 = sum( (C(:,4) - mean_xp ).^2 + (C(:,5) - mean_yp).^2) / (2*C_size);
std2 = sqrt(std2);

C(:, 1) = (C(:,1) - mean_x)/std1;
C(:, 2) = (C(:,2) - mean_y)/std1;
C(:, 4) = (C(:,4) - mean_xp)/std2;
C(:, 5) = (C(:, 5) - mean_yp)/std2;

% recompute the homography using consensus set to fit
A = [];
for j =1:C_size
    x = C(j,1);
    y = C(j,2);
    xp = C(j, 4);
    yp = C(j, 5);
    A = [A; x y 1 0 0 0 (-xp*x) (-xp*y) (-xp)];
    A = [A; 0 0 0 x y 1 (-yp*x) (-yp*y) (-yp)];
end

[U, S, V] = svd(A);
H = double(vpa(V(:, end)));
H = reshape(H, [3 3])';
%%
%IMAGE STITCHING
% Ready to do image stitching with our computed homography

rgb_image1 = im2double(imread('original.jpg'));
rgb_image2 = im2double(imread('rotated.jpg'));

red_channel1 = rgb_image1(:, :, 1);
green_channel2 = rgb_image2(:, :, 2);
blue_channel2 = rgb_image2(:, :, 3);

allBlack1 = zeros(size(rgb_image1, 1), size(rgb_image1, 2));
allBlack2 = zeros(size(rgb_image2, 1), size(rgb_image2, 2));

red_image1 = cat(3, red_channel1, allBlack1, allBlack1);
greenblue_image2 = cat(3, allBlack2, green_channel2, blue_channel2);



[~, x_data, y_data] = imtransform(red_image1,maketform('projective',H'));

x_data_out=[min(1,x_data(1)) max(size(greenblue_image2,2), x_data(2))];
y_data_out=[min(1,y_data(1)) max(size(greenblue_image2,1), y_data(2))];

result1 = imtransform(red_image1, maketform('projective',H'),...
    'XData',x_data_out,'YData',y_data_out);
result2 = imtransform(greenblue_image2, maketform('affine',eye(3)),...
    'XData',x_data_out,'YData',y_data_out);
result = result1 + result2;

figure;
imshow(result);














