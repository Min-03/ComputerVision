img = imread("peppers.png");
sz = size(img);
% disp(sz); %384 * 512 pixel size
% image(img);

%scaling
scaleT = projective2d([3 0 0; 0 3 0; 0 0 1]');
img_scale = imwarp(img, scaleT);
% image(img_scale);

%rotation
rotateT = projective2d([cos(pi / 4) -sin(pi / 4) 0; sin(pi / 4) cos(pi / 4) 0; 0 0 1]');
img_rotate = imwarp(img, rotateT);
% image(img_rotate);

%Similarity transform
simT = projective2d([3 * cos(pi / 4) -3 * sin(pi / 4) 30; 3 * sin(pi / 4) 3 * cos(pi / 4) 0; 0 0 1]');
img_sim = imwarp(img, simT);
image(img_sim);

%Affine transform

%own param