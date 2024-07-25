% Load the image
% Load the image
original_image = imread('peppers.png');

% Display original image
figure;
subplot(3, 3, 1);
image(original_image);
title('Original Image');

% Define transformation matrices for different transformations

% 1. Scaling
scaling_matrix = [2, 0, 0; 0, 2, 0; 0, 0, 1];

% 2. Rotation (30 degrees)
theta = 30; % angle in degrees
rotation_matrix = [cosd(theta), -sind(theta), 0; sind(theta), cosd(theta), 0; 0, 0, 1];

% 3. Similarity Transform (Combination of scaling and rotation)
similarity_matrix = scaling_matrix * rotation_matrix;

% 4. Affine Transform
% For demonstration, let's perform a shear transformation
shear_matrix = [1, 0.5, 0; 0, 1, 0; 0, 0, 1];

% 5. Translation
translation_matrix = [1, 0, 50; 0, 1, 50; 0, 0, 1]; % Translating by (50, 50)

% 6. Projective Transform
% Define your own projective transformation matrix
projective_matrix = [1, 0.5, 0; 0.2, 1, 0; 0.001, 0.001, 1];

% Apply transformations and display transformed images
transformations = {scaling_matrix, rotation_matrix, similarity_matrix, shear_matrix, translation_matrix, projective_matrix};
titles = {'Scaling', 'Rotation', 'Similarity', 'Affine', 'Translation', 'Projective'};

for i = 1:length(transformations)
    % Directly manipulate the transformation matrix to include translation
    if i == 5 % Translation
        transformation_matrix = transformations{i} * translation_matrix;
    else
        transformation_matrix = transformations{i};
    end
    
    transformed_image = imwarp(original_image, projective2d(transformation_matrix));
    subplot(3, 3, i+1);
    image(transformed_image);
    title(titles{i});
end

