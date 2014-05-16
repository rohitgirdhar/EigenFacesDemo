% Eigen Faces
% Reference : http://www.cs.ucsb.edu/~mturk/Papers/jcn.pdf
% Compatible with Matlab 2013b
%% Read images
dpath = 'yale_subset_719/';
rows = 192; cols = 168;

img_files = dir(fullfile(dpath, '*.bmp'));
nimages = length(img_files);
imgs = zeros([rows * cols nimages], 'uint8');
i = 1;
for img_file = img_files'
    imgs(:, i) = reshape(imread(fullfile(dpath, img_file.name)), 1, []);
    i = i + 1;
end
f = figure;
set(f, 'Name', 'Some of the images read');
for i = 1 : 6
    subplot(3, 3, i); imshow(reshape(imgs(:, i), rows, []));
end
waitforbuttonpress;

%% Compute average, diff face
avg_face = uint8(sum(imgs, 2) / nimages); % = psi
set(f, 'Name', 'Average Face');
subplot(1,1,1); imshow(reshape(avg_face, rows, []));
waitforbuttonpress;

A = zeros([rows * cols nimages], 'uint8');
for i = 1 : nimages
    A(:, i) = imgs(:, i) - avg_face;
end
set(f, 'Name', 'Difference faces');
for i = 1 : 6
    subplot(3, 3, i); imshow(reshape(A(:, i), rows, []));
end
waitforbuttonpress;

%% Compute Covariance matrix and Eigen vectors
% Need the eig vectors for A*A^t, but going to compute for A^T * A
% (computational reasons)
L = double(A') * double(A);
[V, D] = eig(L);
% sort in descending order
[D, order] = sort(diag(D),'descend');
V = V(:,order);
% compute for C
U = double(A) * V; % now, V is the eig vectors of C = A*A^T
% normalize the eigen vectors
U = normc(U);
% Display the best 6 eig vectors as images
set(f, 'Name', 'Eigen Faces');
for i = 1 : 6
    subplot(3, 3, i); imshow(reshape(U(:, i), rows, []), []);
end
waitforbuttonpress;

%% Projecting a new/existing image
% Let's try to reproject some of the training image itself
try_reconst = 50;
n_eig = 7; % num of eig vectors to use
omegas = U(:, 1:n_eig)' * double(imgs(:, try_reconst) - avg_face); % weights of each eigen vector
res = U(:, 1:n_eig) * omegas + double(avg_face);
set(f, 'Name', 'Reconstruction using 7 eigen faces');
subplot(1,2,1); imshow(reshape(imgs(:,try_reconst), rows, []));
subplot(1,2,2); imshow(reshape(res, rows, []), []);
