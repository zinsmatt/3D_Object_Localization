clear all 
close all
clc


% data_folder_maskrcnn = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk/MaskRCNN_detections/";
% images_folder = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk/rgb/";
% sample_factor = 50;

 
% detection_associations = "/home/mzins/dev/3D-Object-Localisation/ellipses_association/MaskRCNN/assoc_maskrcnn.txt"
detection_associations = "/home/mzins/dev/3D-Object-Localisation/ellipses_association/Yolo_c/assoc_yolo_c.txt"
[pathstr, name, ext] = fileparts(detection_associations)
images_to_use = strcat(pathstr, "/", name, ".used_images.txt")

% images_to_use = "/home/mzins/dev/3D-Object-Localisation/images_to_use.txt";
% detected_ellipses = "/home/mzins/dev/3D-Object-Localisation/C.txt";
% to_use_detections = "/home/mzins/dev/3D-Object-Localisation/to_use.txt";
dataset_folder = "/home/mzins/Dataset/rgbd_dataset_freiburg2_desk/";
gt_poses_file = strcat(dataset_folder, "groundtruth.txt");
rgb_images_folder = strcat(dataset_folder, "rgb/");
% ouput_folder = "/home/mzins/dev/3D-Object-Localisation/reconstructions/";


fid = fopen(images_to_use);
files = textscan(fid,'%s');
files = files{1};
fclose(fid);


ellipses = table2array(readtable(detection_associations));

poses = table2array(readtable(gt_poses_file, 'HeaderLines', 1, 'CommentStyle', '#'));

timestamps_poses = poses(:, 1);
positions = poses(:, 2:4);
orientations = poses(:, 5:end);

images = {};
world_to_camera_matrices = [];
images_names = {};

for  i = 1:length(files)
    [pathstr, name, ext] = fileparts(char(files(i)));
    images_names{i} = name;
    name_parts = strsplit(name,'.');
    name = strcat(name_parts{1}, "." + name_parts{2});
    timestamp = str2double(name);
    
    d_to_timestamps = abs(timestamps_poses - timestamp);
    [dmin, index] = min(d_to_timestamps);
    
    if dmin > 0.02
        disp("Warning unsure camera pose for : ", name);
    end
    
    images{i}.I = imread(strcat(rgb_images_folder, name, ".png"));
    Rc_w = SpinCalc('QtoDCM', orientations(index, :), 1, 0);
    t = positions(index, :)';
    Tc_w = -Rc_w * t;
    world_to_camera_matrices = [world_to_camera_matrices; [Rc_w, Tc_w]'];
end

 
n_objects = size(ellipses, 2) / 3;
n_images = size(ellipses, 1) / 3;

to_use = ones(n_images, n_objects);
for i=1:n_images
    for j=1:n_objects
        if sum(sum(abs(ellipses(3*i-2:3*i, 3*j-2:3*j)))) < 0.00001
            to_use(i, j) = 0
        end
    end
end
            

% to_use = to_use(:, 1);
% ellipses = ellipses(:, 1:3);

 
% Load intrinsics
K = [520.9, 0, 325.1; 0, 521.0, 249.7; 0, 0, 1];


Imm = images;
M = world_to_camera_matrices;
K = K;
frVw = to_use;
C = ellipses;

%cd /home/vgaudill/3D-Object-Localisation
addpath(genpath('./functions'));

% ~~~~~~ Demo data ~~~~~~~~~~~~~~~~~
% Sequence 7 of the TUW dataset
%load('data/seq7_TUW.mat');

% Data needed:
% bbx   [F x 4O]:  Matrix of the bounding boxes
% frVw  [F x  O]:  Matrix of the views which are considered for a given 
%                  viewpoint
% Imm   [F]     :  Struct of the images I
% K     [3 x 3] :  Camera calibration matrix
% GT    [O]     :  Struct of the GT dual quadrics
% M     [4F x 3]:  Matrix of the motion

%% ~~~~~~~~~~~~~~~~~~ Compute the centres ~~~~~~~~~~~~~~~~~~~~~~
% First step of computing the elliposoids (to reduce ill-posedness of quadrics)

% Compute the projective matrix given the motion matrix 
% (if you have already the projective matrix, skip this step)
P = (K*M')';

% Fit the ellipses to the bounding boxes
%C   = fromBBx2ell(bbx,frVw);

% Convert the GT from primal to dual
%GT = gtconv(GT);

% Compute the centers
x3c0 = [0 0 0 1]'*ones(1,size(frVw,2));

% Compute the reconstructions (dual quadrics)
Rec1 = generateQuadrics(P, x3c0, C, frVw, 'SFD' );

% Compute the reprojections of the reconstruction (dual ellipses)
Crec = reprQ(P,Rec1,frVw,'persp');

% Compute the reprojections of the Ground Truth dual ellipsoids
%Cgt  = reprQ(P,GT,frVw,'persp');

%% ~~~~~~~~~~~~~~~~~~ Compute the centres ~~~~~~~~~~~~~~~~~~~~~~
% Second step: Re-computing the ellipsoids given the centres

% Compute the centres given the first reconstruction
x3c1 = exCnt(Rec1);

% Compute the reconstructions (dual quadrics)
Rec2 = generateQuadrics(P, x3c1, C, frVw, 'SFD' );

% Compute the reprojections (dual ellipses)
all_objects = ones(size(frVw, 1), size(frVw, 2));
Crec = reprQ(P, Rec2, all_objects, 'persp');

% Plot the 2D ellipses
%plot2Dscene( Imm, images_names, C, Crec, ouput_folder, to_use);

% Plot the 3D scene given the dual ellipsoids
rec = plot3Dscene( Rec2, [], [0 1 0]);

ellipsoids = [];
for o = 1 : length(Rec2)
   if isfield(Rec2{o},'Q')
       ellipsoids = [ellipsoids; Rec2{o}.Q];
   end
end

writematrix(ellipsoids,'ellipsoids.txt','Delimiter',' ');