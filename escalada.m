%% Neteja
clear
clearvars,
close all,
clc,

%% 1a idea --> mirar el camí que segueix un escalador
%1. Fer foto a paret sola
%2. yolo v5
% mirar amb detectors (harris, surf) les preses i treure el moviment

%% Prova substracció de fons
se = strel("line", 7, 7);%per feropenings i closings
I = imread('paret_cut.jpg'); figure, imshow(I), title("Imatge 1"); %primer frame del video
I=double(I)./255;
I2=medfilt2(rgb2gray(I),[50 50]); %filtre de mediana per treure objectes petits --> preses
figure, imshow(I2), title("Imatge 2"); %imatge paret sola
preses = abs(rgb2gray(I)-I2);
preses = preses>0.14;

figure, imshow(preses), title("Imatge 3"); %imatge preses sense filtrar
preses_grans = bwareafilt(imclose(preses,se),[200 450]) - bwareafilt(imclose(preses,se),[300 302]) - bwareafilt(imclose(preses,se),[410 415]);  figure, imshow(preses_grans), title('Preses Grans') %preses grans sense 2 que estan molt juntes
%preses_filtre = bwareafilt(imclose(preses,se),[10 500]);  figure, imshow(preses_filtre), title('Preses ')
preses_petites = bwareafilt(imclose(preses,se),[12 30]);  figure, imshow(preses_petites), title('Preses petites')
preses_juntes = bwareafilt(preses,[600 800]) - bwareafilt(preses,[730 750]);  figure, imshow(preses_juntes), title('Preses juntes')
preses_totals = preses_grans + preses_petites + preses_juntes;  figure, imshow(preses_totals), title('Preses totals')
%figure, imshow(preses_totals(50:1160, 10:470)) %retallar imatge per eliminar soroll
%%
% [Gx,Gy] = imgradient(rgb2gray(I));
% Result = conv2(I,Gx,'valid');
% Result = abs(Result);
% minV = min(Result(:));
% maxV = max(Result(:));
% FinalResult = (Result-minV)*255/(maxV-minV);
% BW = im2bw(FinalResult, 0.8);
% imshow(BW);

%histograma de areas
%centroide persona --> quitar manchas pequeñas --> convex hull --> poligono
%(bounding box) --> mirar que presas estan dentro de la box donde persona
%boundix box --> esquinas son ls presas que coge
%que cierra el objeto

%mirar velocitat; medir alçada carla; mida/tamany pixel; 

%fer caixa fent la resta del frame amb només preses - frame del video
%% cercles peces petites

L = labelmatrix(preses_petites); % Not using bwconncomps() for older version users
stats = regionprops(CC,'Extent','Area');
% Find the ones that are like a circle
minExtent = 0.75;
keepMask = [stats.Extent]>minExtent;
% Extract the image of circles only and display
BWcircles = ismember(L, find(keepMask));
BWnonCircles = BW & ~BWcircles;
% Show the circles
figure, imshow(BWcircles)

%% llegir frames i pasarles a binari per tenir com cotxes
files_input = dir("frames_cut/*.jpg");
mkdir img_binary_cut;
ground = zeros([976 440]);
for i = 1:300
    im = rgb2gray(imread(strcat("frames_cut/",files_input(i).name)));
    ground(:,:,i) = im;
    
    %bw = im2bw(im,0.35);
    %bw_def = imcomplement(bw);
    %imwrite(bw_def,strcat("img_binary_def\im",string(i),".jpg"))
    
end

%mediana de tots els frames en BW per fer filtre de sorolls
im_med = median(ground,3);
d_est = std(ground,0,3);

alpha = 0.83;
beta = 17;
seg_elaborada = uint8((d_est*alpha + beta) < abs(ground- im_med)) * 255; %segmentem de forma més elaborada amb paràmetres alpha i beta

%% video cosos; centroides; passar per codi unió frames.
video = VideoWriter("cosos_mov.mp4");
open(video);
mkdir centroides_def
mkdir img_bin
array_centroides = zeros(300,1);
for i=1:300
    seg = seg_elaborada(:,:,i);
    seg1 = uint8(bwareaopen(seg,60))*255; %eliminar soroll

    %Obrim el frame
    seg2 = imclose(seg1,strel('disk',1));
    seg3 =imopen(seg2,strel('disk',1));
    seg4 = bwareafilt(im2bw(imclose(imfill(seg3,26),strel('disk',11))),2);
    seg4 = bwareafilt(seg4,[500 100000]);  %filtrar arees petites primers frames
    %imwrite(seg4,sprintf("img_binary_seg\\im_seg%04d.png",i))
    %imwrite(seg4,strcat("img_binary_def\\im_seg",string(i),".png"))
    
    seg_4_res(:,:,i) = seg4;[x,y] = size(seg4);
    seg4_2 = 255 * repmat(uint8(seg4),1,1,3); %passo a rgb per marcar un punt en color
    imwrite(seg4_2,sprintf("img_bin\\im_seg%04d.png",i));
    if sum(sum(zeros(x,y) == seg4)) ~= x*y
        s  = regionprops(seg4, 'centroid');
        centroids = cat(1, s.Centroid);
        if sum(size(centroids) == [1,2]) == 2 %si només hi ha un cos
            array_centroides(i) = centroids(2); %guardem només coordnada y per fer velocitats
            %seg4(uint16(centroids(1)), uint16(centroids(2))) = 0
            %seg4_2(uint16(centroids(2)), uint16(centroids(1)),:) = [200,100,100];


%             for i=centroids(2)-2:centroids(2)+2
%                 for j=centroids(1)-2:centroids(1)+2
%                     seg4_2(uint16(i),uint16(j),:) = [200,100,100];
%                 end
%             end


            seg4_2(uint16(centroids(2)), uint16(centroids(1)),:) = [200,100,100];  
            seg4_2(uint16(centroids(2)+1), uint16(centroids(1)),:) = [200,100,100];
            seg4_2(uint16(centroids(2)), uint16(centroids(1)+1),:) = [200,100,100];
            seg4_2(uint16(centroids(2)-1), uint16(centroids(1)),:) = [200,100,100];
            seg4_2(uint16(centroids(2)), uint16(centroids(1)-1),:) = [200,100,100];
            seg4_2(uint16(centroids(2)-1), uint16(centroids(1)-1),:) = [200,100,100];
            seg4_2(uint16(centroids(2)+1), uint16(centroids(1)+1),:) = [200,100,100];
            seg4_2(uint16(centroids(2)+1), uint16(centroids(1)-1),:) = [200,100,100];
            seg4_2(uint16(centroids(2)-1), uint16(centroids(1)+1),:) = [200,100,100];


       else %voldrà dir que hi ha dos cossos
%             array_centroides(i) = centroids(4);
%             for i=centroids(3)-2:centroids(3)+2
%                 for j=centroids(1)-2:centroids(1)+2
%                     seg4_2(uint16(i),uint16(j),:) = [200,100,100];
%                 end
%             end
% 
%             for i=centroids(4)-2:centroids(4)+2
%                 for j=centroids(2)-2:centroids(2)+2
%                     seg4_2(uint16(i),uint16(j),:) = [200,100,100];
%                 end
%             end

            
            seg4_2(uint16(centroids(3)), uint16(centroids(1)),:) = [200,100,100];
            seg4_2(uint16(centroids(3)+1), uint16(centroids(1)),:) = [200,100,100];
            seg4_2(uint16(centroids(3)), uint16(centroids(1)+1),:) = [200,100,100];
            seg4_2(uint16(centroids(3)-1), uint16(centroids(1)),:) = [200,100,100];
            seg4_2(uint16(centroids(3)), uint16(centroids(1)-1),:) = [200,100,100];
            seg4_2(uint16(centroids(3)-1), uint16(centroids(1)-1),:) = [200,100,100];
            seg4_2(uint16(centroids(3)+1), uint16(centroids(1)+1),:) = [200,100,100];
            seg4_2(uint16(centroids(3)+1), uint16(centroids(1)-1),:) = [200,100,100];
            seg4_2(uint16(centroids(3)-1), uint16(centroids(1)+1),:) = [200,100,100];


            seg4_2(uint16(centroids(4)), uint16(centroids(2)),:) = [200,100,100];
            seg4_2(uint16(centroids(4)+1), uint16(centroids(2)),:) = [200,100,100];
            seg4_2(uint16(centroids(4)), uint16(centroids(2)+1),:) = [200,100,100];
            seg4_2(uint16(centroids(4)-1), uint16(centroids(2)),:) = [200,100,100];
            seg4_2(uint16(centroids(4)), uint16(centroids(2)-1),:) = [200,100,100];
            seg4_2(uint16(centroids(4)-1), uint16(centroids(2)-1),:) = [200,100,100];
            seg4_2(uint16(centroids(4)+1), uint16(centroids(2)+1),:) = [200,100,100];
            seg4_2(uint16(centroids(4)+1), uint16(centroids(2)-1),:) = [200,100,100];
            seg4_2(uint16(centroids(4)-1), uint16(centroids(2)+1),:) = [200,100,100];
           
        end
    end
    imwrite(seg4_2,sprintf("centroides_def\\im_centroide%04d.png",i));
    
    
    %[y, x] = find(seg4==1);    % y before x (array ordering)
    %plot(mean(x), mean(y), '*');    % x before y (image ordering)
    writeVideo(video,imread(sprintf("img_bin\\im_seg%04d.png",i)));
end 
  
        

disp('adios');
  
close(video);

%% dibuixar linia
mkdir punts
%array_punts = zeros([976 440]);
suma = imread(sprintf("img_bin/im_seg%04d.png",1));
for i=1:300 %ho fem fins la 300 perquè és frame de pujada final; punt més alt on arriba escaladora dreta.
    dif = imread(sprintf("img_bin/im_seg%04d.png",i)) - imread(sprintf("centroides_def/im_centroide%04d.png",i));
    suma = suma + dif;
    frame = suma + imread(sprintf("frames_cut/IMG_%04d.jpg",i));
    imwrite(frame,sprintf("punts\\im_punt%04d.png",i));
    
end
figure, imshow(suma + imread(sprintf("frames_cut/IMG_%04d.jpg",300))) %imatge rgb amb centroides al frame de pujada final

%% velocitat --> com fer mitjana
figure(),imshow(imread(sprintf("frames_cut/IMG_%04d.jpg",181)))
[x,y] = ginput(2);
scale = 1.58/(abs(y(1)-y(2))); % meter/pixel --> frame 181 de frames_cut
frameRate = 300/10; % frame/second
velocity = 5 * frameRate * scale; % pixel/frame * frame/second * meter/pixel
figure(),imshow(imread(sprintf("frames_cut/IMG_%04d.jpg",181)))
% sum_velocidad = 0.0; c = 0;
% [x,y] = ginput(2);
% for i = 2:300
%     scale = 1.58/(abs(y(1)-y(2))); % meter/pixel --> frame 181 de frames_cut
%     frameRate = 300/10; % frame/second
%     if array_centroides(i) ~= 0 | array_centroides(i-1) ~= 0
%         velocity = (abs(array_centroides(i-1) - array_centroides(i))) * frameRate * scale; % pixel/frame * frame/second * meter/pixel
%         c = c+ 1;
%     end
%     sum_velocidad = sum_velocidad + velocity;
% end
% %como calcular pixel/frame; las otras variables son constates
% avg_v = sum_velocidad/c;

    %% centroides marc blanc
    %part centroides --> NO TOCAR
    
video = VideoWriter("cosos_mov.mp4");
open(video);
mkdir centroides
mkdir img_binary_seg

for i=1:355
    seg = seg_elaborada(:,:,i);
    seg1 = uint8(bwareaopen(seg,60))*255; %eliminar soroll

    %Obrim el frame
    seg2 = imclose(seg1,strel('disk',1));
    seg3 =imopen(seg2,strel('disk',1));
    seg4 = bwareafilt(im2bw(imclose(imfill(seg3,26),strel('disk',11))),2);
    seg4 = bwareafilt(seg4,[800 10000]);  %filtrar arees petites primers frames
    imwrite(seg4,sprintf("img_binary_seg\\im_seg%04d.png",i))
    %imwrite(seg4,strcat("img_binary_def\\im_seg",string(i),".png"))
    
    seg_4_res(:,:,i) = seg4;[x,y] = size(seg4);
    if sum(sum(zeros(x,y) == seg4)) ~= x*y
        s  = regionprops(seg4, 'centroid');
        centroids = cat(1, s.Centroid);
        imshow(seg4);
        hold on
        plot(centroids(:,1), centroids(:,2), 'b*')
        %fig = gcf([0,y])
        saveas(gcf,sprintf("centroides\\im_centroide%04d.png",i));
        hold off
    
    else
     imwrite(seg4,sprintf("centroides\\im_centroide%04d.png",i))
    end

    
    %[y, x] = find(seg4==1);    % y before x (array ordering)
    %plot(mean(x), mean(y), '*');    % x before y (image ordering)
    writeVideo(video,double(imread(sprintf("img_binary_seg\\im_seg%04d.png",i))));
end 
  
     

disp('adios')
  
close(video);
%treure soroll amb gaussiana
%ground(:,:,i) = imread(strcat("groundtruth/",files_ground(1049+i).name));

% labeledImage = bwlabel(double("img_binary_seg/im_seg0153"));
%     blobMeasurements = regionprops(labeledImage, 'Centroid');
%     % We can get the centroids of ALL the blobs into 2 arrays,
%     % one for the centroid x values and one for the centroid y values.
%     allBlobCentroids = [blobMeasurements.Centroid];
%     centroidsX = allBlobCentroids(1:2:end-1);
%     centroidsY = allBlobCentroids(2:2:end);
%     % Put the labels on the rgb labeled image also.
%     for k = 1 : 2 % Loop through all blobs.
%         %if isempty(centroidsY) == 1 || isempty(centroidsX) == 1
%           plot(centroidsX(k), centroidsY(k), 'b*', 'MarkerSize', 15);
%           if k == 1
%               hold on;
%           end
%           text(centroidsX(k) + 10, centroidsY(k),...
%                num2str(k), 'FontSize', fontSize, 'FontWeight', 'Bold');
%         %end
%     end
%% extracció preses grans amb clusters + codi kmeans a ma
he = I;
numColors = 2;
L = imsegkmeans(he,numColors);
B = labeloverlay(he,L);

lab_he = rgb2lab(he);

ab = lab_he(:,:,2:3);
ab = im2single(ab);
pixel_labels = imsegkmeans(ab,numColors,"NumAttempts",3);

mask1 = pixel_labels == 1;
cluster1 = he.*uint8(mask1);
figure, imshow(cluster1)
title("Objects in Cluster 1");

%---------------------preses colors; surten només grans; k= 2------------
mask2 = pixel_labels == 2;
cluster2 = he.*uint8(mask2);
figure, imshow(cluster2)
title("Objects in Cluster 2");
%-------------------------------------------------------------------
mask3 = pixel_labels == 3;
cluster3 = he.*uint8(mask3);
figure, imshow(cluster3)
title("Objects in Cluster 3");


%% -------------deteccion contornos -> piezas pequeñas---------------
I = imread('paret_cut.jpg');
grayImage = rgb2gray(I);
%a = imcontour(grayImage); 
BW1 = edge(grayImage,'sobel')>0.5;
BW2 = edge(grayImage,'canny')<0.9;
figure, imshow(BW1),title('Sobel Filter');
figure, imshow(BW2),title('Canny Filter');

%% bounding box


% Ibw = ~im2bw(im_b,graythresh(im_b));
% Ifill = imfill(Ibw,'holes');
% Iarea = bwareaopen(Ifill,100);
% Ifinal = bwlabel(Iarea);
% stat = regionprops(Ifinal,'boundingbox');
% imshow(I); hold on;
% for cnt = 1 : numel(stat)
%     bb = stat(cnt).BoundingBox;
%     rectangle('position',bb,'edgecolor','r','linewidth',2);
% end

for i=200:205
    im = imread(sprintf("img_binary_seg/im_seg%04d.png",i))
    Ifill = imfill(im,'holes');
    Iarea = bwareaopen(Ifill,100);
    Ifinal = bwlabel(Iarea);
    stat = regionprops(Ifinal,'boundingbox');
    figure(); imshow(I); hold on;
    for cnt = 1 : numel(stat)
        bb = stat(cnt).BoundingBox;
        rectangle('position',bb,'edgecolor','r','linewidth',2);
    end
end
