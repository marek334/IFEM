%% Octahedron quad sphere
%% Input triangles

% vertices
Pin = zeros(10,3);

Pin(2:5,1) = 1:4;
Pin(7:10,1) = 1:4;
Pin(6:10,2) = 1;

% faces
TriangIn = [1,7,6; 1,2,7; 2,3,7; 3,8,7; 3,9,8; 3,4,9; 4,5,9; 5,10,9];

%% Output triangles
% vertices
Pout = zeros(6,3);

Pout(1,:) = [-1,0,0];
Pout(2,:) = [ 1,0,0];
Pout(3,:) = [0,-1,0];
Pout(4,:) = [0, 1,0];
Pout(5,:) = [0,0,-1];
Pout(6,:) = [0,0, 1];

% faces
TriangOut = [2,4,6; 2,5,4; 5,1,4; 1,6,4; 1,3,6; 1,5,3; 5,2,3; 2,6,3];

%% Input mesh
% Vertices

for k = 1:10 % mesh level
    time = tic;
    n = 2^k; % number of divisions in y direction
    NumOfPts = (4*n+1)*(n+1);
    
    x = linspace(0,4,4*n+1);
    y = linspace(0,1,n+1);
    [X,Y] = meshgrid(x,y);
    Vin = [reshape(X',[],1),reshape(Y',[],1),zeros(NumOfPts,1)];
    
    % Communication
    Communication = [(2*n+2:4*n)',(2*n:-1:2)']; % bottom edge of squares 3 & 4
    Communication = [Communication;[(4*n+1:4*n+1:NumOfPts-4*n-1)',(1:4*n+1:NumOfPts-8*n-1)']]; % right edge of square 4
    Communication = [Communication;[(NumOfPts-3*n+1:NumOfPts-2*n)',(NumOfPts-3*n-1:-1:NumOfPts-4*n)']]; % top edge of square 2
    Communication = [Communication;[(NumOfPts-n+1:NumOfPts-1)',(NumOfPts-n-1:-1:NumOfPts-2*n+1)']]; % top edge of square 4
    Communication = [Communication;[NumOfPts,NumOfPts-4*n]]; % top right corner

    % Triange IDs
    FtriangID = zeros(NumOfPts,1);
    FtriangID( Vin(:,2) >= Vin(:,1) ) = 1;
    FtriangID( Vin(:,2) < Vin(:,1) & Vin(:,1) <= 1) = 2;
    FtriangID( Vin(:,1) > 1 & Vin(:,2) <= 2 - Vin(:,1)) = 3;
    FtriangID( Vin(:,2) > 2 - Vin(:,1) & Vin(:,1) <= 2) = 4;
    FtriangID( Vin(:,1) > 2 & Vin(:,2) >= Vin(:,1) - 2) = 5;
    FtriangID( Vin(:,2) < Vin(:,1) - 2 & Vin(:,1) <= 3) = 6;
    FtriangID( Vin(:,1) > 3 & Vin(:,2) <= 4 - Vin(:,1)) = 7;
    FtriangID( Vin(:,2) > 4 - Vin(:,1) ) = 8;

    NumOfFaces = 4*n^2;
    Faces = zeros(NumOfFaces,4);

    % bottom left vertex
    ptIDs = 1:4*n^2+(n-1);
    ptIDs(4*n+1:4*n+1:end) = [];
    Faces(1:NumOfFaces,1) = ptIDs;
    % bottom right vertex
    Faces(1:NumOfFaces,2) = Faces(1:NumOfFaces,1) + 1;
    % top left vertex
    ptIDs = 4*n+2:NumOfPts-1;
    ptIDs(4*n+1:4*n+1:end) = [];
    Faces(1:NumOfFaces,4) = ptIDs;
    % top right vertex
    Faces(1:NumOfFaces,3) = Faces(1:NumOfFaces,4) + 1;

    %% Find transformations and transform mesh vertices
    % Transformation to octahedron
    Vout = zeros(NumOfPts,3);

    for id = 1:8
        [T1,T2,M] = triangleTransf(Pin(TriangIn(id,:),:)',Pout(TriangOut(id,:),:)'); % find triangle transformation
        Vout(FtriangID==id,:) = (M*(Vin(FtriangID==id,:)' - T1) + T2)'; % transform mesh vertices
    end

    % Transformation from octahedron to unit sphere
    Vout = Vout./vecnorm(Vout,2,2);

    [L, B,r] = cart2sph(Vout(:,1),Vout(:,2),Vout(:,3)) ;
Vout(:,2)=(L/pi+1)*180;
Vout(:,1)=(B/pi)*180;
Vout(:,3)=0.0;

    time = toc(time);
    fprintf('quad_ellipsoid_WGS84_%d computed in %0.2f s size %d %d\n',k,time,size(Vin))
    
    %% Export
    time = tic;
    % export mesh
    fileID = fopen("quad_ellipsoid_WGS84_BL0_"+string(k)+".dat",'w');
    fprintf(fileID,'%.6f %.6f %.1f\n',Vout');
    fclose(fileID);
    
    time = toc(time);
    fprintf('quad_ellipsoid_WGS84_%d.ply file written in %0.2f s\n',k,time)
end